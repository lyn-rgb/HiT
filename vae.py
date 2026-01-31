# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal AutoencoderKL implementation compatible with SD VAE weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import attention


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class AutoencoderKLOutput:
    latent_dist: "DiagonalGaussianDistribution"


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.mean * self.mean + self.var - 1.0 - self.logvar, dim=(1, 2, 3))


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        if use_conv:
            self.conv = nn.Conv2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name

        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            self.conv = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        pre_norm: bool = True,
        use_in_shortcut: Optional[bool] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm: {self.time_embedding_norm}")
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = (in_channels != out_channels) if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None and temb is not None:
            if self.time_embedding_norm == "default":
                temb = self.nonlinearity(temb)
                hidden_states = hidden_states + self.time_emb_proj(temb)[:, :, None, None]
            elif self.time_embedding_norm == "scale_shift":
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + scale) + shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return (input_tensor + hidden_states) / self.output_scale_factor


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        upcast_softmax: bool = True,
        norm_num_groups: Optional[int] = None,
        eps: float = 1e-6,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        fa_version: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.upcast_softmax = upcast_softmax
        self.residual_connection = residual_connection
        self.rescale_output_factor = rescale_output_factor
        self.fa_version = fa_version

        self.group_norm = None
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=eps, affine=True)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(dropout)])

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        if hidden_states.dim() == 4:
            b, c, h, w = hidden_states.shape
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states)
            hidden_states = hidden_states.view(b, c, h * w).permute(0, 2, 1)
        else:
            b, seq_len, c = hidden_states.shape
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        if hidden_states.is_cuda and q.size(-1) <= 256:
            dropout_p = self.to_out[1].p if self.training else 0.0
            dtype = hidden_states.dtype if hidden_states.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            attn = attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
                dtype=dtype,
                fa_version=self.fa_version,
            )
            hidden_states = attn.transpose(1, 2)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.upcast_softmax:
                attn_scores = attn_scores.float()
            attn_probs = torch.softmax(attn_scores, dim=-1).to(v.dtype)
            hidden_states = torch.matmul(attn_probs, v)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(b, -1, self.heads * self.dim_head)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if residual.dim() == 4:
            hidden_states = hidden_states.permute(0, 2, 1).view(b, c, h, w)

        if self.residual_connection:
            hidden_states = (hidden_states + residual) / self.rescale_output_factor
        else:
            hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        self.add_attention = add_attention
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=None,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=0.0,
                time_embedding_norm="default",
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(1):
            if add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=1.0,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding)]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


def get_down_block(down_block_type: str, **kwargs) -> nn.Module:
    if down_block_type != "DownEncoderBlock2D":
        raise ValueError(f"Unsupported down_block_type: {down_block_type}")
    return DownEncoderBlock2D(**kwargs)


def get_up_block(up_block_type: str, **kwargs) -> nn.Module:
    if up_block_type != "UpDecoderBlock2D":
        raise ValueError(f"Unsupported up_block_type: {up_block_type}")
    return UpDecoderBlock2D(**kwargs)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_block_types: Tuple[str, ...],
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        act_fn: str,
        norm_num_groups: int,
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample, None)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_block_types: Tuple[str, ...],
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
        act_fn: str,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.up_blocks = nn.ModuleList([])

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, None)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        mid_block_add_attention: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "layers_per_block": layers_per_block,
            "act_fn": act_fn,
            "latent_channels": latent_channels,
            "norm_num_groups": norm_num_groups,
            "sample_size": sample_size,
            "scaling_factor": scaling_factor,
            "mid_block_add_attention": mid_block_add_attention,
            "use_quant_conv": use_quant_conv,
            "use_post_quant_conv": use_post_quant_conv,
        }

        self.scaling_factor = scaling_factor
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return DecoderOutput(sample=dec)

    def forward(self, sample: torch.Tensor) -> DecoderOutput:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample()
        return self.decode(z)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "AutoencoderKL":
        resolved_path = _resolve_pretrained_path(pretrained_path)
        config_path, weights_path = _resolve_config_and_weights(resolved_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            down_block_types=tuple(config.get("down_block_types", ("DownEncoderBlock2D",))),
            up_block_types=tuple(config.get("up_block_types", ("UpDecoderBlock2D",))),
            block_out_channels=tuple(config.get("block_out_channels", (64,))),
            layers_per_block=config.get("layers_per_block", 1),
            act_fn=config.get("act_fn", "silu"),
            latent_channels=config.get("latent_channels", 4),
            norm_num_groups=config.get("norm_num_groups", 32),
            sample_size=config.get("sample_size", 32),
            scaling_factor=config.get("scaling_factor", 0.18215),
            mid_block_add_attention=config.get("mid_block_add_attention", True),
            use_quant_conv=config.get("use_quant_conv", True),
            use_post_quant_conv=config.get("use_post_quant_conv", True),
        )
        state_dict = _load_state_dict(weights_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _remap_attn_keys(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model


@dataclass
class HierarchicalVAEOutput:
    recon: torch.Tensor
    posterior: DiagonalGaussianDistribution
    mu: torch.Tensor
    sub_posteriors: Tuple[DiagonalGaussianDistribution, ...]
    sub_mus: Tuple[torch.Tensor, ...]
    sub_recons: Tuple[torch.Tensor, ...]


class HierarchicalVAE(nn.Module):
    def __init__(
        self,
        base_vae: AutoencoderKL,
        num_levels: int = 2,
        sub_vae_configs: Optional[Tuple[dict, ...]] = None,
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if sub_vae_configs is not None and len(sub_vae_configs) != num_levels - 1:
            raise ValueError("sub_vae_configs must have num_levels - 1 entries when provided.")

        self.base_vae = base_vae
        self.sub_vaes = nn.ModuleList()
        self.num_levels = num_levels

        prev_channels = base_vae.config["latent_channels"]
        for i in range(num_levels - 1):
            if sub_vae_configs is None:
                sub_config = self._default_sub_vae_config(base_vae.config, prev_channels)
            else:
                sub_config = dict(sub_vae_configs[i])
                sub_config.setdefault("in_channels", prev_channels)
                sub_config.setdefault("out_channels", prev_channels)
                sub_config.setdefault("latent_channels", prev_channels)
            sub_vae = AutoencoderKL(**sub_config)
            self.sub_vaes.append(sub_vae)
            prev_channels = sub_vae.config["latent_channels"]

        self.config = {
            "num_levels": num_levels,
            "base_config": dict(base_vae.config),
            "sub_vae_configs": [dict(m.config) for m in self.sub_vaes],
        }

    @staticmethod
    def _default_sub_vae_config(base_config: dict, in_channels: int) -> dict:
        # Shallow VAE to avoid excessive downsampling on latent feature maps.
        return {
            "in_channels": in_channels,
            "out_channels": in_channels,
            "down_block_types": ("DownEncoderBlock2D",),
            "up_block_types": ("UpDecoderBlock2D",),
            "block_out_channels": (base_config["block_out_channels"][0],),
            "layers_per_block": base_config["layers_per_block"],
            "act_fn": base_config["act_fn"],
            "latent_channels": in_channels,
            "norm_num_groups": base_config["norm_num_groups"],
            "sample_size": max(1, int(base_config.get("sample_size", 32) // 8)),
            "scaling_factor": base_config["scaling_factor"],
            "mid_block_add_attention": base_config["mid_block_add_attention"],
            "use_quant_conv": base_config["use_quant_conv"],
            "use_post_quant_conv": base_config["use_post_quant_conv"],
        }

    def forward(self, sample: torch.Tensor) -> HierarchicalVAEOutput:
        posterior = self.base_vae.encode(sample).latent_dist
        mu = posterior.mean
        z = posterior.sample()
        recon = self.base_vae.decode(z).sample

        sub_posteriors = []
        sub_mus = []
        sub_recons = []
        input_mu = mu
        for sub_vae in self.sub_vaes:
            sub_posterior = sub_vae.encode(input_mu).latent_dist
            sub_posteriors.append(sub_posterior)
            sub_mus.append(sub_posterior.mean)
            sub_z = sub_posterior.sample()
            sub_recon = sub_vae.decode(sub_z).sample
            sub_recons.append(sub_recon)
            input_mu = sub_posterior.mean

        return HierarchicalVAEOutput(
            recon=recon,
            posterior=posterior,
            mu=mu,
            sub_posteriors=tuple(sub_posteriors),
            sub_mus=tuple(sub_mus),
            sub_recons=tuple(sub_recons),
        )


def _resolve_pretrained_path(pretrained_path: str) -> str:
    if os.path.isdir(pretrained_path):
        return pretrained_path

    cache_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hub_dir = os.path.join(cache_home, "hub")
    if "/" in pretrained_path and os.path.isdir(hub_dir):
        repo_dir = os.path.join(hub_dir, "models--" + pretrained_path.replace("/", "--"))
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            for snapshot in reversed(snapshots):
                snap_path = os.path.join(snapshots_dir, snapshot)
                vae_path = os.path.join(snap_path, "vae")
                if os.path.isdir(vae_path):
                    return vae_path
                if os.path.isdir(snap_path):
                    return snap_path

    raise FileNotFoundError(f"Could not resolve pretrained VAE path: {pretrained_path}")


def _resolve_config_and_weights(pretrained_path: str) -> Tuple[str, str]:
    config_path = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {pretrained_path}")

    candidates = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    ]
    for name in candidates:
        weights_path = os.path.join(pretrained_path, name)
        if os.path.isfile(weights_path):
            return config_path, weights_path

    raise FileNotFoundError(f"Missing model weights in {pretrained_path}")


def _remap_attn_keys(state_dict: dict) -> dict:
    mapping = (
        (".query.", ".to_q."),
        (".key.", ".to_k."),
        (".value.", ".to_v."),
        (".proj_attn.", ".to_out.0."),
    )
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for src, dst in mapping:
            if src in new_key:
                new_key = new_key.replace(src, dst)
        if new_key in state_dict and new_key != key:
            # Prefer already-correct keys in the checkpoint.
            continue
        remapped[new_key] = value
    return remapped


def _load_state_dict(weights_path: str) -> dict:
    if weights_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required to load .safetensors weights") from exc
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")
