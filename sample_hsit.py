# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample images from a hierarchical SiT using chained per-level models.
"""
import argparse
import math
import os
import sys
from time import time
import json

import torch
from torchvision.utils import save_image

from models import HierarchicalSiT_models
from transport import create_transport, Sampler
from vae import AutoencoderKL, HierarchicalVAE


def _parse_class_labels(value):
    if value is None:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if not parts:
        return None
    return [int(v) for v in parts]


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def compute_mu_pyramid(hvae, x):
    posterior = hvae.base_vae.encode(x).latent_dist
    mu = posterior.mean
    mus = [mu]
    for sub_vae in hvae.sub_vaes:
        sub_posterior = sub_vae.encode(mu).latent_dist
        mu = sub_posterior.mean
        mus.append(mu)
    return mus


def load_json_arg(value, name):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON string or a path to a JSON file.") from exc




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_levels is None:
        raise ValueError("--ckpt-levels is required (comma-separated list).")
    ckpt_levels = [p.strip() for p in args.ckpt_levels.split(",") if p.strip()]
    if len(ckpt_levels) != args.num_levels:
        raise ValueError("--ckpt-levels must have num_levels entries.")

    class_labels = _parse_class_labels(args.class_labels)
    if class_labels is not None:
        num_samples = len(class_labels)
    else:
        num_samples = args.num_samples
        class_labels = torch.randint(0, args.num_classes, (num_samples,)).tolist()

    y = torch.tensor(class_labels, device=device)

    vae_path = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    base_vae = AutoencoderKL.from_pretrained(vae_path)
    sub_vae_configs = load_json_arg(args.sub_vae_configs, "sub_vae_configs")
    hvae = HierarchicalVAE(base_vae=base_vae, num_levels=args.num_levels, sub_vae_configs=sub_vae_configs)
    if args.hvae_ckpt is not None:
        state = torch.load(args.hvae_ckpt, map_location="cpu")
        hvae.load_state_dict(state["model"] if "model" in state else state)
    hvae = hvae.to(device)
    hvae.eval()

    # Probe latent shapes using dummy input.
    with torch.no_grad():
        dummy = torch.zeros(num_samples, 3, args.image_size, args.image_size, device=device)
        mus = compute_mu_pyramid(hvae, dummy)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    transport_sampler = Sampler(transport)
    if args.mode == "ODE":
        sample_fn = transport_sampler.sample_ode(
            atol=args.atol,
            rtol=args.rtol,
            method=args.sampling_method,
            num_steps=args.num_sampling_steps,
        )
    else:
        sample_fn = transport_sampler.sample_sde(
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    prev_latent = None
    start_time = time()
    for level in range(args.num_levels):
        mu = mus[level]
        cond_mu = mus[level - 1] if level > 0 else None
        input_size = mu.shape[-1]
        in_channels = mu.shape[1]
        cond_input_size = cond_mu.shape[-1] if cond_mu is not None else input_size
        cond_in_channels = cond_mu.shape[1] if cond_mu is not None else in_channels

        model = HierarchicalSiT_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            num_classes=args.num_classes,
            fa_version=args.fa_version,
            use_flash_attn=args.use_flash_attn,
            cond_input_size=cond_input_size,
            cond_in_channels=cond_in_channels,
        ).to(device)
        ckpt = torch.load(ckpt_levels[level], map_location="cpu")
        if args.use_ema and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
        else:
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()

        z = torch.randn(num_samples, in_channels, input_size, input_size, device=device)
        cond = prev_latent

        if args.cfg_scale != 1.0:
            z = torch.cat([z, z], dim=0)
            y_null = torch.tensor([args.num_classes] * num_samples, device=device)
            y_cfg = torch.cat([y, y_null], dim=0)
            if cond is not None:
                cond = torch.cat([cond, cond], dim=0)
            model_kwargs = {"y": y_cfg, "cfg_scale": args.cfg_scale}
            if cond is not None:
                model_kwargs["cond"] = cond
            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)
        else:
            model_kwargs = {"y": y}
            if cond is not None:
                model_kwargs["cond"] = cond
            samples = sample_fn(z, model.forward, **model_kwargs)[-1]

        prev_latent = samples
        elapsed = time() - start_time
        eta = _format_eta((args.num_levels - level - 1) * elapsed / max(1, level + 1))
        print(f"Sampled level {level} in {elapsed:.2f}s, ETA {eta}.")

    decoded = hvae.decode_from_level(prev_latent, args.num_levels - 1)

    os.makedirs(args.outdir, exist_ok=True)
    nrow = int(math.sqrt(num_samples)) if int(math.sqrt(num_samples)) ** 2 == num_samples else 4
    out_path = os.path.join(args.outdir, args.outname)
    save_image(decoded, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sample_hsit.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode.startswith("--") or mode not in ["ODE", "SDE"]:
        print("Usage: sample_hsit.py <mode> [options]")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(HierarchicalSiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local path to a diffusers-format VAE folder")
    parser.add_argument("--hvae-ckpt", type=str, default=None,
                        help="Optional path to a hierarchical VAE checkpoint")
    parser.add_argument("--sub-vae-configs", type=str, default=None,
                        help="JSON string or path to JSON file with per-sub-vae configs")
    parser.add_argument("--ckpt-levels", type=str, required=True,
                        help="Comma-separated list of HSIT checkpoints per level")
    parser.add_argument("--num-levels", type=int, default=2)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--class-labels", type=str, default=None,
                        help="Comma-separated class labels (overrides num-samples)")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fa-version", type=int, default=None, choices=[2, 3],
                        help="Select FlashAttention version (2 or 3) when enabled.")
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable FlashAttention when available (off by default).")
    parser.add_argument("--outdir", type=str, default="samples")
    parser.add_argument("--outname", type=str, default="hsit_sample.png")

    # Transport/sampling args
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--sampling-method", type=str, default="dopri5")
    parser.add_argument("--diffusion-form", type=str, default="GVP")
    parser.add_argument("--diffusion-norm", type=float, default=1.0)
    parser.add_argument("--last-step", type=str, default="Mean")
    parser.add_argument("--last-step-size", type=float, default=0.04)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--path-type", type=str, default="VE")
    parser.add_argument("--prediction", type=str, default="velocity")
    parser.add_argument("--loss-weight", type=str, default="velocity")
    parser.add_argument("--train-eps", type=float, default=1e-3)
    parser.add_argument("--sample-eps", type=float, default=1e-3)

    args = parser.parse_args(sys.argv[2:])
    args.mode = mode
    main(args)
