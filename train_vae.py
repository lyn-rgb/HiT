# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for the VAE using PyTorch DDP.
"""
import torch
import warnings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
torch.set_num_threads(1)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image, PngImagePlugin
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from contextlib import nullcontext
from time import time
import argparse
import logging
import os
import signal
import textwrap

from vae import AutoencoderKL
from muon import MuonWithAuxAdam
import wandb_utils
from log_utils import TrainingLogger
from snapshot_utils import snapshot_code
import tb_utils

warnings.filterwarnings("ignore")
from data_utils import ParquetImageDataset
from train_utils import ResumableBatchSampler

PngImagePlugin.MAX_TEXT_CHUNK = (1024 ** 2) * 64    # to avoid image load error `Decompressed Data Too Large`
Image.MAX_IMAGE_PIXELS = None  # Disable PIL decompression bomb limit; handle large images explicitly.


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        return img.convert("RGB")

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank, log_all_ranks=False):
    return TrainingLogger(logging_dir, rank=rank, name="vae", log_all_ranks=log_all_ranks)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _find_latest_checkpoint(results_dir: str) -> str | None:
    candidates = glob(os.path.join(results_dir, "**", "checkpoints", "*.pt"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _format_args(args) -> str:
    items = sorted(vars(args).items(), key=lambda kv: kv[0])
    lines = ["Training configuration:"]
    for key, value in items:
        value_str = repr(value)
        wrapped = textwrap.wrap(value_str, width=72)
        if not wrapped:
            lines.append(f"  - {key}:")
        else:
            lines.append(f"  - {key}: {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"    {cont}")
    return "\n".join(lines)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains the VAE model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    resume_ckpt = None
    resume_experiment_dir = None
    if args.auto_resume and args.ckpt is None:
        resume_ckpt = _find_latest_checkpoint(args.results_dir)
        if resume_ckpt is not None:
            args.ckpt = resume_ckpt
            resume_experiment_dir = os.path.dirname(os.path.dirname(resume_ckpt))

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if resume_experiment_dir is not None:
            experiment_dir = resume_experiment_dir
            checkpoint_dir = f"{experiment_dir}/checkpoints"
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_name = f"{experiment_index:03d}-vae"
            experiment_dir = f"{args.results_dir}/{experiment_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank, log_all_ranks=args.log_all_ranks)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(_format_args(args))
        if args.run_notes:
            logger.info(f"Run notes: {args.run_notes}")
        if args.save_code:
            snapshot_code(os.getcwd(), f"{experiment_dir}/code")

        if args.wandb and entity and project:
            entity = os.environ.get("ENTITY")
            project = os.environ.get("PROJECT")
            wandb_utils.initialize(args, entity, experiment_name, project)
        tb_writer = tb_utils.setup(f"{experiment_dir}/tensorboard", enabled=args.tensorboard)
    else:
        logger = create_logger("results", rank, log_all_ranks=args.log_all_ranks)
        tb_writer = None
        experiment_dir = None
        checkpoint_dir = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    vae_path = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path)
    resume_train_steps = 0
    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location="cpu")
        vae.load_state_dict(state["model"])
        resume_train_steps = state.get("train_steps", 0)
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.ckpt}")
            logger.info(f"Resume train_steps: {resume_train_steps}")
    vae = DDP(vae.to(device), device_ids=[device])
    ema = deepcopy(vae.module).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    update_ema(ema, vae.module, decay=0)
    ema.eval()
    vae.train()
    logger.info(f"VAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Setup optimizer:
    if args.optimizer == "muon":
        hidden_weights = [p for p in vae.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in vae.parameters() if p.ndim < 2]
        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=args.muon_lr,
                momentum=args.muon_momentum,
                weight_decay=args.muon_weight_decay,
            ),
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=args.muon_aux_lr,
                betas=tuple(args.muon_aux_betas),
                eps=args.muon_aux_eps,
                weight_decay=args.muon_aux_weight_decay,
            ),
        ]
        opt = MuonWithAuxAdam(param_groups)
    else:
        opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0)
    use_amp = args.amp_dtype != "fp32"
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    def amp_autocast():
        return torch.cuda.amp.autocast(dtype=autocast_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp_dtype == "fp16")
    if args.ckpt is not None and "opt" in state:
        opt.load_state_dict(state["opt"])
    if args.ckpt is not None and "ema" in state:
        ema.load_state_dict(state["ema"])

    def _save_last_ckpt():
        if rank != 0 or checkpoint_dir is None:
            return
        checkpoint = {
            "model": vae.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
        }
        last_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint to {last_path}")

    def _handle_signal(signum, _frame):
        logger.info(f"Received signal {signum}. Saving last checkpoint...")
        _save_last_ckpt()
        cleanup()
        raise SystemExit(0)

    if rank == 0:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if args.data_format == "parquet":
        dataset = ParquetImageDataset(
            args.data_path,
            transform=transform,
            image_key=args.parquet_image_key,
            label_key=args.parquet_label_key,
        )
    else:
        dataset = ImageFolder(args.data_path, transform=transform, loader=pil_loader)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    batch_sampler = BatchSampler(sampler, batch_size=local_batch_size, drop_last=True)
    resume_batch_sampler = ResumableBatchSampler(batch_sampler)
    loader = DataLoader(
        dataset,
        batch_sampler=resume_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory_device=args.pin_memory_device
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Variables for monitoring/logging purposes:
    train_steps = resume_train_steps
    log_steps = 0
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    steps_per_epoch = len(batch_sampler)
    total_steps = args.epochs * steps_per_epoch
    start_epoch = train_steps // steps_per_epoch
    start_step_in_epoch = train_steps % steps_per_epoch
    start_time = time()
    train_start_time = start_time
    if rank == 0:
        logger.info(f"Total training steps: {total_steps:,}")
        if train_steps > 0:
            logger.info(
                f"Resuming at epoch {start_epoch} step {start_step_in_epoch} "
                f"(global step {train_steps})"
            )

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        resume_batch_sampler.set_start_step(start_step_in_epoch if epoch == start_epoch else 0)
        for x, _ in loader:
            x = x.to(device)
            with amp_autocast():
                posterior = vae(x, mode="encode").latent_dist
                z = posterior.sample()
                recon = vae(mode="decode", latent=z).sample
                recon_loss = torch.mean((recon - x) ** 2)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + args.kl_weight * kl_loss

            opt.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            update_ema(ema, vae.module, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                elapsed = end_time - train_start_time
                avg_steps_per_sec = train_steps / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(0, total_steps - train_steps)
                eta_seconds = remaining_steps / avg_steps_per_sec if avg_steps_per_sec > 0 else 0.0
                eta_str = _format_eta(eta_seconds)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_recon = torch.tensor(running_recon / log_steps, device=device)
                avg_kl = torch.tensor(running_kl / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_recon, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_kl, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                avg_loss = avg_loss.item() / world_size
                avg_recon = avg_recon.item() / world_size
                avg_kl = avg_kl.item() / world_size

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, ETA: {eta_str}"
                )
                if args.wandb:
                    wandb_utils.log(
                        {
                            "loss": avg_loss,
                            "recon": avg_recon,
                            "kl": avg_kl,
                            "steps/sec": steps_per_sec
                        },
                        step=train_steps
                    )
                tb_utils.log(
                    tb_writer,
                    {
                        "train/loss": avg_loss,
                        "train/recon": avg_recon,
                        "train/kl": avg_kl,
                        "train/steps_per_sec": steps_per_sec,
                    },
                    train_steps,
                )
                running_loss = 0.0
                running_recon = 0.0
                running_kl = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    recon_dir = os.path.join(experiment_dir, "recon", f"{train_steps:07d}")
                    os.makedirs(recon_dir, exist_ok=True)
                    x_cpu = x.detach().cpu()
                    recon_cpu = recon.detach().cpu()
                    for idx in range(x_cpu.shape[0]):
                        combined = torch.cat([x_cpu[idx], recon_cpu[idx]], dim=2)
                        save_image(
                            combined,
                            os.path.join(recon_dir, f"input_recon_{idx:04d}.png"),
                            normalize=True,
                            value_range=(-1, 1),
                        )
                    logger.info(f"Saved reconstructions to {recon_dir}")

            # Save VAE checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": vae.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    vae.eval()
    logger.info("Done!")
    tb_utils.close(tb_writer)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="imagefolder",
                        choices=["imagefolder", "parquet"])
    parser.add_argument("--parquet-image-key", type=str, default="image")
    parser.add_argument("--parquet-label-key", type=str, default="label")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local path to a diffusers-format VAE folder")
    parser.add_argument("--amp-dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Enable mixed precision training with fp16 or bf16")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "muon"],
                        help="Optimizer to use for training")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Learning rate for Muon parameter group")
    parser.add_argument("--muon-weight-decay", type=float, default=0.01,
                        help="Weight decay for Muon parameter group")
    parser.add_argument("--muon-momentum", type=float, default=0.95,
                        help="Momentum for Muon parameter group")
    parser.add_argument("--muon-aux-lr", type=float, default=3e-4,
                        help="Learning rate for Adam aux parameter group")
    parser.add_argument("--muon-aux-betas", type=float, nargs=2, default=(0.9, 0.95),
                        metavar=("BETA1", "BETA2"),
                        help="Betas for Adam aux parameter group")
    parser.add_argument("--muon-aux-eps", type=float, default=1e-10,
                        help="Epsilon for Adam aux parameter group")
    parser.add_argument("--muon-aux-weight-decay", type=float, default=0.01,
                        help="Weight decay for Adam aux parameter group")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory-device", type=str, default="")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=50_000)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging (rank 0 only)")
    parser.add_argument("--log-all-ranks", action="store_true",
                        help="Log to stdout from all ranks (file logging stays rank 0 only)")
    parser.add_argument("--save-code", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a copy of the training code into the experiment folder")
    parser.add_argument("--run-notes", type=str, default="",
                        help="Free-form notes describing the training setup")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a VAE checkpoint to resume from")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False,
                        help="Auto-resume from the most recent checkpoint in results-dir if no --ckpt is provided")

    args = parser.parse_args()
    main(args)
