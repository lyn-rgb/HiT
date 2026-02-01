# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
import warnings
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
torch.set_num_threads(1)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image, PngImagePlugin
from copy import deepcopy
from contextlib import nullcontext
from glob import glob
from time import time
import argparse
import logging
import os
import math
import signal

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from vae import AutoencoderKL
from train_utils import parse_transport_args
from data_utils import ParquetImageDataset, VAELatentDataset
from muon import MuonWithAuxAdam
import wandb_utils
from log_utils import TrainingLogger
from snapshot_utils import snapshot_code
import tb_utils

warnings.filterwarnings("ignore")

PngImagePlugin.MAX_TEXT_CHUNK = (1024 ** 2) * 64    # to avoid image load error `Decompressed Data Too Large`
Image.MAX_IMAGE_PIXELS = None  # Disable PIL decompression bomb limit; handle large images explicitly.


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f).load()
        return img.convert("RGB")

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank, log_all_ranks=False):
    return TrainingLogger(logging_dir, rank=rank, name="sit", log_all_ranks=log_all_ranks)

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

def _build_muon_param_groups(model, args):
    hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
    nonhidden_params = [
        *model.x_embedder.parameters(),
        *model.t_embedder.parameters(),
        *model.y_embedder.parameters(),
        *model.final_layer.parameters(),
    ]
    aux_params = hidden_gains_biases + nonhidden_params
    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
        ),
        dict(
            params=aux_params,
            use_muon=False,
            lr=args.muon_aux_lr,
            betas=tuple(args.muon_aux_betas),
            eps=args.muon_aux_eps,
            weight_decay=args.muon_aux_weight_decay,
        ),
    ]
    return param_groups


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
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
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
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if resume_experiment_dir is not None:
            experiment_dir = resume_experiment_dir
            checkpoint_dir = f"{experiment_dir}/checkpoints"
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
            experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
            experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank, log_all_ranks=args.log_all_ranks)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.run_notes:
            logger.info(f"Run notes: {args.run_notes}")
        if args.save_code:
            snapshot_code(os.getcwd(), f"{experiment_dir}/code")

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
        tb_writer = tb_utils.setup(f"{experiment_dir}/tensorboard", enabled=args.tensorboard)
    else:
        logger = create_logger("results", rank, log_all_ranks=args.log_all_ranks)
        tb_writer = None
        experiment_dir = None
        checkpoint_dir = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        fa_version=args.fa_version,
    )
    if args.grad_checkpoint:
        model.set_grad_checkpointing(True)

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    state_dict = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        args = state_dict["args"]

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae_path = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    if args.amp_dtype == "bf16":
        vae = vae.to(dtype=torch.bfloat16)
    vae_scale = vae.scaling_factor
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.optimizer == "muon":
        param_groups = _build_muon_param_groups(model.module, args)
        opt = MuonWithAuxAdam(param_groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if state_dict is not None:
        opt.load_state_dict(state_dict["opt"])
    use_amp = args.amp_dtype != "fp32"
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_autocast = torch.cuda.amp.autocast if use_amp else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp_dtype == "fp16")

    def _save_last_ckpt():
        if rank != 0 or checkpoint_dir is None:
            return
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
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
    use_latents = args.latent_path is not None
    if use_latents:
        dataset = VAELatentDataset(args.latent_path)
    elif args.data_format == "parquet":
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
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory_device=args.pin_memory_device,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    total_steps = args.epochs * len(loader)
    start_time = time()
    train_start_time = start_time

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            if use_latents:
                mu, logvar, y = batch
                mu = mu.to(device)
                logvar = logvar.to(device)
                y = y.to(device)
                with torch.no_grad():
                    eps = torch.randn_like(mu)
                    x = (mu + eps * torch.exp(0.5 * logvar)).mul_(vae_scale)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad(), amp_autocast(dtype=autocast_dtype):
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(vae_scale)
            model_kwargs = dict(y=y)
            with amp_autocast(dtype=autocast_dtype):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                elapsed = end_time - train_start_time
                avg_steps_per_sec = train_steps / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(0, total_steps - train_steps)
                eta_seconds = remaining_steps / avg_steps_per_sec if avg_steps_per_sec > 0 else 0.0
                eta_str = _format_eta(eta_seconds)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, ETA: {eta_str}"
                )
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                tb_utils.log(
                    tb_writer,
                    { "train/loss": avg_loss, "train/steps_per_sec": steps_per_sec },
                    train_steps,
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad(), amp_autocast(dtype=autocast_dtype):
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / vae_scale).sample
                    out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device, dtype=samples.dtype)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                if rank == 0:
                    out_samples_cpu = out_samples.detach().cpu()
                    sample_dir = os.path.join(experiment_dir, "samples", f"{train_steps:07d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    for idx, img in enumerate(out_samples_cpu):
                        img_path = os.path.join(sample_dir, f"{idx:04d}.png")
                        save_image(
                            img,
                            img_path,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                    logger.info(f"Saved EMA samples to {sample_dir}")
                logger.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    tb_utils.close(tb_writer)
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="imagefolder",
                        choices=["imagefolder", "parquet"])
    parser.add_argument("--parquet-image-key", type=str, default="image")
    parser.add_argument("--parquet-label-key", type=str, default="label")
    parser.add_argument("--latent-path", type=str, default=None,
                        help="Optional root with pre-extracted VAE latents (class subdirs of .pt files).")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
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
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory-device", type=str, default="")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging (rank 0 only)")
    parser.add_argument("--log-all-ranks", action="store_true",
                        help="Log to stdout from all ranks (file logging stays rank 0 only)")
    parser.add_argument("--fa-version", type=int, default=None, choices=[2, 3],
                        help="Force FlashAttention version (2 or 3). Default uses available version.")
    parser.add_argument("--save-code", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a copy of the training code into the experiment folder")
    parser.add_argument("--run-notes", type=str, default="",
                        help="Free-form notes describing the training setup")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to reduce GPU memory usage")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-resume from the most recent checkpoint in results-dir if no --ckpt is provided")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
