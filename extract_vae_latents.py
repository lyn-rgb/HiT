#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import multiprocessing as mp
from time import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image, PngImagePlugin
from vae import AutoencoderKL

PngImagePlugin.MAX_TEXT_CHUNK = (1024 ** 2) * 64    # to avoid image load error `Decompressed Data Too Large`
Image.MAX_IMAGE_PIXELS = None  # Disable PIL decompression bomb limit; handle large images explicitly.


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f).load()
        return img.convert("RGB")


def _writer_loop(q):
    while True:
        item = q.get()
        if item is None:
            break
        out_path, mu_item, logvar_item = item
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"mu": mu_item, "logvar": logvar_item}, out_path)


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        attempts = 0
        last_err = None
        while attempts < 5:
            try:
                image, label = super().__getitem__(index)
                path, _ = self.samples[index]
                return image, label, path
            except Exception as e:
                path, _ = self.samples[index]
                print(f"[DataError] index={index} path={path} err={e}", flush=True)
                last_err = e
                attempts += 1
                index = (index + 1) % len(self.samples)
        raise last_err


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-extract VAE latents (mu/logvar) into ImageFolder-like directories."
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="ImageFolder root.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root for .pt files (class subdirs).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local path to a diffusers-format VAE folder")
    parser.add_argument("--amp-dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-writers", type=int, default=1,
                        help="Number of writer processes for async disk I/O.")
    parser.add_argument("--queue-size", type=int, default=256,
                        help="Max queued write items before producers block.")
    return parser.parse_args()


def main(args):
    assert torch.cuda.is_available(), "Extraction requires at least one GPU."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolderWithPaths(args.data_path, transform=transform, loader=pil_loader)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae_path = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    use_amp = args.amp_dtype != "fp32"
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if args.amp_dtype == "bf16":
        vae = vae.to(dtype=torch.bfloat16)
    vae.eval()

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        start_time = time()
    else:
        start_time = None

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=args.queue_size)

    writers = []
    for _ in range(max(1, args.num_writers)):
        writer = ctx.Process(target=_writer_loop, args=(queue,))
        writer.daemon = True
        writer.start()
        writers.append(writer)

    log_every = 100
    processed = 0
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            images = images.to(device)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    posterior = vae.encode(images).latent_dist
                    mu = posterior.mean.detach().cpu()
                    logvar = posterior.logvar.detach().cpu()
            else:
                posterior = vae.encode(images).latent_dist
                mu = posterior.mean.detach().cpu()
                logvar = posterior.logvar.detach().cpu()

            for i in range(mu.shape[0]):
                label = int(labels[i])
                class_name = dataset.classes[label]
                rel_name = os.path.splitext(os.path.basename(paths[i]))[0] + ".pt"
                class_dir = os.path.join(args.output, class_name)
                out_path = os.path.join(class_dir, rel_name)
                queue.put((out_path, mu[i], logvar[i]))

            processed += mu.shape[0]
            if rank == 0 and (batch_idx + 1) % log_every == 0:
                elapsed = time() - start_time if start_time is not None else 0.0
                imgs_per_sec = processed / elapsed if elapsed > 0 else 0.0
                total = len(dataset)
                remaining = max(0, total - processed)
                eta = _format_eta(remaining / imgs_per_sec if imgs_per_sec > 0 else 0.0)
                print(
                    f"Processed {batch_idx + 1} batches "
                    f"({processed} images), {imgs_per_sec:.2f} img/s, ETA {eta}",
                    flush=True,
                )

    for _ in writers:
        queue.put(None)
    for writer in writers:
        writer.join()
    dist.barrier()
    if rank == 0:
        elapsed = time() - start_time if start_time is not None else 0.0
        print(f"Done. Elapsed {elapsed:.2f} seconds.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
