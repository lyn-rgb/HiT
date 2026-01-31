#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from contextlib import nullcontext

from vae import AutoencoderKL


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
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


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
    dataset = ImageFolderWithPaths(args.data_path, transform=transform)
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
                os.makedirs(class_dir, exist_ok=True)
                out_path = os.path.join(class_dir, rel_name)
                torch.save({"mu": mu[i], "logvar": logvar[i]}, out_path)

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1} batches", flush=True)

    dist.barrier()
    if rank == 0:
        print("Done.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
