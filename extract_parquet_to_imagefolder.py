#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from data_utils import ParquetImageDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract images from parquet files into an ImageFolder-compatible directory."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Parquet file, glob, or directory containing parquet files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root directory for ImageFolder.")
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional subset prefix: train, test, validation (or val).")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--all", action="store_true",
                        help="Extract train/val/test splits into subdirectories.")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Optional starting index for naming output files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max number of samples to extract.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        if args.subset is not None:
            raise ValueError("--all cannot be used with --subset.")
        subsets = [("train", "train"), ("validation", "val"), ("test", "test")]
    else:
        subsets = [(args.subset, None)]

    for subset, subset_dir in subsets:
        dataset = ParquetImageDataset(
            args.input,
            transform=None,
            image_key=args.image_key,
            label_key=args.label_key,
            subset=subset,
        )

        output_root = args.output
        if subset_dir is not None:
            output_root = os.path.join(output_root, subset_dir)
        elif subset:
            subset_lower = subset.lower()
            output_root = os.path.join(output_root, subset_lower if subset_lower != "validation" else "val")
        os.makedirs(output_root, exist_ok=True)

        total = len(dataset)
        limit = args.limit if args.limit is not None else total
        start = args.start_idx
        end = min(total, start + limit)

        num_digits = max(6, len(str(end)))
        for idx in range(start, end):
            image, label = dataset[idx]
            class_dir = os.path.join(output_root, str(label))
            os.makedirs(class_dir, exist_ok=True)
            filename = f"{idx:0{num_digits}d}.png"
            image.save(os.path.join(class_dir, filename))
            if (idx + 1) % 1000 == 0 or idx + 1 == end:
                print(f"[{subset_dir or subset or 'all'}] Saved {idx + 1 - start}/{end - start} images", flush=True)


if __name__ == "__main__":
    main()
