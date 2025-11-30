#!/usr/bin/env python
"""Compute mean and std of a dataset for normalization.

Usage:
    poetry run python scripts/compute_dataset_stats.py
    poetry run python scripts/compute_dataset_stats.py --data-dir data/plantsegv3/images/train
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def compute_mean_std(image_dir: Path) -> tuple[list[float], list[float]]:
    """Compute channel-wise mean and std for all images in directory.

    Args:
        image_dir: Directory containing images.

    Returns:
        Tuple of (mean, std) as lists of 3 floats (RGB).
    """
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    to_tensor = transforms.ToTensor()
    
    # Running sums for Welford's online algorithm
    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    
    for img_path in tqdm(image_paths, desc="Computing stats"):
        img = Image.open(img_path).convert("RGB")
        tensor = to_tensor(img)  # (3, H, W), values in [0, 1]
        
        # Count pixels
        pixels = tensor.shape[1] * tensor.shape[2]
        n_pixels += pixels
        
        # Sum per channel
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).sum(dim=(1, 2))
    
    # Compute mean and std
    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)
    
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Compute dataset mean and std")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/plantsegv3/images/train"),
        help="Directory containing training images",
    )
    args = parser.parse_args()
    
    mean, std = compute_mean_std(args.data_dir)
    
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Mean (RGB): [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"Std (RGB):  [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    print()


if __name__ == "__main__":
    main()

