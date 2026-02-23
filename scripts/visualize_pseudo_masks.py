"""Visualize pseudo masks side-by-side with original images and GT masks.

Usage:
    python scripts/visualize_pseudo_masks.py
    python scripts/visualize_pseudo_masks.py --n 10 --output outputs/vis_masks.png
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

VOC_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)

VOC_CLASSES = [
    "bg", "aero", "bike", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "table", "dog", "horse", "mbike", "person",
    "plant", "sheep", "sofa", "train", "tv",
]


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        rgb[mask == c] = palette[c]
    rgb[mask == 255] = [224, 224, 224]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/VOC2012/JPEGImages")
    parser.add_argument("--pseudo_dir", default="outputs/pseudo_masks")
    parser.add_argument("--gt_dir", default="data/VOC2012/SegmentationClassAug")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", default="outputs/vis_pseudo_masks.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pseudo_dir = Path(args.pseudo_dir)
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)

    names = sorted(f.stem for f in pseudo_dir.glob("*.png"))
    if not names:
        print(f"No masks found in {pseudo_dir}")
        return

    random.seed(args.seed)
    samples = random.sample(names, min(args.n, len(names)))

    has_gt = gt_dir.exists()
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(args.n, ncols, figsize=(5 * ncols, 4.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(samples):
        img = np.array(Image.open(image_dir / f"{name}{args.image_ext}").convert("RGB"))
        pseudo = np.array(Image.open(pseudo_dir / f"{name}.png"))

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(name, fontsize=10)
        axes[row, 0].axis("off")

        pseudo_rgb = colorize_mask(pseudo, VOC_PALETTE)
        axes[row, 1].imshow(pseudo_rgb)
        classes_present = sorted(set(np.unique(pseudo)) - {0, 255})
        label_str = ", ".join(VOC_CLASSES[c] for c in classes_present if c < len(VOC_CLASSES))
        axes[row, 1].set_title(f"Pseudo: {label_str}", fontsize=9)
        axes[row, 1].axis("off")

        if has_gt:
            gt_path = gt_dir / f"{name}.png"
            if gt_path.exists():
                gt = np.array(Image.open(gt_path))
                gt_rgb = colorize_mask(gt, VOC_PALETTE)
                gt_classes = sorted(set(np.unique(gt)) - {0, 255})
                gt_label_str = ", ".join(
                    VOC_CLASSES[c] for c in gt_classes if c < len(VOC_CLASSES)
                )
                axes[row, 2].imshow(gt_rgb)
                axes[row, 2].set_title(f"GT: {gt_label_str}", fontsize=9)
            else:
                axes[row, 2].text(0.5, 0.5, "No GT", ha="center", va="center")
            axes[row, 2].axis("off")

    headers = ["Image", "Pseudo Mask"]
    if has_gt:
        headers.append("Ground Truth")
    for col, header in enumerate(headers):
        axes[0, col].set_title(f"{header}\n{axes[0, col].get_title()}", fontsize=10)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
