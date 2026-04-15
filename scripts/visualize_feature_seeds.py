"""Visualize feat_chmean seeds and CRF-refined masks against ground truth.

Generates per-image and summary grid comparisons:
  1. Original image + GT overlay
  2. feat_chmean seed (continuous heatmap)
  3. Thresholded binary mask overlay
  4. CRF-refined mask overlay (srgb=5, bg_thr=0.30)

Usage:
    python scripts/visualize_feature_seeds.py \
        --seed_dir outputs/spdnet_plantseg/feature_seed_eval/seeds_feat_chmean \
        --num_images 25
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.wsss.refinement.crf import apply_crf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

GT_COLOR = np.array([0.85, 0.15, 0.85])       # magenta fill
PRED_COLOR = np.array([0.85, 0.25, 0.15])      # red fill for threshold
CRF_COLOR = np.array([0.0, 0.75, 0.75])         # teal/cyan fill for CRF
GT_CONTOUR_BGR = (255, 50, 220)                 # bright magenta contour (RGB)


def overlay_heatmap(img_np, heatmap, alpha=0.55):
    hm_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)[:, :, ::-1]
    blended = (
        img_np.astype(np.float32) / 255 * (1 - alpha)
        + hm_color.astype(np.float32) / 255 * alpha
    )
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def overlay_mask(img_np, mask, color=GT_COLOR, alpha=0.45):
    img_f = img_np.astype(np.float32) / 255
    result = img_f.copy()
    fg = mask > 0
    result[fg] = result[fg] * (1 - alpha) + color * alpha
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def overlay_contour(img_np, mask, color=(0, 255, 0), thickness=2):
    result = img_np.copy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def compute_iou(pred, gt):
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dir",
                    default="outputs/spdnet_plantseg/feature_seed_eval/seeds_feat_chmean")
    p.add_argument("--image_dir", default="data/plantsegv3/images/val")
    p.add_argument("--gt_dir", default="outputs/plantseg_binary_mc115/gt_binary_val")
    p.add_argument("--output_dir",
                    default="outputs/visualizations/feat_chmean_crf_comparison")
    p.add_argument("--image_ext", default=".jpg")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--crf_srgb", type=float, default=5.0)
    p.add_argument("--crf_bg_threshold", type=float, default=0.30)
    p.add_argument("--crf_scale_factor", type=float, default=1.0)
    p.add_argument("--crf_iters", type=int, default=10)
    p.add_argument("--num_images", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_dir = Path(args.seed_dir)
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(seed_dir.glob("*.npy"))
    names = [f.stem for f in npy_files]
    gt_available = {f.stem for f in gt_dir.glob("*.png")}
    names = [n for n in names if n in gt_available]

    random.seed(args.seed)
    selected = random.sample(names, min(args.num_images, len(names)))
    selected.sort()
    log.info(f"Visualizing {len(selected)} images")

    all_panels = []
    all_ious = {"threshold": [], "crf": []}

    for name in tqdm(selected, desc="Rendering"):
        img_pil = Image.open(image_dir / f"{name}{args.image_ext}").convert("RGB")
        img_np = np.array(img_pil)
        img_h, img_w = img_np.shape[:2]

        gt = np.array(Image.open(gt_dir / f"{name}.png"))
        if gt.shape[:2] != (img_h, img_w):
            gt = np.array(Image.fromarray(gt).resize((img_w, img_h), Image.NEAREST))

        cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        seed_2d = cam_dict[0]
        if seed_2d.shape != (img_h, img_w):
            seed_2d = np.array(
                Image.fromarray(seed_2d.astype(np.float32), mode="F")
                .resize((img_w, img_h), Image.BILINEAR)
            )

        binary_mask = (seed_2d > args.threshold).astype(np.uint8)

        cam_dict_resized = {
            k: np.array(
                Image.fromarray(v.astype(np.float32), mode="F")
                .resize((img_w, img_h), Image.BILINEAR)
            ) if v.shape != (img_h, img_w) else v
            for k, v in cam_dict.items()
        }

        crf_probs = apply_crf(
            img_np, cam_dict_resized,
            bg_threshold=args.crf_bg_threshold,
            t=args.crf_iters,
            num_cls=2,
            scale_factor=args.crf_scale_factor,
            srgb=args.crf_srgb,
        )
        crf_mask = np.argmax(crf_probs, axis=0).astype(np.uint8)

        iou_thr = compute_iou(binary_mask, gt)
        iou_crf = compute_iou(crf_mask, gt)
        all_ious["threshold"].append(iou_thr)
        all_ious["crf"].append(iou_crf)

        p1 = overlay_mask(img_np, gt, color=GT_COLOR, alpha=0.40)
        p1 = overlay_contour(p1, gt, color=GT_CONTOUR_BGR, thickness=2)

        p2 = overlay_heatmap(img_np, seed_2d, alpha=0.55)

        p3 = overlay_mask(img_np, binary_mask, color=PRED_COLOR, alpha=0.40)
        p3 = overlay_contour(p3, gt, color=GT_CONTOUR_BGR, thickness=2)

        p4 = overlay_mask(img_np, crf_mask, color=CRF_COLOR, alpha=0.40)
        p4 = overlay_contour(p4, gt, color=GT_CONTOUR_BGR, thickness=2)

        # Individual figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        titles = [
            "Original + GT",
            "feat_chmean seed",
            f"Threshold={args.threshold}\nIoU={iou_thr:.1%}",
            f"CRF (srgb={args.crf_srgb:.0f})\nIoU={iou_crf:.1%}",
        ]
        for ax, panel, title in zip(axes, [p1, p2, p3, p4], titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(name, fontsize=11, fontweight="bold", y=1.0)
        plt.tight_layout(pad=0.3)
        fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        all_panels.append({
            "name": name,
            "panels": [p1, p2, p3, p4],
            "iou_thr": iou_thr,
            "iou_crf": iou_crf,
        })

    # Summary grid
    n_rows = min(len(all_panels), 12)
    col_labels = [
        "Original + GT", "feat_chmean seed",
        f"Threshold={args.threshold}", f"CRF(srgb={args.crf_srgb:.0f})",
    ]
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.2 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_rows):
        entry = all_panels[row]
        for col in range(4):
            axes[row, col].imshow(entry["panels"][col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=11, fontweight="bold")
        label = f"{entry['name']}\nthr={entry['iou_thr']:.0%}  crf={entry['iou_crf']:.0%}"
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0, labelpad=120, va="center")

    mean_thr = np.mean(all_ious["threshold"])
    mean_crf = np.mean(all_ious["crf"])
    fig.suptitle(
        f"feat_chmean + CRF Comparison  |  Mean IoU: threshold={mean_thr:.1%}, CRF={mean_crf:.1%}",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_dir / "summary_grid.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log.info(f"\nMean IoU — threshold: {mean_thr:.2%}, CRF: {mean_crf:.2%}")
    log.info(f"Saved {len(selected)} figures + summary grid to {output_dir}")


if __name__ == "__main__":
    main()
