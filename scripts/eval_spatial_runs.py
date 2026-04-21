"""Evaluate spatial cross-attention runs: seeds, threshold sweep, CRF, visualizations.

Processes both spdnet_spatial_n1_ps and spdnet_spatial_n1_ps_pv checkpoints.
For each: generates seeds on full val set, runs threshold sweep,
applies CRF with optimal params, computes IoU metrics, and generates
comparison visualizations.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.refinement.crf import apply_crf
from src.wsss.spdnet.cam_generator import (
    generate_all_seeds,
    load_spdnet_from_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

GT_COLOR = np.array([0.85, 0.15, 0.85])
CRF_COLOR = np.array([0.0, 0.75, 0.75])
PRED_COLOR = np.array([0.85, 0.25, 0.15])
GT_CONTOUR_RGB = (255, 50, 220)

CRF_SRGB = 5.0
CRF_BG_THR = 0.30
CRF_SCALE = 1.0
CRF_ITERS = 10

SEED_MODES = ["feat_neg_chmean", "feat_chvar"]

RUNS = [
    {
        "name": "spdnet_spatial_n1_ps",
        "checkpoint": "outputs/spdnet_plantseg/spdnet_spatial_n1_ps/checkpoints/"
                      "epoch=epoch=76-val_mAP=val/mAP=0.7970.ckpt",
        "label": "Spatial PS-only (mAP=79.7%)",
    },
    {
        "name": "spdnet_spatial_n1_ps_pv",
        "checkpoint": "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/"
                      "epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt",
        "label": "Spatial PS+PV (mAP=88.8%)",
    },
]

IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
NUM_CLASSES = 115
VIZ_COUNT = 25
VIZ_SEED = 42


def overlay_heatmap(img, heatmap, alpha=0.55):
    hm = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]
    bl = img.astype(np.float32) / 255 * (1 - alpha) + hm_c.astype(np.float32) / 255 * alpha
    return (np.clip(bl, 0, 1) * 255).astype(np.uint8)


def overlay_mask(img, mask, color, alpha=0.40):
    r = img.astype(np.float32) / 255
    out = r.copy()
    fg = mask > 0
    out[fg] = out[fg] * (1 - alpha) + color * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def overlay_contour(img, mask, color=GT_CONTOUR_RGB, thickness=2):
    out = img.copy()
    m8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def resize_seed(s, w, h):
    if s.shape == (h, w):
        return s
    return np.array(Image.fromarray(s.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))


def normalize(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(x)


def compute_iou(pred, gt):
    p, g = pred > 0, gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def apply_crf_to_seed(img_np, cam_dict, img_h, img_w):
    resized = {k: resize_seed(v, img_w, img_h) for k, v in cam_dict.items()}
    probs = apply_crf(
        img_np, resized, bg_threshold=CRF_BG_THR,
        t=CRF_ITERS, num_cls=2, scale_factor=CRF_SCALE, srgb=CRF_SRGB,
    )
    return np.argmax(probs, axis=0).astype(np.uint8)


def generate_visualizations(run_name, seed_dir, threshold, output_dir, label, seed_mode):
    import random
    random.seed(VIZ_SEED)

    names = sorted(f.stem for f in seed_dir.glob("*.npy"))
    gt_avail = {f.stem for f in GT_DIR.glob("*.png")}
    names = [n for n in names if n in gt_avail]
    selected = random.sample(names, min(VIZ_COUNT, len(names)))
    selected.sort()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_panels = []
    ious_thr, ious_crf = [], []

    for name in tqdm(selected, desc=f"Viz {run_name}"):
        img = np.array(Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        gt = np.array(Image.open(GT_DIR / f"{name}.png"))
        if gt.shape[:2] != (h, w):
            gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))

        cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        seed = normalize(resize_seed(cam_dict[0], w, h))
        binary = (seed > threshold).astype(np.uint8)
        crf_mask = apply_crf_to_seed(img, cam_dict, h, w)

        iou_t = compute_iou(binary, gt)
        iou_c = compute_iou(crf_mask, gt)
        ious_thr.append(iou_t)
        ious_crf.append(iou_c)

        p1 = overlay_contour(overlay_mask(img, gt, GT_COLOR), gt)
        p2 = overlay_heatmap(img, seed)
        p3 = overlay_contour(overlay_mask(img, binary, PRED_COLOR), gt)
        p4 = overlay_contour(overlay_mask(img, crf_mask, CRF_COLOR), gt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        titles = [
            "Original + GT",
            f"{seed_mode} seed",
            f"Thr={threshold:.2f}\nIoU={iou_t:.1%}",
            f"CRF(srgb={CRF_SRGB:.0f})\nIoU={iou_c:.1%}",
        ]
        for ax, panel, title in zip(axes, [p1, p2, p3, p4], titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{name}", fontsize=11, fontweight="bold", y=1.0)
        plt.tight_layout(pad=0.3)
        fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        all_panels.append({"name": name, "panels": [p1, p2, p3, p4],
                           "iou_thr": iou_t, "iou_crf": iou_c})

    n_rows = min(len(all_panels), 12)
    col_labels = ["Original + GT", f"{seed_mode} seed",
                  f"Threshold={threshold:.2f}", f"CRF(srgb={CRF_SRGB:.0f})"]
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.2 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for row in range(n_rows):
        e = all_panels[row]
        for col in range(4):
            axes[row, col].imshow(e["panels"][col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=12, fontweight="bold")
        lbl = f"{e['name']}\nthr={e['iou_thr']:.0%}  crf={e['iou_crf']:.0%}"
        axes[row, 0].set_ylabel(lbl, fontsize=7, rotation=0, labelpad=120, va="center")

    mean_t = np.mean(ious_thr)
    mean_c = np.mean(ious_crf)
    fig.suptitle(
        f"{label}  |  Mean IoU: thr={mean_t:.1%}, CRF={mean_c:.1%}",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_dir / "summary_grid.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {len(selected)} figures + grid to {output_dir}")
    return mean_t, mean_c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_labels = np.load(LABEL_FILE, allow_pickle=True).item()

    val_names = sorted(f.stem for f in GT_DIR.glob("*.png"))
    label_dict = {}
    for name in val_names:
        if name in full_labels:
            label_dict[name] = full_labels[name]
        else:
            gt = np.array(Image.open(GT_DIR / f"{name}.png"))
            lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
            if (gt > 0).any():
                lbl[0] = 1.0
            label_dict[name] = lbl

    all_metrics = {}

    for run in RUNS:
        rname = run["name"]
        ckpt = run["checkpoint"]
        label = run["label"]
        output_base = Path(f"outputs/spdnet_plantseg/{rname}_eval")

        log.info(f"\n{'='*60}\n  Evaluating: {rname} ({label})\n{'='*60}")

        log.info(f"Loading checkpoint: {ckpt}")
        model = load_spdnet_from_checkpoint(ckpt, NUM_CLASSES).to(device)
        model.eval()

        run_metrics = {}

        for seed_mode in SEED_MODES:
            seed_dir = output_base / f"seeds_{seed_mode}"

            log.info(f"[{seed_mode}] Generating seeds (single-scale for speed)...")
            t0 = time.time()
            generate_all_seeds(
                model=model, label_dict=label_dict,
                image_dir=IMAGE_DIR, output_dir=seed_dir,
                image_ext=".jpg", scales=[1.0],
                input_size=448, num_ref_images=1,
                seed_mode=seed_mode, device=device,
            )
            gen_time = time.time() - t0
            log.info(f"[{seed_mode}] Seed generation took {gen_time:.0f}s")

            log.info(f"[{seed_mode}] Threshold sweep...")
            avail = [n for n in val_names if (seed_dir / f"{n}.npy").exists()]
            sweep = evaluate_cam_threshold_sweep(
                predict_dir=str(seed_dir), gt_dir=str(GT_DIR),
                name_list=avail, num_cls=2, optimize_metric="disease_iou",
            )
            best_at = sweep.get("result_at_best", {})
            fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
            disease_iou = best_at[fg_keys[0]] if fg_keys else 0.0
            best_thr = sweep["best_threshold"]
            log.info(f"[{seed_mode}] Best threshold={best_thr:.2f}, "
                     f"disease_iou={disease_iou:.2f}%, "
                     f"bg_iou={best_at.get('background', 0):.2f}%, "
                     f"mIoU={best_at.get('mIoU', 0):.2f}%")

            log.info(f"[{seed_mode}] CRF refinement...")
            t0 = time.time()
            crf_ious_disease, crf_ious_bg = [], []
            for name in tqdm(avail, desc=f"CRF {seed_mode}"):
                img = np.array(Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB"))
                h, w = img.shape[:2]
                gt = np.array(Image.open(GT_DIR / f"{name}.png"))
                if gt.shape[:2] != (h, w):
                    gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
                cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
                crf_mask = apply_crf_to_seed(img, cam_dict, h, w)

                gt_bin = (gt > 0).astype(np.uint8)
                pred_bin = (crf_mask > 0).astype(np.uint8)
                inter_d = ((pred_bin == 1) & (gt_bin == 1)).sum()
                union_d = ((pred_bin == 1) | (gt_bin == 1)).sum()
                inter_b = ((pred_bin == 0) & (gt_bin == 0)).sum()
                union_b = ((pred_bin == 0) | (gt_bin == 0)).sum()
                crf_ious_disease.append(inter_d / union_d if union_d > 0 else 1.0)
                crf_ious_bg.append(inter_b / union_b if union_b > 0 else 1.0)

            crf_disease = np.mean(crf_ious_disease) * 100
            crf_bg = np.mean(crf_ious_bg) * 100
            crf_miou = (crf_disease + crf_bg) / 2
            crf_time = time.time() - t0
            log.info(f"[{seed_mode}] CRF: disease_iou={crf_disease:.2f}%, "
                     f"bg_iou={crf_bg:.2f}%, mIoU={crf_miou:.2f}%  ({crf_time:.0f}s)")

            run_metrics[seed_mode] = {
                "best_threshold": best_thr,
                "threshold_disease_iou": disease_iou,
                "threshold_bg_iou": best_at.get("background", 0),
                "threshold_miou": best_at.get("mIoU", 0),
                "crf_disease_iou": crf_disease,
                "crf_bg_iou": crf_bg,
                "crf_miou": crf_miou,
                "gen_time_s": gen_time,
            }

        # Pick best seed mode by CRF disease IoU for visualization
        best_mode = max(run_metrics, key=lambda m: run_metrics[m]["crf_disease_iou"])
        best_seed_dir = output_base / f"seeds_{best_mode}"
        best_thr = run_metrics[best_mode]["best_threshold"]

        log.info(f"Best seed mode for {rname}: {best_mode}")
        log.info("Generating visualizations...")
        viz_dir = Path(f"outputs/visualizations/{rname}_{best_mode}_crf")
        generate_visualizations(rname, best_seed_dir, best_thr, viz_dir, label, best_mode)

        all_metrics[rname] = {
            "label": label,
            "num_images": len(avail),
            "best_seed_mode": best_mode,
            "seed_results": run_metrics,
            "crf_params": {"srgb": CRF_SRGB, "bg_threshold": CRF_BG_THR, "scale_factor": CRF_SCALE},
        }

        results_path = output_base / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(all_metrics[rname], f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 100)
    print("SPATIAL CROSS-ATTENTION EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Run':<30} {'Seed':<18} {'Thr':>5} {'Dis IoU(thr)':>13} {'Dis IoU(CRF)':>13} {'BG IoU(CRF)':>12} {'mIoU(CRF)':>10}")
    print("-" * 100)
    for rname, m in all_metrics.items():
        for smode, sm in m["seed_results"].items():
            marker = " *" if smode == m["best_seed_mode"] else "  "
            print(f"{m['label']:<30} {smode+marker:<18} {sm['best_threshold']:>5.2f} "
                  f"{sm['threshold_disease_iou']:>12.2f}% "
                  f"{sm['crf_disease_iou']:>12.2f}% "
                  f"{sm['crf_bg_iou']:>11.2f}% "
                  f"{sm['crf_miou']:>9.2f}%")

    print("-" * 100)
    print("Reference baselines:")
    print(f"  SPDNet token N=1 feat_chmean (200 imgs):   36.50% disease IoU -> 42.13% with CRF(srgb=5)")
    print(f"  MCTformer MC115 (token):                   29.98% disease IoU")
    print("=" * 100)

    summary_path = Path("outputs/spdnet_plantseg/spatial_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
