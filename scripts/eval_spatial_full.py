"""Full evaluation of SPDNet models with per-distribution CRF tuning.

Compares the token-fusion baseline (n1_heavy) against the spatial cross-
attention models (PS-only and PS+PV) under apples-to-apples conditions.

For each model x seed_mode:
  1. Generate seeds (single-scale, skip if already exist)
     -- references are picked from PlantSeg train, same class as query
        (class parsed from val filename), matching training-time pairing
  2. Threshold sweep (find optimal binarization threshold)
  3. CRF parameter sweep (tune srgb, bg_threshold, scale_factor for THIS distribution)
  4. Full CRF evaluation with best params
  5. Generate visualizations (25 images)

Usage:
    python scripts/eval_spatial_full.py 2>&1 | tee logs/eval_spatial_full.log
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.refinement.crf import apply_crf
from src.wsss.spdnet.cam_generator import generate_all_seeds, load_spdnet_from_checkpoint
from src.wsss.spdnet.class_resolver import (
    build_class_pool_from_labels,
    load_class_names,
    make_filename_class_resolver,
)
from scripts.sweep_crf_params import sweep_crf_params

IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
REF_IMAGE_DIR = Path("data/plantsegv3/images/train")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
CLASS_NAMES_FILE = "outputs/plantseg_binary_mc115/labels/class_names.txt"
NUM_CLASSES = 115
CRF_SWEEP_IMAGES = 200
CRF_WORKERS = 8
VIZ_COUNT = 25

GT_COLOR = np.array([0.85, 0.15, 0.85])
CRF_COLOR = np.array([0.0, 0.75, 0.75])
PRED_COLOR = np.array([0.85, 0.25, 0.15])
GT_CONTOUR_RGB = (255, 50, 220)

RUNS = [
    {
        "name": "spdnet_token_n1_heavy",
        "checkpoint": "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt",
        "label": "Token N=1 PS (baseline, mAP=85.9%)",
        "seed_subdir": "_corrected_refs",
    },
    {
        "name": "spdnet_spatial_n1_ps",
        "checkpoint": "outputs/spdnet_plantseg/spdnet_spatial_n1_ps/checkpoints/"
                      "epoch=epoch=76-val_mAP=val/mAP=0.7970.ckpt",
        "label": "Spatial PS-only (mAP=79.7%)",
        "seed_subdir": "_corrected_refs",
    },
    {
        "name": "spdnet_spatial_n1_ps_pv",
        "checkpoint": "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/"
                      "epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt",
        "label": "Spatial PS+PV (mAP=88.8%)",
        "seed_subdir": "_corrected_refs",
    },
]
SEED_MODES = ["feat_chmean", "feat_chvar"]


def resize_seed(s, w, h):
    if s.shape == (h, w):
        return s
    return np.array(
        Image.fromarray(s.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    )


def normalize(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(x)


def compute_iou(pred, gt):
    p, g = pred > 0, gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def apply_crf_to_seed(img_np, cam_dict, img_h, img_w, srgb, bg_thr, scale):
    resized = {k: resize_seed(v, img_w, img_h) for k, v in cam_dict.items()}
    probs = apply_crf(
        img_np, resized, bg_threshold=bg_thr, t=10,
        num_cls=2, scale_factor=scale, srgb=srgb,
    )
    return np.argmax(probs, axis=0).astype(np.uint8)


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


def generate_visualizations(rname, seed_dir, threshold, crf_params, viz_dir, label, seed_mode):
    random.seed(42)
    names = sorted(f.stem for f in seed_dir.glob("*.npy"))
    gt_avail = {f.stem for f in GT_DIR.glob("*.png")}
    names = [n for n in names if n in gt_avail]
    selected = random.sample(names, min(VIZ_COUNT, len(names)))
    selected.sort()

    viz_dir.mkdir(parents=True, exist_ok=True)
    all_panels = []

    bp = crf_params
    for name in tqdm(selected, desc=f"Viz {rname}"):
        img = np.array(Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        gt = np.array(Image.open(GT_DIR / f"{name}.png"))
        if gt.shape[:2] != (h, w):
            gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
        cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        seed = normalize(resize_seed(cam_dict[0], w, h))
        binary = (seed > threshold).astype(np.uint8)
        crf_mask = apply_crf_to_seed(img, cam_dict, h, w, bp["srgb"], bp["bg_threshold"], bp["scale_factor"])
        iou_t = compute_iou(binary, gt)
        iou_c = compute_iou(crf_mask, gt)

        p1 = overlay_contour(overlay_mask(img, gt, GT_COLOR), gt)
        p2 = overlay_heatmap(img, seed)
        p3 = overlay_contour(overlay_mask(img, binary, PRED_COLOR), gt)
        p4 = overlay_contour(overlay_mask(img, crf_mask, CRF_COLOR), gt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        titles = [
            "Original + GT",
            f"{seed_mode} seed",
            f"Thr={threshold:.2f} IoU={iou_t:.1%}",
            f"CRF(srgb={bp['srgb']:.0f}) IoU={iou_c:.1%}",
        ]
        for ax, panel, title in zip(axes, [p1, p2, p3, p4], titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(name, fontsize=11, fontweight="bold", y=1.0)
        plt.tight_layout(pad=0.3)
        fig.savefig(str(viz_dir / f"{name}.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        all_panels.append({"name": name, "panels": [p1, p2, p3, p4], "iou_thr": iou_t, "iou_crf": iou_c})

    n_rows = min(len(all_panels), 12)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.2 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = ["Original + GT", f"{seed_mode} seed",
                  f"Threshold={threshold:.2f}", f"CRF(srgb={bp['srgb']:.0f})"]
    for row in range(n_rows):
        e = all_panels[row]
        for col in range(4):
            axes[row, col].imshow(e["panels"][col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(
            f"{e['name']}\nthr={e['iou_thr']:.0%} crf={e['iou_crf']:.0%}",
            fontsize=7, rotation=0, labelpad=120, va="center",
        )
    mean_t = np.mean([e["iou_thr"] for e in all_panels])
    mean_c = np.mean([e["iou_crf"] for e in all_panels])
    fig.suptitle(
        f"{label} | Mean IoU: thr={mean_t:.1%}, CRF={mean_c:.1%}",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(str(viz_dir / "summary_grid.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(selected)} figures + grid to {viz_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    val_names = sorted(f.stem for f in GT_DIR.glob("*.png"))

    print("Building same-class reference pool from PlantSeg train...")
    train_ref_pool = build_class_pool_from_labels(
        LABEL_FILE, REF_IMAGE_DIR, image_ext=".jpg",
    )
    class_names = load_class_names(CLASS_NAMES_FILE)
    class_resolver = make_filename_class_resolver(class_names)

    resolved = sum(1 for n in val_names if class_resolver(n) is not None)
    refable = sum(1 for n in val_names
                  if class_resolver(n) is not None
                  and class_resolver(n) in train_ref_pool
                  and len(train_ref_pool[class_resolver(n)]) > 0)
    print(f"  Train ref pool covers {len(train_ref_pool)}/{NUM_CLASSES} classes")
    print(f"  Val image classes resolved from filename: {resolved}/{len(val_names)}")
    print(f"  Val images with at least one same-class train ref: {refable}/{len(val_names)}")

    label_dict = {}
    for name in val_names:
        cls = class_resolver(name)
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        lbl[cls if cls is not None else 0] = 1.0
        label_dict[name] = lbl

    all_metrics = {}

    for run in RUNS:
        rname = run["name"]
        ckpt = run["checkpoint"]
        label = run["label"]
        seed_subdir = run.get("seed_subdir", "")
        output_base = Path(f"outputs/spdnet_plantseg/{rname}_eval")
        output_base.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  {rname} ({label})")
        print(f"{'=' * 70}")

        model = load_spdnet_from_checkpoint(ckpt, NUM_CLASSES).to(device)
        model.eval()
        print(f"  fusion_mode = {model.fusion_mode}")

        run_metrics = {}

        for seed_mode in SEED_MODES:
            seed_dir = output_base / f"seeds_{seed_mode}{seed_subdir}"
            existing = list(seed_dir.glob("*.npy")) if seed_dir.exists() else []

            # Phase 1: Generate seeds
            if len(existing) >= len(val_names) * 0.95:
                print(f"[{seed_mode}] Seeds exist ({len(existing)} files), skipping generation")
            else:
                print(f"[{seed_mode}] Generating seeds (single-scale, same-class refs)...")
                t0 = time.time()
                generate_all_seeds(
                    model=model, label_dict=label_dict,
                    image_dir=IMAGE_DIR, output_dir=seed_dir,
                    image_ext=".jpg", scales=[1.0],
                    input_size=448, num_ref_images=1,
                    seed_mode=seed_mode, device=device,
                    ref_pool=train_ref_pool,
                    ref_image_dir=REF_IMAGE_DIR,
                    query_class_resolver=class_resolver,
                )
                print(f"[{seed_mode}] Seed generation: {time.time() - t0:.0f}s")

            avail = [n for n in val_names if (seed_dir / f"{n}.npy").exists()]
            print(f"[{seed_mode}] {len(avail)} seeds available")

            # Phase 2: Threshold sweep
            print(f"[{seed_mode}] Threshold sweep...")
            t0 = time.time()
            sweep = evaluate_cam_threshold_sweep(
                predict_dir=str(seed_dir), gt_dir=str(GT_DIR),
                name_list=avail, num_cls=2, optimize_metric="disease_iou",
            )
            best_at = sweep.get("result_at_best", {})
            fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
            disease_iou_thr = best_at[fg_keys[0]] if fg_keys else 0.0
            best_thr = sweep["best_threshold"]
            print(f"[{seed_mode}] Threshold sweep: {time.time() - t0:.0f}s")
            print(f"[{seed_mode}] Best thr={best_thr:.2f}  "
                  f"disease_iou={disease_iou_thr:.2f}%  "
                  f"bg_iou={best_at.get('background', 0):.2f}%  "
                  f"mIoU={best_at.get('mIoU', 0):.2f}%")

            # Phase 3: CRF parameter sweep (on subset)
            print(f"[{seed_mode}] CRF parameter sweep ({CRF_SWEEP_IMAGES} images)...")
            t0 = time.time()
            crf_results = sweep_crf_params(
                seed_dir=seed_dir,
                image_dir=IMAGE_DIR,
                gt_dir=GT_DIR,
                image_ext=".jpg",
                num_cls=2,
                max_images=CRF_SWEEP_IMAGES,
                num_workers=CRF_WORKERS,
            )
            best_crf = crf_results[0]
            crf_p = {
                "srgb": best_crf["srgb"],
                "bg_threshold": best_crf["bg_threshold"],
                "scale_factor": best_crf["scale_factor"],
            }
            print(f"[{seed_mode}] CRF sweep: {time.time() - t0:.0f}s")
            print(f"[{seed_mode}] Best CRF: srgb={crf_p['srgb']}, "
                  f"bg_thr={crf_p['bg_threshold']}, scale={crf_p['scale_factor']}")
            print(f"[{seed_mode}] CRF sweep disease_iou={best_crf['disease_iou']:.2f}% "
                  f"(on {CRF_SWEEP_IMAGES} imgs)")

            crf_config_path = output_base / f"crf_sweep_{seed_mode}.json"
            with open(crf_config_path, "w") as f:
                json.dump({"best": best_crf, "top5": crf_results[:5]}, f, indent=2)

            # Phase 4: Full CRF evaluation with tuned params
            print(f"[{seed_mode}] Full CRF eval (all {len(avail)} images, tuned params)...")
            t0 = time.time()
            crf_ious_disease, crf_ious_bg = [], []
            for name in tqdm(avail, desc=f"CRF {seed_mode}"):
                img = np.array(Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB"))
                h, w = img.shape[:2]
                gt = np.array(Image.open(GT_DIR / f"{name}.png"))
                if gt.shape[:2] != (h, w):
                    gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
                cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
                crf_mask = apply_crf_to_seed(
                    img, cam_dict, h, w,
                    crf_p["srgb"], crf_p["bg_threshold"], crf_p["scale_factor"],
                )

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
            print(f"[{seed_mode}] Full CRF: {time.time() - t0:.0f}s")
            print(f"[{seed_mode}] disease_iou={crf_disease:.2f}%  "
                  f"bg_iou={crf_bg:.2f}%  mIoU={crf_miou:.2f}%")

            run_metrics[seed_mode] = {
                "best_threshold": best_thr,
                "threshold_disease_iou": disease_iou_thr,
                "threshold_bg_iou": best_at.get("background", 0),
                "threshold_miou": best_at.get("mIoU", 0),
                "crf_params": crf_p,
                "crf_disease_iou": crf_disease,
                "crf_bg_iou": crf_bg,
                "crf_miou": crf_miou,
            }

        # Phase 5: Visualizations for best seed mode
        best_mode = max(run_metrics, key=lambda m: run_metrics[m]["crf_disease_iou"])
        best_seed_dir = output_base / f"seeds_{best_mode}{seed_subdir}"
        best_thr = run_metrics[best_mode]["best_threshold"]
        bp = run_metrics[best_mode]["crf_params"]

        print(f"\nBest seed mode: {best_mode} "
              f"(CRF disease_iou={run_metrics[best_mode]['crf_disease_iou']:.2f}%)")
        print("Generating visualizations...")

        viz_dir = Path(f"outputs/visualizations/{rname}_{best_mode}_crf{seed_subdir}")
        generate_visualizations(rname, best_seed_dir, best_thr, bp, viz_dir, label, best_mode)

        all_metrics[rname] = {
            "label": label,
            "num_images": len(avail),
            "best_seed_mode": best_mode,
            "seed_results": run_metrics,
        }

        with open(output_base / "evaluation_results.json", "w") as f:
            json.dump(all_metrics[rname], f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 110)
    print("SPDNet EVALUATION (corrected same-class refs) — FINAL SUMMARY")
    print("=" * 110)
    print(f"{'Run':<35} {'Seed':<20} {'Thr':>5} {'DisIoU(thr)':>12} "
          f"{'CRF params':>22} {'DisIoU(CRF)':>12} {'BG(CRF)':>9} {'mIoU(CRF)':>10}")
    print("-" * 110)
    for rname, m in all_metrics.items():
        for smode, sm in m["seed_results"].items():
            marker = " *" if smode == m["best_seed_mode"] else "  "
            cp = sm["crf_params"]
            crf_str = f"s={cp['srgb']:.0f} bg={cp['bg_threshold']:.2f} sc={cp['scale_factor']:.0f}"
            print(f"{m['label']:<35} {smode + marker:<20} {sm['best_threshold']:>5.2f} "
                  f"{sm['threshold_disease_iou']:>11.2f}% "
                  f"{crf_str:>22} "
                  f"{sm['crf_disease_iou']:>11.2f}% "
                  f"{sm['crf_bg_iou']:>8.2f}% "
                  f"{sm['crf_miou']:>9.2f}%")
    print("-" * 110)
    print("Historical baselines (BUGGY random-class refs from val, for comparison):")
    print("  Spatial PS-only feat_chvar:   34.21% (thr) -> 36.94% (CRF)")
    print("  Spatial PS+PV   feat_chvar:   34.08% (thr) -> 37.15% (CRF)")
    print("  Token N=1 feat_chmean (200):  ~36.5% (thr) -> ~42.1% (CRF, srgb=5)")
    print("  MCTformer MC115 (raw CAM):    29.98% disease IoU")
    print("=" * 110)

    summary_path = Path("outputs/spdnet_plantseg/eval_summary_corrected_refs.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
