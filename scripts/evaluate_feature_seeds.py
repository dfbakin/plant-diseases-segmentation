"""End-to-end evaluation of feature-based seeds vs CAM seeds.

Orchestrates:
  1. Generate seeds for N val images using SPDNet (feat_chmean, cam_max, spatial_proto)
  2. Threshold sweep on each seed type
  3. CRF parameter sweep on best seed type (feat_chmean)
  4. Summary table comparing all results

Usage:
    python scripts/evaluate_feature_seeds.py \
        --checkpoint outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
        --max_images 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.spdnet.cam_generator import (
    generate_all_cams,
    generate_all_seeds,
    load_spdnet_from_checkpoint,
)
from scripts.sweep_crf_params import sweep_crf_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate feature seeds end-to-end")
    p.add_argument("--checkpoint",
                    default="outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt")
    p.add_argument("--label_file",
                    default="outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy",
                    help="Label file for reference pool (train set for reference selection)")
    p.add_argument("--image_dir", default="data/plantsegv3/images/val")
    p.add_argument("--gt_dir", default="outputs/plantseg_binary_mc115/gt_binary_val")
    p.add_argument("--output_base", default="outputs/spdnet_plantseg/feature_seed_eval")
    p.add_argument("--image_ext", default=".jpg")
    p.add_argument("--num_classes", type=int, default=115)
    p.add_argument("--input_size", type=int, default=448)
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.75, 1.25])
    p.add_argument("--num_ref_images", type=int, default=1)
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--crf_workers", type=int, default=8)
    p.add_argument("--seed_modes", nargs="+", default=["feat_chmean", "cam_max", "spatial_proto"])
    p.add_argument("--skip_generation", action="store_true",
                    help="Skip seed generation if .npy files already exist")
    return p.parse_args()


def _get_eval_names(gt_dir: Path, max_images: int, seed: int = 42) -> list[str]:
    """Get list of image names to evaluate, subsampled if requested."""
    all_names = sorted(f.stem for f in gt_dir.glob("*.png"))
    if max_images > 0 and len(all_names) > max_images:
        rng = np.random.default_rng(seed)
        all_names = list(rng.choice(all_names, max_images, replace=False))
    return all_names


def _build_label_dict_for_names(
    label_file: str, names: list[str], gt_dir: Path,
) -> dict[str, np.ndarray]:
    """Build a label dict restricted to requested names.

    Uses the train label file for reference pool but filters to only
    the names we actually want seeds for. For names missing from the
    train labels (val-only images), synthesizes a label vector from GT.
    """
    full_labels = np.load(label_file, allow_pickle=True).item()
    num_cls = next(iter(full_labels.values())).shape[0] if full_labels else 115

    label_dict = {}
    for name in names:
        if name in full_labels:
            label_dict[name] = full_labels[name]
        else:
            gt_file = gt_dir / f"{name}.png"
            if gt_file.exists():
                from PIL import Image
                gt = np.array(Image.open(gt_file))
                has_disease = (gt > 0).any() if gt.max() < 255 else False
                lbl = np.zeros(num_cls, dtype=np.float32)
                if has_disease:
                    lbl[0] = 1.0
                label_dict[name] = lbl
    return label_dict, full_labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading SPDNet from {args.checkpoint}")
    model = load_spdnet_from_checkpoint(
        args.checkpoint, args.num_classes
    ).to(device)
    model.eval()

    eval_names = _get_eval_names(Path(args.gt_dir), args.max_images)
    log.info(f"Evaluating on {len(eval_names)} images")

    label_dict, full_labels = _build_label_dict_for_names(
        args.label_file, eval_names, Path(args.gt_dir)
    )

    all_results = {}
    timings = {}

    # Phase 1: Generate seeds for each mode
    for mode in args.seed_modes:
        seed_dir = output_base / f"seeds_{mode}"

        existing = list(seed_dir.glob("*.npy")) if seed_dir.exists() else []
        if args.skip_generation and len(existing) >= len(eval_names) * 0.9:
            log.info(f"[{mode}] Skipping generation ({len(existing)} files exist)")
        else:
            log.info(f"[{mode}] Generating seeds...")
            t0 = time.time()

            if mode == "cam_max":
                generate_all_cams(
                    model=model,
                    label_dict=label_dict,
                    image_dir=Path(args.image_dir),
                    output_dir=seed_dir,
                    image_ext=args.image_ext,
                    scales=args.scales,
                    input_size=args.input_size,
                    num_ref_images=args.num_ref_images,
                    binary_aggregate="max",
                    device=device,
                )
            else:
                generate_all_seeds(
                    model=model,
                    label_dict=label_dict,
                    image_dir=Path(args.image_dir),
                    output_dir=seed_dir,
                    image_ext=args.image_ext,
                    scales=args.scales,
                    input_size=args.input_size,
                    num_ref_images=args.num_ref_images,
                    seed_mode=mode,
                    device=device,
                )
            timings[f"gen_{mode}"] = time.time() - t0
            log.info(f"[{mode}] Generation took {timings[f'gen_{mode}']:.1f}s")

        # Phase 2: Threshold sweep
        log.info(f"[{mode}] Running threshold sweep...")
        t0 = time.time()
        avail_names = [n for n in eval_names if (seed_dir / f"{n}.npy").exists()]
        sweep_result = evaluate_cam_threshold_sweep(
            predict_dir=str(seed_dir),
            gt_dir=args.gt_dir,
            name_list=avail_names,
            num_cls=2,
            optimize_metric="disease_iou",
        )
        timings[f"sweep_{mode}"] = time.time() - t0

        best_at = sweep_result.get("result_at_best", {})
        fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
        disease_iou = best_at[fg_keys[0]] if fg_keys else 0.0
        all_results[mode] = {
            "best_threshold": sweep_result["best_threshold"],
            "disease_iou": disease_iou,
            "bg_iou": best_at.get("background", 0.0),
            "mIoU": best_at.get("mIoU", 0.0),
            "num_images": len(avail_names),
        }
        log.info(
            f"[{mode}] Best threshold={sweep_result['best_threshold']:.2f}  "
            f"disease_iou={all_results[mode]['disease_iou']:.2f}%  "
            f"mIoU={all_results[mode]['mIoU']:.2f}%"
        )

    # Phase 3: CRF sweep on feat_chmean
    best_seed_mode = "feat_chmean"
    crf_seed_dir = output_base / f"seeds_{best_seed_mode}"

    if crf_seed_dir.exists() and list(crf_seed_dir.glob("*.npy")):
        log.info(f"\n--- CRF Parameter Sweep on {best_seed_mode} ---")
        t0 = time.time()
        crf_results = sweep_crf_params(
            seed_dir=crf_seed_dir,
            image_dir=Path(args.image_dir),
            gt_dir=Path(args.gt_dir),
            image_ext=args.image_ext,
            num_cls=2,
            max_images=args.max_images,
            num_workers=args.crf_workers,
        )
        timings["crf_sweep"] = time.time() - t0
        log.info(f"CRF sweep took {timings['crf_sweep']:.1f}s")

        if crf_results:
            all_results["crf_best"] = crf_results[0]
            best_crf = crf_results[0]
            log.info(
                f"Best CRF: srgb={best_crf['srgb']}, "
                f"bg_thr={best_crf['bg_threshold']}, "
                f"scale={best_crf['scale_factor']}  "
                f"disease_iou={best_crf['disease_iou']:.2f}%  "
                f"mIoU={best_crf['mIoU']:.2f}%"
            )
            all_results["crf_all"] = crf_results
    else:
        log.warning(f"No seeds in {crf_seed_dir}, skipping CRF sweep")

    # Save full results
    results_path = output_base / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({"results": all_results, "timings": timings, "args": vars(args)}, f, indent=2)
    log.info(f"Saved full results to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("FEATURE SEED EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Seed Mode':<18} {'Threshold':>10} {'Disease IoU':>12} {'BG IoU':>10} {'mIoU':>8}")
    print("-" * 70)
    for mode in args.seed_modes:
        if mode in all_results:
            r = all_results[mode]
            print(
                f"{mode:<18} {r['best_threshold']:>10.2f} "
                f"{r['disease_iou']:>11.2f}% {r['bg_iou']:>9.2f}% {r['mIoU']:>7.2f}%"
            )

    if "crf_best" in all_results:
        cb = all_results["crf_best"]
        label = f"CRF(s={cb['srgb']:.0f},b={cb['bg_threshold']:.2f})"
        print(
            f"{label:<18} {'N/A':>10} "
            f"{cb['disease_iou']:>11.2f}% {cb['bg_iou']:>9.2f}% {cb['mIoU']:>7.2f}%"
        )
    print("=" * 70)

    # Save best CRF config
    if "crf_best" in all_results:
        crf_config_path = output_base / "best_crf_config.json"
        with open(crf_config_path, "w") as f:
            json.dump(all_results["crf_best"], f, indent=2)
        log.info(f"Saved best CRF config to {crf_config_path}")


if __name__ == "__main__":
    main()
