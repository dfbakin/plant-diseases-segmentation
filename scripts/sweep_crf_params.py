"""Sweep CRF parameters (srgb, bg_threshold, scale_factor) on a seed directory.

Evaluates disease IoU for each parameter combination against GT masks.
Outputs a sorted table of results and optionally saves best config to JSON.

Usage:
    python scripts/sweep_crf_params.py \
        --seed_dir outputs/spdnet_plantseg/seeds/feat_chmean \
        --image_dir data/plantsegv3/images/val \
        --gt_dir outputs/plantseg_binary_mc115/gt_binary_val \
        --max_images 200 --num_workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wsss.refinement.crf import apply_crf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SRGB = [3, 5, 8, 13]
DEFAULT_BG_THR = [0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_SCALE = [1.0, 6.0, 12.0]


def _evaluate_one_config(
    names: list[str],
    seed_dir: Path,
    image_dir: Path,
    gt_dir: Path,
    image_ext: str,
    num_cls: int,
    crf_iters: int,
    srgb: float,
    bg_threshold: float,
    scale_factor: float,
) -> dict:
    """Run CRF with one param set on all images and compute IoU."""
    tp = np.zeros(num_cls, dtype=np.int64)
    p_sum = np.zeros(num_cls, dtype=np.int64)
    t_sum = np.zeros(num_cls, dtype=np.int64)

    for name in names:
        cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        pil_img = Image.open(image_dir / f"{name}{image_ext}").convert("RGB")
        sample_cam = next(iter(cam_dict.values()))
        cam_h, cam_w = sample_cam.shape
        if (pil_img.height, pil_img.width) != (cam_h, cam_w):
            pil_img = pil_img.resize((cam_w, cam_h), Image.BILINEAR)
        img = np.array(pil_img)

        q = apply_crf(
            img, cam_dict, bg_threshold=bg_threshold, t=crf_iters,
            num_cls=num_cls, scale_factor=scale_factor, srgb=srgb,
        )
        pred = np.argmax(q, axis=0).astype(np.uint8)

        gt = np.array(Image.open(gt_dir / f"{name}.png"))
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            )

        cal = gt < 255
        mask = (pred == gt) * cal
        for i in range(num_cls):
            p_sum[i] += np.sum((pred == i) * cal)
            t_sum[i] += np.sum((gt == i) * cal)
            tp[i] += np.sum((gt == i) * mask)

    iou = tp / (t_sum + p_sum - tp + 1e-10)
    bg_iou = float(iou[0] * 100)
    disease_iou = float(iou[1] * 100) if num_cls > 1 else 0.0
    miou = float(np.mean(iou) * 100)

    return {
        "srgb": srgb,
        "bg_threshold": bg_threshold,
        "scale_factor": scale_factor,
        "bg_iou": bg_iou,
        "disease_iou": disease_iou,
        "mIoU": miou,
    }


def _worker_wrapper(args: tuple) -> dict:
    names, seed_dir, image_dir, gt_dir, image_ext, num_cls, crf_iters, srgb, bg_thr, sf = args
    return _evaluate_one_config(
        names, seed_dir, image_dir, gt_dir, image_ext, num_cls, crf_iters,
        srgb, bg_thr, sf,
    )


def sweep_crf_params(
    seed_dir: Path,
    image_dir: Path,
    gt_dir: Path,
    image_ext: str = ".jpg",
    num_cls: int = 2,
    crf_iters: int = 10,
    srgb_values: list[float] = DEFAULT_SRGB,
    bg_thr_values: list[float] = DEFAULT_BG_THR,
    scale_values: list[float] = DEFAULT_SCALE,
    max_images: int = 0,
    num_workers: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Run CRF parameter sweep and return sorted results."""
    npy_files = sorted(seed_dir.glob("*.npy"))
    names = [f.stem for f in npy_files]

    gt_available = {f.stem for f in gt_dir.glob("*.png")}
    names = [n for n in names if n in gt_available]

    if max_images > 0 and len(names) > max_images:
        rng = np.random.default_rng(seed)
        names = list(rng.choice(names, max_images, replace=False))

    log.info(f"Sweeping CRF on {len(names)} images")
    log.info(f"  srgb: {srgb_values}")
    log.info(f"  bg_threshold: {bg_thr_values}")
    log.info(f"  scale_factor: {scale_values}")

    grid = list(itertools.product(srgb_values, bg_thr_values, scale_values))
    log.info(f"  Total configs: {len(grid)}")

    tasks = [
        (names, seed_dir, image_dir, gt_dir, image_ext, num_cls, crf_iters,
         srgb, bg_thr, sf)
        for srgb, bg_thr, sf in grid
    ]

    results = []
    if num_workers > 1 and len(grid) > 1:
        with Pool(num_workers) as pool:
            from tqdm import tqdm
            for r in tqdm(pool.imap_unordered(_worker_wrapper, tasks),
                          total=len(tasks), desc="CRF sweep"):
                results.append(r)
    else:
        from tqdm import tqdm
        for t in tqdm(tasks, desc="CRF sweep"):
            results.append(_worker_wrapper(t))

    results.sort(key=lambda r: r["disease_iou"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep CRF parameters on seeds")
    parser.add_argument("--seed_dir", required=True, help="Directory with .npy seed files")
    parser.add_argument("--image_dir", required=True, help="Directory with RGB images")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT masks (.png)")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--num_cls", type=int, default=2)
    parser.add_argument("--crf_iters", type=int, default=10)
    parser.add_argument("--srgb", nargs="+", type=float, default=DEFAULT_SRGB)
    parser.add_argument("--bg_thr", nargs="+", type=float, default=DEFAULT_BG_THR)
    parser.add_argument("--scale_factor", nargs="+", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", default="", help="Save results to JSON")
    args = parser.parse_args()

    results = sweep_crf_params(
        seed_dir=Path(args.seed_dir),
        image_dir=Path(args.image_dir),
        gt_dir=Path(args.gt_dir),
        image_ext=args.image_ext,
        num_cls=args.num_cls,
        crf_iters=args.crf_iters,
        srgb_values=args.srgb,
        bg_thr_values=args.bg_thr,
        scale_values=args.scale_factor,
        max_images=args.max_images,
        num_workers=args.num_workers,
    )

    print(f"\n{'srgb':>6} {'bg_thr':>7} {'scale':>6} {'bg_iou':>8} {'dis_iou':>8} {'mIoU':>7}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['srgb']:6.0f} {r['bg_threshold']:7.2f} {r['scale_factor']:6.1f} "
            f"{r['bg_iou']:7.2f}% {r['disease_iou']:7.2f}% {r['mIoU']:6.2f}%"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"best": results[0], "all_results": results}, f, indent=2)
        log.info(f"Saved results to {out}")


if __name__ == "__main__":
    main()
