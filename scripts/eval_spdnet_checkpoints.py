"""Evaluate SPDNet checkpoints: CAM generation, threshold sweep, and visualization.

Runs for each checkpoint:
  1. Generate CAMs in both 'max' and 'top_energy' binary aggregation modes
  2. Run threshold sweep against GT masks (optimize disease_iou)
  3. Generate 8-panel visualization grids (reuses visualize_spdnet_activations logic)

Usage:
    python scripts/eval_spdnet_checkpoints.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PYTHON = sys.executable

LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy"
IMAGE_DIR = "data/plantsegv3/images/val"
GT_DIR = "outputs/plantseg_binary_mc115/gt_binary_val"
VIZ_SCRIPT = "scripts/visualize_spdnet_activations.py"

CHECKPOINTS = {
    "n1_best": {
        "ckpt": "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt",
        "num_refs": 1,
        "desc": "N=1 heavy, epoch 69, val/mAP=0.859",
    },
    "n3_best": {
        "ckpt": "outputs/spdnet_plantseg/spdnet_fix_n3_heavy/checkpoints/best.ckpt",
        "num_refs": 3,
        "desc": "N=3 heavy, epoch 53, val/mAP=0.898",
    },
    "n3_last": {
        "ckpt": "outputs/spdnet_plantseg/spdnet_fix_n3_heavy/checkpoints/last.ckpt",
        "num_refs": 3,
        "desc": "N=3 heavy, epoch 56, val/mAP=0.880",
    },
}

MODES = ["max", "top_energy"]


def run_cam_generation(tag: str, ckpt: str, num_refs: int, mode: str) -> str:
    out_dir = f"outputs/spdnet_plantseg/cams/{tag}_{mode}"
    cmd = [
        PYTHON, "src/generate_spdnet_cams.py",
        f"checkpoint={ckpt}",
        f"output_dir={out_dir}",
        f"binary_aggregate={mode}",
        f"num_ref_images={num_refs}",
        f"label_file={LABEL_FILE}",
        f"image_dir={IMAGE_DIR}",
        f"gt_dir={GT_DIR}",
        "eval_threshold_sweep=true",
        "eval_optimize_metric=disease_iou",
    ]
    log.info(f"=== CAM generation: {tag} / {mode} ===")
    log.info(f"  checkpoint: {ckpt}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        log.error(f"FAILED: {result.stderr[-2000:]}")
    else:
        for line in result.stderr.split("\n"):
            if "Best threshold" in line or "threshold" in line.lower():
                log.info(f"  {line.strip()}")
    print(result.stderr[-3000:])
    return out_dir


def run_visualization(tag: str, ckpt: str, cam_dir_max: str) -> str:
    out_dir = f"outputs/visualizations/spdnet_{tag}"
    cmd = [
        PYTHON, VIZ_SCRIPT,
        "--checkpoint", ckpt,
        "--image_dir", IMAGE_DIR,
        "--gt_dir", GT_DIR,
        "--cam_dir", cam_dir_max,
        "--label_file", LABEL_FILE,
        "--output_dir", out_dir,
        "--num_images", "25",
        "--seed", "42",
    ]
    log.info(f"=== Visualization: {tag} ===")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        log.error(f"VIZ FAILED: {result.stderr[-2000:]}")
    print(result.stderr[-1000:])
    return out_dir


def main():
    results = {}

    for tag, info in CHECKPOINTS.items():
        log.info(f"\n{'='*70}")
        log.info(f"  Processing: {tag} -- {info['desc']}")
        log.info(f"{'='*70}")

        cam_dirs = {}
        for mode in MODES:
            cam_dir = run_cam_generation(tag, info["ckpt"], info["num_refs"], mode)
            cam_dirs[mode] = cam_dir

        viz_dir = run_visualization(tag, info["ckpt"], cam_dirs["max"])
        results[tag] = {"cam_dirs": cam_dirs, "viz_dir": viz_dir}

    log.info("\n" + "=" * 70)
    log.info("  ALL DONE")
    log.info("=" * 70)
    for tag, r in results.items():
        log.info(f"  {tag}:")
        for mode, d in r["cam_dirs"].items():
            log.info(f"    CAMs ({mode}): {d}")
        log.info(f"    Viz: {r['viz_dir']}")


if __name__ == "__main__":
    main()
