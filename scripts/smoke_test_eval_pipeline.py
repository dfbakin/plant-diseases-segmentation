"""Smoke test for the full eval_spatial_full.py pipeline using a temporary
RUNS list, a tiny seed mode, and a 5-image subset of val.

Verifies:
- Class resolver + train ref pool are built properly
- Both token (n1_heavy) and spatial (PS+PV) checkpoints load correctly
- generate_all_seeds with corrected refs runs end-to-end on GPU
- Saved seeds have the expected structure
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wsss.spdnet.cam_generator import (
    generate_all_seeds,
    load_spdnet_from_checkpoint,
)
from src.wsss.spdnet.class_resolver import (
    build_class_pool_from_labels,
    load_class_names,
    make_filename_class_resolver,
)


IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
REF_IMAGE_DIR = Path("data/plantsegv3/images/train")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
CLASS_NAMES_FILE = "outputs/plantseg_binary_mc115/labels/class_names.txt"
NUM_CLASSES = 115

CHECKPOINTS = [
    ("token n1_heavy",
     "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"),
    ("spatial PS+PV",
     "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/"
     "epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt"),
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    log = logging.getLogger("smoke_pipe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pool = build_class_pool_from_labels(LABEL_FILE, REF_IMAGE_DIR, ".jpg")
    class_names = load_class_names(CLASS_NAMES_FILE)
    resolver = make_filename_class_resolver(class_names)
    log.info(f"Train ref pool: {len(train_pool)}/{NUM_CLASSES} classes")

    val_names_all = sorted(f.stem for f in GT_DIR.glob("*.png"))
    seen, val_names = set(), []
    for n in val_names_all:
        c = resolver(n)
        if c is not None and c not in seen and c in train_pool and train_pool[c]:
            val_names.append(n)
            seen.add(c)
            if len(val_names) >= 5:
                break
    log.info(f"Selected {len(val_names)} val queries (one per class)")

    label_dict = {n: np.zeros(NUM_CLASSES, dtype=np.float32) for n in val_names}
    for n in val_names:
        label_dict[n][resolver(n)] = 1.0

    failures = 0
    for tag, ckpt in CHECKPOINTS:
        log.info("=" * 70)
        log.info(f"Testing checkpoint: {tag}")
        log.info(f"  path: {ckpt}")
        try:
            model = load_spdnet_from_checkpoint(ckpt, NUM_CLASSES).to(device).eval()
            log.info(f"  fusion_mode = {model.fusion_mode}")
        except Exception as e:
            log.exception(f"  FAILED to load: {e}")
            failures += 1
            continue

        for seed_mode in ("feat_chmean", "feat_chvar"):
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                try:
                    generate_all_seeds(
                        model=model, label_dict=label_dict,
                        image_dir=IMAGE_DIR, output_dir=out_dir,
                        image_ext=".jpg", scales=[1.0],
                        input_size=448, num_ref_images=1,
                        seed_mode=seed_mode, device=device,
                        ref_pool=train_pool, ref_image_dir=REF_IMAGE_DIR,
                        query_class_resolver=resolver,
                    )
                    saved = sorted(out_dir.glob("*.npy"))
                    log.info(f"  [{seed_mode}] saved {len(saved)} seeds")
                    if len(saved) != len(val_names):
                        log.error(f"    expected {len(val_names)}, got {len(saved)}")
                        failures += 1
                        continue
                    sample = np.load(str(saved[0]), allow_pickle=True).item()
                    keys = list(sample.keys())
                    arr = sample[keys[0]]
                    log.info(f"    sample keys={keys[:3]}, "
                             f"shape={arr.shape}, range=[{arr.min():.3f},{arr.max():.3f}]")
                except Exception as e:
                    log.exception(f"  [{seed_mode}] FAILED: {e}")
                    failures += 1

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("=" * 70)
    if failures == 0:
        log.info("PASS - all checkpoints + seed modes work end-to-end with corrected refs")
        return 0
    else:
        log.error(f"FAIL - {failures} step(s) failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
