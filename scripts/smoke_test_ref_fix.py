"""Smoke test: verify that the corrected reference selection in
``generate_all_seeds`` actually picks same-class references from the
PlantSeg train set when the query comes from val.

It does *not* save anything to disk -- just monkey-patches PIL.Image.open
to log every image opened and prints a per-query summary.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import PIL.Image
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

CKPT = "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    log = logging.getLogger("smoke")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Building same-class reference pool from PlantSeg train ...")
    train_pool = build_class_pool_from_labels(LABEL_FILE, REF_IMAGE_DIR, ".jpg")
    class_names = load_class_names(CLASS_NAMES_FILE)
    resolver = make_filename_class_resolver(class_names)
    log.info(f"  Train ref pool covers {len(train_pool)}/{NUM_CLASSES} classes")

    val_names_all = sorted(f.stem for f in GT_DIR.glob("*.png"))
    seen_classes: set[int] = set()
    val_names: list[str] = []
    for n in val_names_all:
        cls = resolver(n)
        if cls is None or cls in seen_classes:
            continue
        if cls not in train_pool or not train_pool[cls]:
            continue
        seen_classes.add(cls)
        val_names.append(n)
        if len(val_names) >= 8:
            break
    log.info(f"Selected {len(val_names)} val queries (one per class) for smoke test")
    for n in val_names:
        log.info(f"  query={n}  -> class={resolver(n)} ({class_names[resolver(n)]})")

    label_dict = {}
    for n in val_names:
        cls = resolver(n)
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        lbl[cls] = 1.0
        label_dict[n] = lbl

    queries_seen, refs_seen = [], []
    real_open = PIL.Image.open

    def patched_open(fp, *a, **kw):
        s = str(fp)
        if "/val/" in s:
            queries_seen.append(Path(s).stem)
        elif "/train/" in s:
            refs_seen.append(Path(s).stem)
        return real_open(fp, *a, **kw)

    PIL.Image.open = patched_open
    try:
        log.info(f"Loading model from {CKPT} ...")
        model = load_spdnet_from_checkpoint(CKPT, NUM_CLASSES).to(device).eval()
        log.info(f"  fusion_mode = {model.fusion_mode}")

        out_dir = Path("/tmp/spdnet_smoke_seeds")
        if out_dir.exists():
            for p in out_dir.glob("*.npy"):
                p.unlink()
        log.info(f"Generating seeds to {out_dir} (corrected refs) ...")
        generate_all_seeds(
            model=model, label_dict=label_dict,
            image_dir=IMAGE_DIR, output_dir=out_dir,
            image_ext=".jpg", scales=[1.0],
            input_size=448, num_ref_images=1,
            seed_mode="feat_chmean", device=device,
            ref_pool=train_pool, ref_image_dir=REF_IMAGE_DIR,
            query_class_resolver=resolver,
        )
    finally:
        PIL.Image.open = real_open

    log.info("=" * 70)
    log.info("Per-query reference summary (verifying same-class):")
    log.info("=" * 70)
    saved = sorted(p.stem for p in out_dir.glob("*.npy"))
    log.info(f"Seeds saved: {len(saved)}")
    for n in saved:
        log.info(f"  query={n} -> truth class={resolver(n)} ({class_names[resolver(n)]})")

    val_unique = sorted(set(queries_seen))
    train_unique = sorted(set(refs_seen))
    log.info("")
    log.info(f"Val images opened: {len(val_unique)}")
    log.info(f"Train images opened (refs): {len(train_unique)}")
    log.info("")

    log.info("Reference picks (each train ref class vs each val query class):")
    same_class, diff_class = 0, 0
    for q, r in zip(val_unique, train_unique):
        q_cls = resolver(q)
        r_cls = resolver(r)
        ok = q_cls == r_cls
        same_class += int(ok)
        diff_class += int(not ok)
        flag = "OK" if ok else "MISMATCH"
        log.info(f"  [{flag}] q={q} (cls {q_cls}: {class_names[q_cls]}) "
                 f"<-- r={r} (cls {r_cls}: {class_names[r_cls] if r_cls is not None else '?'})")
    log.info("")
    log.info(f"Result: {same_class}/{same_class + diff_class} pairs are same-class")
    if diff_class == 0:
        log.info("PASS - all references are same-class as queries")
        return 0
    else:
        log.error("FAIL - some references are NOT same-class")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
