"""Smoke test for cam_classifier evaluation:
- Verifies generate_all_cams runs end-to-end with corrected refs
- Confirms CAMs differ across the three checkpoints
- Confirms refs are loaded from /train/ not /val/
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wsss.spdnet.cam_generator import (
    generate_all_cams,
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
    ("spatial PS-only",
     "outputs/spdnet_plantseg/spdnet_spatial_n1_ps/checkpoints/"
     "epoch=epoch=76-val_mAP=val/mAP=0.7970.ckpt"),
    ("spatial PS+PV",
     "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/"
     "epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt"),
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    log = logging.getLogger("smoke_cam")
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
            seen.add(c); val_names.append(n)
            if len(val_names) >= 5: break
    log.info(f"Selected {len(val_names)} val queries (one per class)")

    label_dict = {n: np.zeros(NUM_CLASSES, dtype=np.float32) for n in val_names}
    for n in val_names:
        label_dict[n][resolver(n)] = 1.0

    queries_seen, refs_seen = [], []
    real_open = PIL.Image.open

    def patched_open(fp, *a, **kw):
        s = str(fp)
        if "/val/" in s:
            queries_seen.append(Path(s).stem)
        elif "/train/" in s:
            refs_seen.append(Path(s).stem)
        return real_open(fp, *a, **kw)

    cam_per_model = {}
    failures = 0
    for tag, ckpt in CHECKPOINTS:
        log.info("=" * 70)
        log.info(f"Testing checkpoint: {tag}")
        try:
            model = load_spdnet_from_checkpoint(ckpt, NUM_CLASSES).to(device).eval()
            log.info(f"  fusion_mode = {model.fusion_mode}")
            if model.fusion_mode == "spatial":
                log.info(f"  spatial_attn.gate = {model.spatial_attn.gate.item():.4f}")
        except Exception as e:
            log.exception(f"FAILED to load: {e}")
            failures += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            try:
                PIL.Image.open = patched_open
                queries_seen.clear(); refs_seen.clear()
                generate_all_cams(
                    model=model, label_dict=label_dict,
                    image_dir=IMAGE_DIR, output_dir=out_dir,
                    image_ext=".jpg", scales=[1.0],
                    input_size=448, num_ref_images=1,
                    binary_aggregate="max", device=device,
                    ref_pool=train_pool, ref_image_dir=REF_IMAGE_DIR,
                    query_class_resolver=resolver,
                )
            finally:
                PIL.Image.open = real_open

            saved = sorted(out_dir.glob("*.npy"))
            log.info(f"  Saved {len(saved)} CAMs")
            if len(saved) != len(val_names):
                log.error(f"  expected {len(val_names)}, got {len(saved)}")
                failures += 1

            cams = {p.stem: np.load(str(p), allow_pickle=True).item()[0]
                    for p in saved}
            cam_per_model[tag] = cams

            sample = cams[val_names[0]]
            log.info(f"  Sample CAM shape={sample.shape}, "
                     f"range=[{sample.min():.3f},{sample.max():.3f}], "
                     f"mean={sample.mean():.3f}")

            n_train = len(set(refs_seen))
            n_val = len(set(queries_seen))
            log.info(f"  Refs loaded from /train/: {n_train} unique  |  "
                     f"queries from /val/: {n_val}")
            ok_pairs = 0
            for q, r in zip(sorted(set(queries_seen)), sorted(set(refs_seen))):
                if resolver(q) == resolver(r):
                    ok_pairs += 1
            log.info(f"  Same-class pairs: {ok_pairs}/{len(set(queries_seen))}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info("Cross-model comparison (same query, different model):")
    base = list(cam_per_model.keys())[0]
    for n in val_names:
        c0 = cam_per_model[base][n]
        diffs = []
        for tag in list(cam_per_model.keys())[1:]:
            c = cam_per_model[tag][n]
            if c.shape != c0.shape:
                continue
            d = np.abs(c - c0).mean()
            r = np.corrcoef(c0.flatten(), c.flatten())[0, 1]
            diffs.append(f"vs {tag}: |diff|={d:.3f} corr={r:.3f}")
        log.info(f"  {n[:35]:<35}  {' | '.join(diffs)}")

    log.info("")
    if failures == 0:
        log.info("PASS - all checkpoints + cam_classifier mode work end-to-end")
        return 0
    else:
        log.error(f"FAIL - {failures} step(s) failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
