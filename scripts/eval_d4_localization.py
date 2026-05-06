"""Offline localization evaluation for D4 ablation checkpoints + baselines.

Goal: quantify disease-IoU of the strongest D4 recipe(s) against the historical
~42 % DisIoU / 60 % mIoU ceiling (`feat_chmean + CRF(srgb=5, bg=0.30)` on the
token baseline). Uses:

* 750-image deterministic val subset (shared across every checkpoint).
* Multi-scale + flip TTA during seed generation (4 scales x 2 flips = 8 augs).
* Three seed modes: ``cam_max`` (classifier CAM), ``feat_chmean``,
  ``feat_chvar``.
* Disease-IoU-optimised threshold sweep.
* CRF parameter sweep on 250 images, then full CRF re-evaluation on all 750.
* Per-checkpoint summary JSON + aggregate markdown table.

Reuses utilities from ``scripts/eval_spatial_full.py`` and
``scripts/sweep_crf_params.py``.

Run:
    uv run python scripts/eval_d4_localization.py \\
        --output_dir outputs/d4_localization \\
        --subset_size 750 --crf_sweep_images 250
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
from PIL import Image
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.refinement.crf import apply_crf
from src.wsss.spdnet.cam_generator import (
    generate_all_cams,
    generate_all_seeds,
    load_spdnet_from_checkpoint,
)
from src.wsss.spdnet.class_resolver import (
    build_class_pool_from_labels,
    load_class_names,
    make_filename_class_resolver,
)
from src.wsss.spdnet.online_loc_metric import select_deterministic_subset
from scripts.sweep_crf_params import sweep_crf_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_d4_loc")


# ---------- static paths ----------
IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
REF_IMAGE_DIR = Path("data/plantsegv3/images/train")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
CLASS_NAMES_FILE = "outputs/plantseg_binary_mc115/labels/class_names.txt"
NUM_CLASSES = 115
IMAGE_EXT = ".jpg"

# TTA scales (each scale also gets flipped inside `generate_all_seeds`)
TTA_SCALES: list[float] = [0.75, 1.0, 1.25, 1.5]

# CRF sweep grid.
#
# ``srgb`` is NOT resolution-invariant: the pairwise-color kernel
#   exp(-||Dxy||^2 / 2 sxy^2 - ||Drgb||^2 / 2 srgb^2)
# sees much smaller Drgb at CAM resolution (block-averaged colour) than at
# full image resolution (raw pixels). CAM-resolution sweeps tend to prefer
# low srgb (3-8); full-resolution sweeps need to explore higher values too,
# hence the extension to {18, 25}. 6 srgb x 4 bg x 3 scale = 72 configs.
CRF_SRGB = [3, 5, 8, 13, 18, 25]
CRF_BG_THR = [0.1, 0.2, 0.3, 0.4]
CRF_SCALE = [1.0, 6.0, 12.0]


# ---------- checkpoints under test ----------
# label, path, fusion_mode (None -> auto-detect from ckpt hyper_parameters).
RUNS_DEFAULT: list[dict] = [
    {
        "label": "d4_ac_safe (L_ac=0.1 + L_mask(union)=0.1)",
        "name": "d4_ac_safe",
        "checkpoint": "outputs/spdnet_aux_losses/spdnet_spatial_d4_ac_safe_warmstart_20260427/checkpoints/last.ckpt",
    },
    {
        "label": "D3 (L_mask(union)=1.0 + L_con=0.5)",
        "name": "D3",
        "checkpoint": "outputs/spdnet_aux_losses/spdnet_spatial_d3_d2plus_union_warmstart_20260427/checkpoints/last.ckpt",
    },
    {
        "label": "D2 (L_mask(chvar_only)=1.0)",
        "name": "D2",
        "checkpoint": "outputs/spdnet_aux_losses/spdnet_spatial_d2_mask_warmstart_20260427/checkpoints/last.ckpt",
    },
    {
        "label": "eq_only (warmstart parent)",
        "name": "eq_only",
        "checkpoint": "outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/last.ckpt",
    },
]

# Four readouts, stacked by abstraction level:
#   feat_chvar   -- pre-fusion channel variance (backbone/L_mask-responsive)
#   fused_chvar  -- post-fusion channel variance (direct attention effect)
#   attn_map     -- raw attention concentration M_q (direct L_ac/L_marg_H target;
#                   can be flat when M saturates at 1.0 across the whole image)
#   cam_max      -- classifier CAM max over classes (indirect, what we report)
SEED_MODES_DEFAULT: list[str] = [
    "feat_chvar", "fused_chvar", "attn_map", "cam_max",
]


def _resize_to(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    if arr.shape == (h, w):
        return arr
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.array(pil.resize((w, h), Image.BILINEAR))


def _per_image_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (disease_iou, bg_iou) for a single image (empty-pair => 1.0)."""
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    inter_d = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    union_d = int(((pred_bin == 1) | (gt_bin == 1)).sum())
    inter_b = int(((pred_bin == 0) & (gt_bin == 0)).sum())
    union_b = int(((pred_bin == 0) | (gt_bin == 0)).sum())
    dis_iou = inter_d / union_d if union_d > 0 else 1.0
    bg_iou = inter_b / union_b if union_b > 0 else 1.0
    return dis_iou, bg_iou


def _select_subset(gt_dir: Path, subset_size: int, seed: int) -> list[str]:
    all_names = sorted(f.stem for f in gt_dir.glob("*.png"))
    if subset_size >= len(all_names):
        return all_names
    return select_deterministic_subset(all_names, subset_size, seed=seed)


def _build_label_dict(
    names: list[str], class_resolver, num_classes: int,
) -> dict[str, np.ndarray]:
    """Build a multilabel dict for seed generation. Uses filename->class when
    possible, falls back to class 0 otherwise (so the image is still
    processed but with an arbitrary reference class -- the seed modes we
    use are label-independent).
    """
    out: dict[str, np.ndarray] = {}
    for n in names:
        cls = class_resolver(n)
        lbl = np.zeros(num_classes, dtype=np.float32)
        lbl[cls if cls is not None else 0] = 1.0
        out[n] = lbl
    return out


def _apply_crf_to_seed(
    img: np.ndarray,
    seed_dict: dict[int, np.ndarray],
    img_h: int,
    img_w: int,
    srgb: float,
    bg_threshold: float,
    scale_factor: float,
    num_cls: int = 2,
) -> np.ndarray:
    resized = {k: _resize_to(v, img_w, img_h) for k, v in seed_dict.items()}
    probs = apply_crf(
        img, resized, bg_threshold=bg_threshold, t=10,
        num_cls=num_cls, scale_factor=scale_factor, srgb=srgb,
    )
    return np.argmax(probs, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full-resolution CRF sweep.
#
# Why a separate sweep: `sweep_crf_params.sweep_crf_params` applies CRF at
# CAM resolution (downsamples the RGB to CAM size, runs CRF, NEAREST-upsamples
# the argmax to GT size before computing IoU). That's cheap but produces
# segmentations that differ qualitatively from full-resolution CRF output:
# the hyperparameters (especially ``srgb``) are not resolution-invariant.
#
# These helpers mirror `sweep_crf_params._evaluate_one_config` one-for-one,
# but run the CRF at full image resolution using the same pipeline as
# `_apply_crf_to_seed` / `_full_crf_eval`, so the sweep winner IS the best
# full-resolution config and transfers directly to the final 1000-image eval.
#
# They must be module-level because multiprocessing.Pool pickles callables.
# ---------------------------------------------------------------------------


def _fullres_eval_one_config(
    names: list[str],
    seed_dir: Path,
    image_dir: Path,
    gt_dir: Path,
    image_ext: str,
    num_cls: int,
    srgb: float,
    bg_threshold: float,
    scale_factor: float,
) -> dict:
    """Run full-resolution CRF with one param set on ``names`` and return
    a single MICRO IoU dict (matching ``sweep_crf_params`` output schema).
    """
    tp = np.zeros(num_cls, dtype=np.int64)
    p_sum = np.zeros(num_cls, dtype=np.int64)
    t_sum = np.zeros(num_cls, dtype=np.int64)

    for name in names:
        img = np.array(Image.open(image_dir / f"{name}{image_ext}").convert("RGB"))
        h, w = img.shape[:2]
        gt = np.array(Image.open(gt_dir / f"{name}.png"))
        if gt.shape[:2] != (h, w):
            gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
        seed_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        pred = _apply_crf_to_seed(
            img, seed_dict, h, w, srgb, bg_threshold, scale_factor, num_cls,
        )

        cal = gt < 255
        gt_bin = (gt > 0).astype(np.uint8) if num_cls == 2 else gt.astype(np.uint8)
        for i in range(num_cls):
            p_sum[i] += int(((pred == i) & cal).sum())
            t_sum[i] += int(((gt_bin == i) & cal).sum())
            tp[i] += int(((pred == i) & (gt_bin == i) & cal).sum())

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


def _fullres_worker_wrapper(args: tuple) -> dict:
    names, seed_dir, image_dir, gt_dir, image_ext, num_cls, srgb, bg_thr, sf = args
    return _fullres_eval_one_config(
        names, seed_dir, image_dir, gt_dir, image_ext, num_cls,
        srgb, bg_thr, sf,
    )


# Per-config hard timeout. A single CRF config processes ``max_images``
# images serially inside one worker; pathological pydensecrf images can
# push this well past the average. 1800 s (30 min) is ~10x the slowest
# healthy config we've measured (a 250-img fullres sweep at srgb=5 tops
# out near 180 s); anything past this is almost certainly a hang, not
# slow. See ``_full_crf_eval`` in eval_seg_probes.py for the same pattern
# at per-image granularity.
FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT = 1800.0
# Per-image cap inside the final full-subset CRF refinement pass. Same
# root cause as the sweep-level cap above (pydensecrf spinning forever
# on a single pathological input); see ``eval_seg_probes.py`` for the
# empirical observation (1 image took > 55 min while neighbours finished
# in < 60 s). 300 s is ~5x the slowest healthy image we've measured.
FULLRES_EVAL_IMG_TIMEOUT_SEC_DEFAULT = 300.0


def _sweep_crf_fullres(
    seed_dir: Path,
    image_dir: Path,
    gt_dir: Path,
    image_ext: str,
    num_cls: int,
    srgb_values: list[float],
    bg_thr_values: list[float],
    scale_values: list[float],
    max_images: int,
    num_workers: int,
    seed: int = 42,
    per_config_timeout_sec: float = FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT,
) -> list[dict]:
    """Full-resolution CRF sweep. Returns results sorted by disease IoU desc.

    Parallelism contract:
        * num_workers <= 1 or len(grid) <= 1  -> serial loop (debuggable).
        * otherwise                           -> multiprocessing.Pool.apply_async
                                                  with per-config .get(timeout=...).

    Per-config timeout: pydensecrf is a C++ extension that releases the GIL
    but does not honour Python signals. On rare pathological images (e.g.
    D3's cam_max seed at srgb=5, observed 2026-04-29) a single
    ``apply_crf`` call can spin indefinitely, deadlocking any orchestrator
    using ``imap_unordered`` (which has no per-task timeout). We therefore
    enforce a HARD wall-clock cap per config: when
    ``apply_async.get(timeout=N)`` raises ``multiprocessing.TimeoutError``
    we abandon that future and continue with the rest. ``Pool.__exit__``
    then ``terminate()``s any still-spinning worker on the way out.

    Timed-out configs are dropped from the returned list and logged so
    the operator can see that a parameter combination was pathological.
    """
    import itertools

    npy_files = sorted(seed_dir.glob("*.npy"))
    names = [f.stem for f in npy_files]
    gt_available = {f.stem for f in gt_dir.glob("*.png")}
    names = [n for n in names if n in gt_available]

    if max_images > 0 and len(names) > max_images:
        rng = np.random.default_rng(seed)
        names = list(rng.choice(names, max_images, replace=False))

    log.info(
        "[fullres-sweep] %d images, srgb=%s, bg=%s, sc=%s",
        len(names), srgb_values, bg_thr_values, scale_values,
    )

    grid = list(itertools.product(srgb_values, bg_thr_values, scale_values))
    log.info("[fullres-sweep] %d configs total", len(grid))

    tasks = [
        (names, seed_dir, image_dir, gt_dir, image_ext, num_cls, srgb, bg_thr, sf)
        for srgb, bg_thr, sf in grid
    ]

    results: list[dict] = []
    n_timed_out = 0
    n_errored = 0
    skipped_configs: list[dict] = []

    use_pool = num_workers > 1 and len(grid) > 1
    if use_pool:
        from multiprocessing import Pool
        from multiprocessing import TimeoutError as MPTimeoutError

        with Pool(num_workers) as pool:
            asyncs = [pool.apply_async(_fullres_worker_wrapper, (t,)) for t in tasks]
            pbar = tqdm(total=len(asyncs), desc="Full-res CRF sweep")
            for ar, t in zip(asyncs, tasks):
                srgb, bg_thr, sf = t[6], t[7], t[8]
                try:
                    r = ar.get(timeout=per_config_timeout_sec)
                    results.append(r)
                except MPTimeoutError:
                    n_timed_out += 1
                    skipped_configs.append(
                        {"srgb": srgb, "bg_threshold": bg_thr, "scale_factor": sf,
                         "reason": "timeout"}
                    )
                    log.warning(
                        "[fullres-sweep] timeout >%.0fs on srgb=%s bg=%s sc=%s -- skipping",
                        per_config_timeout_sec, srgb, bg_thr, sf,
                    )
                except Exception as e:
                    n_errored += 1
                    skipped_configs.append(
                        {"srgb": srgb, "bg_threshold": bg_thr, "scale_factor": sf,
                         "reason": f"error: {e!r}"}
                    )
                    log.warning(
                        "[fullres-sweep] worker error on srgb=%s bg=%s sc=%s: %r -- skipping",
                        srgb, bg_thr, sf, e,
                    )
                pbar.update(1)
            pbar.close()
            # `with Pool(...)` -> __exit__ -> terminate() will SIGKILL any
            # still-spinning worker (including any stuck on a timed-out
            # config). This is the whole reason we use apply_async+timeout
            # instead of imap_unordered.
    else:
        for t in tqdm(tasks, desc="Full-res CRF sweep"):
            try:
                results.append(_fullres_worker_wrapper(t))
            except Exception as e:
                n_errored += 1
                srgb, bg_thr, sf = t[6], t[7], t[8]
                skipped_configs.append(
                    {"srgb": srgb, "bg_threshold": bg_thr, "scale_factor": sf,
                     "reason": f"error: {e!r}"}
                )
                log.warning(
                    "[fullres-sweep] error on srgb=%s bg=%s sc=%s: %r -- skipping",
                    srgb, bg_thr, sf, e,
                )

    if n_timed_out or n_errored:
        log.warning(
            "[fullres-sweep] skipped %d/%d configs (%d timeouts, %d errors); "
            "ranking computed over %d completed configs",
            n_timed_out + n_errored, len(grid), n_timed_out, n_errored, len(results),
        )

    results.sort(key=lambda r: r["disease_iou"], reverse=True)
    return results


# Module-level worker for ``_full_crf_eval``'s Pool. MUST stay top-level
# so Linux fork-mode multiprocessing can pickle it. Keeps no live state
# (no CUDA tensors, no Hydra config, no open file handles); everything
# arrives through the ``args`` tuple. Returns both the per-image
# disease/bg IoU (for macro aggregation) AND the per-class TP/P/T pixel
# counts (for micro aggregation over the GT-ignore-calibrated region).
def _fullres_eval_img_worker(
    args: tuple[str, str, str, str, str, float, float, float, int],
) -> tuple[str, float, float, bool, np.ndarray, np.ndarray, np.ndarray]:
    """Run CRF + per-class pixel tallies for ONE image. Pool-safe.

    Returns (name, dis_iou_01, bg_iou_01, has_disease, tp, p_sum, t_sum)
    -- the arrays are length-``num_cls`` int64 partials that the parent
    accumulates into a single micro ratio. IoUs are in [0, 1] (not %)
    so the parent can multiply by 100 once.

    Raising is fine: the parent's apply_async.get() catches it and the
    image is skipped without poisoning the batch.
    """
    (name, seed_dir_s, image_dir_s, gt_dir_s, image_ext,
     srgb, bg_thr, sf, num_cls) = args
    seed_dir = Path(seed_dir_s)
    image_dir = Path(image_dir_s)
    gt_dir = Path(gt_dir_s)

    img = np.array(Image.open(image_dir / f"{name}{image_ext}").convert("RGB"))
    h, w = img.shape[:2]
    gt = np.array(Image.open(gt_dir / f"{name}.png"))
    if gt.shape[:2] != (h, w):
        gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
    seed_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
    crf_mask = _apply_crf_to_seed(img, seed_dict, h, w, srgb, bg_thr, sf, num_cls)

    cal = gt < 255
    pred_bin = (crf_mask > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8) if num_cls == 2 else gt.astype(np.uint8)

    tp = np.zeros(num_cls, dtype=np.int64)
    p_sum = np.zeros(num_cls, dtype=np.int64)
    t_sum = np.zeros(num_cls, dtype=np.int64)
    for i in range(num_cls):
        p_sum[i] = int(((pred_bin == i) & cal).sum())
        t_sum[i] = int(((gt_bin == i) & cal).sum())
        tp[i] = int(((pred_bin == i) & (gt_bin == i) & cal).sum())

    inter_d = int(((pred_bin == 1) & (gt_bin == 1) & cal).sum())
    union_d = int((((pred_bin == 1) | (gt_bin == 1)) & cal).sum())
    inter_b = int(((pred_bin == 0) & (gt_bin == 0) & cal).sum())
    union_b = int((((pred_bin == 0) | (gt_bin == 0)) & cal).sum())
    dis_iou = float(inter_d / union_d) if union_d > 0 else 1.0
    bg_iou = float(inter_b / union_b) if union_b > 0 else 1.0
    has_disease = bool(((gt_bin == 1) & cal).any())

    return name, dis_iou, bg_iou, has_disease, tp, p_sum, t_sum


def _full_crf_eval(
    seed_dir: Path,
    names: list[str],
    srgb: float,
    bg_thr: float,
    sf: float,
    num_workers: int = 1,
    per_image_timeout_sec: float = FULLRES_EVAL_IMG_TIMEOUT_SEC_DEFAULT,
) -> dict[str, float]:
    """Apply CRF with given params on every image and return a metric panel.

    Why a panel?
        Several things in this repo quietly use different IoU aggregators:
          - ``sweep_crf_params`` (CRF param sweep): MICRO -- pool TP/P/T pixels
            across all images, compute one ratio. This is what the historical
            42.13 % token baseline reported (``evaluate_feature_seeds.py``
            forwards the sweep top-1 directly).
          - ``OnlineCAMIoU`` (training ``val/cam_iou_best``): MACRO -- per-image
            IoU, then mean. Empty-pair images count as 1.
        The two numbers can differ by 5-15 pp on WSSS data with skewed lesion
        sizes, so we must be explicit about which one we're reporting.

    Parallelism contract:
        * num_workers <= 1 or len(names) <= 1 -> serial loop (debuggable,
          deterministic).
        * otherwise                           -> multiprocessing.Pool.apply_async
                                                 with per-image .get(timeout=...).

    Per-image timeout: pydensecrf releases the GIL but does not honour
    Python signals; on rare pathological inputs a single ``apply_crf``
    call can spin indefinitely (empirically: zucchini_downy_mildew_Bing_0120
    at srgb=5, scale_factor=1.0 ran > 55 min while every sibling image in
    the same batch finished in < 60 s). Timed-out images are dropped from
    the IoU averages and reported. ``Pool.__exit__`` calls ``terminate()``
    so any still-spinning worker is SIGKILLed on the way out.

    Returned panel
    --------------
    Primary (baseline-comparable, matches ``sweep_crf_params`` metric):
      * ``disease_iou_micro``, ``bg_iou_micro``, ``mIoU_micro``
    Diagnostic (per-image distribution):
      * ``disease_iou_macro``, ``bg_iou_macro``, ``mIoU_macro``
      * ``disease_iou_macro_nonempty`` -- macro over GT-nonempty images only
      * ``frac_imgs_disease_iou_ge_0.3`` / ``_ge_0.5`` -- tail quality
      * ``median_per_img_disease_iou``
    Plus: ``n_images``, ``n_images_nonempty``, ``n_skipped``, and the
    CRF params used.

    255 in GT is treated as ignore (no-op for our binary 0/1 GT, but future
    proof for VOC-style masks).
    """
    num_cls = 2
    tp = np.zeros(num_cls, dtype=np.int64)
    p_sum = np.zeros(num_cls, dtype=np.int64)
    t_sum = np.zeros(num_cls, dtype=np.int64)
    dis_ious: list[float] = []
    bg_ious: list[float] = []
    has_disease: list[bool] = []
    n_skipped = 0
    skipped_names: list[str] = []

    tasks = [
        (name, str(seed_dir), str(IMAGE_DIR), str(GT_DIR), IMAGE_EXT,
         float(srgb), float(bg_thr), float(sf), num_cls)
        for name in names
    ]
    desc = f"CRF eval srgb={srgb} bg={bg_thr} sc={sf}"

    use_pool = num_workers > 1 and len(tasks) > 1
    if use_pool:
        from multiprocessing import Pool
        from multiprocessing import TimeoutError as MPTimeoutError

        with Pool(num_workers) as pool:
            asyncs = [pool.apply_async(_fullres_eval_img_worker, (t,)) for t in tasks]
            pbar = tqdm(total=len(asyncs), desc=desc)
            for ar, t in zip(asyncs, tasks):
                try:
                    name, d, b, hd, tp_i, p_i, t_i = ar.get(
                        timeout=per_image_timeout_sec,
                    )
                    dis_ious.append(d)
                    bg_ious.append(b)
                    has_disease.append(hd)
                    tp += tp_i
                    p_sum += p_i
                    t_sum += t_i
                except MPTimeoutError:
                    n_skipped += 1
                    skipped_names.append(t[0])
                    log.warning(
                        "[CRF] %s: timeout > %.0fs on '%s' -- skipping",
                        seed_dir.name, per_image_timeout_sec, t[0],
                    )
                except Exception as e:
                    n_skipped += 1
                    skipped_names.append(t[0])
                    log.warning(
                        "[CRF] %s: worker error on '%s': %r -- skipping",
                        seed_dir.name, t[0], e,
                    )
                pbar.update(1)
            pbar.close()
    else:
        for t in tqdm(tasks, desc=desc):
            try:
                name, d, b, hd, tp_i, p_i, t_i = _fullres_eval_img_worker(t)
                dis_ious.append(d)
                bg_ious.append(b)
                has_disease.append(hd)
                tp += tp_i
                p_sum += p_i
                t_sum += t_i
            except Exception as e:
                n_skipped += 1
                skipped_names.append(t[0])
                log.warning(
                    "[CRF] %s: error on '%s': %r -- skipping",
                    seed_dir.name, t[0], e,
                )

    if n_skipped:
        preview = ", ".join(skipped_names[:5])
        more = f" (+{n_skipped - 5} more)" if n_skipped > 5 else ""
        log.warning(
            "[CRF] %s: SKIPPED %d/%d image(s) (timeout or worker error); "
            "IoU averages computed over %d/%d images. Skipped: %s%s",
            seed_dir.name, n_skipped, len(tasks),
            len(dis_ious), len(tasks), preview, more,
        )

    n_completed = len(dis_ious)
    if n_completed == 0:
        log.error(
            "[CRF] %s: EVERY image timed out or errored; returning NaN panel",
            seed_dir.name,
        )
        nan = float("nan")
        return {
            "n_images": len(names),
            "n_images_completed": 0,
            "n_images_nonempty": 0,
            "n_skipped": n_skipped,
            "srgb": float(srgb),
            "bg_threshold": float(bg_thr),
            "scale_factor": float(sf),
            "disease_iou_micro": nan, "bg_iou_micro": nan, "mIoU_micro": nan,
            "disease_iou_macro": nan, "bg_iou_macro": nan, "mIoU_macro": nan,
            "disease_iou_macro_nonempty": nan,
            "frac_imgs_disease_iou_ge_0.3": nan,
            "frac_imgs_disease_iou_ge_0.5": nan,
            "median_per_img_disease_iou": nan,
        }

    iou_micro = tp / (t_sum + p_sum - tp + 1e-10)
    disease_iou_micro = float(iou_micro[1] * 100)
    bg_iou_micro = float(iou_micro[0] * 100)
    miou_micro = float(np.mean(iou_micro) * 100)

    dis_arr = np.asarray(dis_ious) * 100
    bg_arr = np.asarray(bg_ious) * 100
    has_dis = np.asarray(has_disease, dtype=bool)
    disease_iou_macro = float(dis_arr.mean())
    bg_iou_macro = float(bg_arr.mean())
    miou_macro = (disease_iou_macro + bg_iou_macro) / 2.0
    dis_iou_nonempty = (
        float(dis_arr[has_dis].mean()) if has_dis.any() else float("nan")
    )
    frac_ge_30 = float(np.mean(dis_arr >= 30.0))
    frac_ge_50 = float(np.mean(dis_arr >= 50.0))
    median_dis = float(np.median(dis_arr))

    return {
        "n_images": len(names),
        "n_images_completed": n_completed,
        "n_images_nonempty": int(has_dis.sum()),
        "n_skipped": n_skipped,
        "srgb": float(srgb),
        "bg_threshold": float(bg_thr),
        "scale_factor": float(sf),
        # Primary -- baseline-comparable, matches CRF-sweep metric.
        "disease_iou_micro": disease_iou_micro,
        "bg_iou_micro": bg_iou_micro,
        "mIoU_micro": miou_micro,
        # Per-image diagnostics.
        "disease_iou_macro": disease_iou_macro,
        "bg_iou_macro": bg_iou_macro,
        "mIoU_macro": miou_macro,
        "disease_iou_macro_nonempty": dis_iou_nonempty,
        "frac_imgs_disease_iou_ge_0.3": frac_ge_30,
        "frac_imgs_disease_iou_ge_0.5": frac_ge_50,
        "median_per_img_disease_iou": median_dis,
    }


def _run_one_checkpoint(
    run_cfg: dict,
    subset_names: list[str],
    label_dict: dict[str, np.ndarray],
    train_ref_pool: dict,
    class_resolver,
    output_base: Path,
    seed_modes: list[str],
    crf_sweep_images: int,
    crf_workers: int,
    device: torch.device,
    skip_seed_gen: bool,
    fullres_sweep: bool = True,
    fullres_sweep_cfg_timeout_sec: float = FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT,
    fullres_eval_img_timeout_sec: float = FULLRES_EVAL_IMG_TIMEOUT_SEC_DEFAULT,
) -> dict:
    """Pipeline for ONE checkpoint: generate seeds -> threshold sweep -> CRF
    sweep -> full CRF eval on the 750-image subset."""
    ckpt_path = Path(run_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    log.info("loading %s from %s", run_cfg["name"], ckpt_path)
    model = load_spdnet_from_checkpoint(str(ckpt_path), NUM_CLASSES).to(device)
    model.eval()
    log.info("  fusion_mode=%s  num_classes=%s", model.fusion_mode, NUM_CLASSES)

    output_base.mkdir(parents=True, exist_ok=True)
    per_mode: dict[str, dict] = {}

    for mode in seed_modes:
        seed_dir = output_base / f"seeds_{mode}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        existing = {f.stem for f in seed_dir.glob("*.npy")}
        want = set(subset_names)
        missing = sorted(want - existing)

        if skip_seed_gen and not missing:
            log.info("[%s] seeds already present (%d), skipping gen",
                     mode, len(existing))
        else:
            log.info("[%s] generating seeds for %d imgs (TTA scales=%s)",
                     mode, len(subset_names), TTA_SCALES)
            # Only regenerate missing ones: build a pruned label_dict.
            target = {n: label_dict[n] for n in subset_names if n in label_dict}
            t0 = time.time()
            if mode == "cam_max":
                generate_all_cams(
                    model=model, label_dict=target,
                    image_dir=IMAGE_DIR, output_dir=seed_dir,
                    image_ext=IMAGE_EXT, scales=TTA_SCALES, input_size=448,
                    num_ref_images=1, binary_aggregate="max", device=device,
                    ref_pool=train_ref_pool, ref_image_dir=REF_IMAGE_DIR,
                    query_class_resolver=class_resolver,
                )
            else:
                generate_all_seeds(
                    model=model, label_dict=target,
                    image_dir=IMAGE_DIR, output_dir=seed_dir,
                    image_ext=IMAGE_EXT, scales=TTA_SCALES, input_size=448,
                    num_ref_images=1, seed_mode=mode, device=device,
                    ref_pool=train_ref_pool, ref_image_dir=REF_IMAGE_DIR,
                    query_class_resolver=class_resolver,
                )
            log.info("[%s] seed generation took %.0fs", mode, time.time() - t0)

        avail = sorted(n for n in subset_names if (seed_dir / f"{n}.npy").exists())
        log.info("[%s] %d seeds available", mode, len(avail))
        if len(avail) < len(subset_names):
            log.warning("[%s] missing %d seeds (not all images had valid forwards)",
                        mode, len(subset_names) - len(avail))

        # ---- threshold sweep ----
        # Parallelised across the 100 threshold values via
        # ``evaluate_cam_threshold_sweep(num_workers=...)``. Each threshold
        # is an independent ``evaluate_cam_miou`` call over all ``avail``
        # seeds; pure NumPy/PIL inside so no need for a per-task timeout.
        # Historical serial pass took ~4 min / mode on 1000 imgs; with
        # 16 workers it drops to ~15--30 s.
        log.info("[%s] threshold sweep (disease_iou optimise)...", mode)
        t0 = time.time()
        sweep = evaluate_cam_threshold_sweep(
            predict_dir=str(seed_dir), gt_dir=str(GT_DIR),
            name_list=avail, num_cls=2, optimize_metric="disease_iou",
            num_workers=crf_workers,
        )
        best_at = sweep.get("result_at_best", {})
        fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
        disease_thr = float(best_at[fg_keys[0]]) if fg_keys else 0.0
        best_thr = float(sweep["best_threshold"])
        thr_miou = float(best_at.get("mIoU", 0))
        thr_bg = float(best_at.get("background", 0))
        log.info("[%s] thr=%.2f  disease_iou=%.2f%%  bg_iou=%.2f%%  mIoU=%.2f%%  (%.0fs)",
                 mode, best_thr, disease_thr, thr_bg, thr_miou, time.time() - t0)

        # ---- CRF sweep on subset ----
        # Cache filename depends on pipeline so CAM-res and full-res caches
        # coexist without collision. Re-running with --fullres_sweep will
        # miss the CAM-res cache on purpose.
        sweep_kind = "fullres" if fullres_sweep else "camres"
        cache_fname = f"crf_top_{mode}_{sweep_kind}.json"
        crf_cache_path: Path | None = output_base / cache_fname
        if crf_cache_path.exists():
            cached = json.loads(crf_cache_path.read_text())
            top3_sweep = cached.get("top3", [])
            if top3_sweep:
                log.info(
                    "[%s] CRF sweep cached (%s, %d configs), skipping",
                    mode, sweep_kind, len(top3_sweep),
                )
            else:
                crf_cache_path = None
        else:
            crf_cache_path = None

        if crf_cache_path is None:
            log.info(
                "[%s] CRF sweep [%s] (%d imgs, %d configs)...",
                mode, sweep_kind, crf_sweep_images,
                len(CRF_SRGB) * len(CRF_BG_THR) * len(CRF_SCALE),
            )
            t0 = time.time()
            if fullres_sweep:
                crf_results = _sweep_crf_fullres(
                    seed_dir=seed_dir, image_dir=IMAGE_DIR, gt_dir=GT_DIR,
                    image_ext=IMAGE_EXT, num_cls=2,
                    srgb_values=CRF_SRGB, bg_thr_values=CRF_BG_THR,
                    scale_values=CRF_SCALE,
                    max_images=min(crf_sweep_images, len(avail)),
                    num_workers=crf_workers,
                    per_config_timeout_sec=fullres_sweep_cfg_timeout_sec,
                )
            else:
                crf_results = sweep_crf_params(
                    seed_dir=seed_dir, image_dir=IMAGE_DIR, gt_dir=GT_DIR,
                    image_ext=IMAGE_EXT, num_cls=2,
                    srgb_values=CRF_SRGB, bg_thr_values=CRF_BG_THR,
                    scale_values=CRF_SCALE,
                    max_images=min(crf_sweep_images, len(avail)),
                    num_workers=crf_workers,
                )
            log.info("[%s] CRF sweep took %.0fs", mode, time.time() - t0)
            top3_sweep = crf_results[:3]
            with open(output_base / cache_fname, "w") as f:
                json.dump({"kind": sweep_kind, "top3": top3_sweep}, f, indent=2)

        log.info(
            "[%s] CRF sweep top-3 [%s, %d imgs; metric=MICRO]:",
            mode, sweep_kind, crf_sweep_images,
        )
        for r in top3_sweep:
            log.info(
                "  srgb=%.0f bg=%.2f sc=%.1f  dis=%.2f%%  bg=%.2f%%  mIoU=%.2f%%",
                r["srgb"], r["bg_threshold"], r["scale_factor"],
                r["disease_iou"], r["bg_iou"], r["mIoU"],
            )

        per_mode[mode] = {
            "threshold": {
                "best_threshold": best_thr,
                "disease_iou": disease_thr,
                "bg_iou": thr_bg,
                "mIoU": thr_miou,
                "n_images": len(avail),
            },
            "crf_sweep_top3_on_subset": top3_sweep,
            "avail": avail,
        }

    # Pick cross-mode winner from the 250-image CRF sweep.
    best_mode = max(
        per_mode,
        key=lambda m: per_mode[m]["crf_sweep_top3_on_subset"][0]["disease_iou"],
    )
    winner_cfg = per_mode[best_mode]["crf_sweep_top3_on_subset"][0]
    log.info("[%s] cross-mode winner: %s srgb=%.0f bg=%.2f sc=%.1f  "
             "(sweep DisIoU %.2f%%)",
             run_cfg["name"], best_mode,
             winner_cfg["srgb"], winner_cfg["bg_threshold"], winner_cfg["scale_factor"],
             winner_cfg["disease_iou"])

    # Single full eval on all available images for the cross-mode winner.
    # Parallelised with per-image hard timeout. Prior to 2026-04-29 this
    # was serial, which deadlocked overnight orchestrators on pydensecrf
    # hangs (reproduced on 1.8 MP zucchini images that stall indefinitely
    # at srgb=5). See ``_fullres_eval_img_worker`` and the launch guide.
    seed_dir_best = output_base / f"seeds_{best_mode}"
    full_best = _full_crf_eval(
        seed_dir_best, per_mode[best_mode]["avail"],
        srgb=winner_cfg["srgb"], bg_thr=winner_cfg["bg_threshold"],
        sf=winner_cfg["scale_factor"],
        num_workers=crf_workers,
        per_image_timeout_sec=fullres_eval_img_timeout_sec,
    )
    log.info(
        "[%s] FULL subset CRF (n=%d, n_nonempty=%d): "
        "srgb=%.0f bg=%.2f sc=%.1f",
        run_cfg["name"], full_best["n_images"], full_best["n_images_nonempty"],
        full_best["srgb"], full_best["bg_threshold"], full_best["scale_factor"],
    )
    log.info(
        "  MICRO (primary, matches baseline 42.13%%): "
        "dis=%.2f%% bg=%.2f%% mIoU=%.2f%%",
        full_best["disease_iou_micro"], full_best["bg_iou_micro"],
        full_best["mIoU_micro"],
    )
    log.info(
        "  MACRO (per-image mean): "
        "dis=%.2f%% bg=%.2f%% mIoU=%.2f%%  (nonempty-only: %.2f%%)",
        full_best["disease_iou_macro"], full_best["bg_iou_macro"],
        full_best["mIoU_macro"], full_best["disease_iou_macro_nonempty"],
    )
    log.info(
        "  TAIL: median=%.2f%%  frac(DisIoU>=0.3)=%.2f  frac(DisIoU>=0.5)=%.2f",
        full_best["median_per_img_disease_iou"],
        full_best["frac_imgs_disease_iou_ge_0.3"],
        full_best["frac_imgs_disease_iou_ge_0.5"],
    )

    per_mode[best_mode]["crf_best_full"] = full_best
    # Drop the intermediate "avail" list before serialising.
    for mode_key in per_mode:
        per_mode[mode_key].pop("avail", None)

    summary = {
        "name": run_cfg["name"],
        "label": run_cfg["label"],
        "checkpoint": run_cfg["checkpoint"],
        "n_images_subset": len(subset_names),
        "tta_scales": TTA_SCALES,
        "best_seed_mode": best_mode,
        "per_mode": per_mode,
    }
    (output_base / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[%s] wrote %s", run_cfg["name"], output_base / "summary.json")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/d4_localization")
    p.add_argument("--subset_size", type=int, default=750)
    p.add_argument("--subset_seed", type=int, default=2024)
    p.add_argument("--crf_sweep_images", type=int, default=250)
    p.add_argument("--crf_workers", type=int, default=8)
    p.add_argument("--seed_modes", nargs="+", default=SEED_MODES_DEFAULT)
    p.add_argument("--only", nargs="*", default=None,
                    help="Optional: restrict to these run names (e.g. d4_ac_safe D3)")
    p.add_argument("--skip_seed_gen", action="store_true", default=True)
    p.add_argument("--no_skip_seed_gen", dest="skip_seed_gen", action="store_false")
    p.add_argument(
        "--fullres_sweep", action="store_true", default=True,
        help="Run CRF sweep at FULL image resolution (default). Matches the "
        "final full-res eval pipeline so the sweep winner IS the full-res "
        "winner. Expensive; pair with enough CRF workers.",
    )
    p.add_argument(
        "--camres_sweep", dest="fullres_sweep", action="store_false",
        help="Restore legacy CAM-resolution CRF sweep (cheap, but winner "
        "does not transfer to full-res).",
    )
    p.add_argument(
        "--fullres_sweep_cfg_timeout_sec",
        type=float,
        default=FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT,
        help="Per-config hard timeout (seconds) inside the fullres CRF sweep. "
        "pydensecrf can hang indefinitely on pathological inputs; configs "
        "exceeding this are dropped from the ranking. 0 disables (not "
        "recommended for unattended runs).",
    )
    p.add_argument(
        "--fullres_eval_img_timeout_sec",
        type=float,
        default=FULLRES_EVAL_IMG_TIMEOUT_SEC_DEFAULT,
        help="Per-image hard timeout (seconds) inside the final full-subset "
        "CRF refinement pass. Images exceeding it are skipped from the "
        "IoU averages and reported in the log.",
    )
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # ---- shared subset + reference pool (once, cached) ----
    subset_names = _select_subset(GT_DIR, args.subset_size, args.subset_seed)
    (out_root / "subset_names.json").write_text(json.dumps(subset_names, indent=2))
    log.info("selected %d val images (seed=%d)", len(subset_names), args.subset_seed)

    class_names = load_class_names(CLASS_NAMES_FILE)
    class_resolver = make_filename_class_resolver(class_names)
    train_ref_pool = build_class_pool_from_labels(
        LABEL_FILE, REF_IMAGE_DIR, image_ext=IMAGE_EXT,
    )
    refable = sum(
        1 for n in subset_names
        if class_resolver(n) is not None and len(train_ref_pool.get(class_resolver(n), [])) > 0
    )
    log.info("train ref pool covers %d/%d classes",
             len(train_ref_pool), NUM_CLASSES)
    log.info("val subset with >=1 same-class train reference: %d/%d",
             refable, len(subset_names))

    label_dict = _build_label_dict(subset_names, class_resolver, NUM_CLASSES)

    runs = RUNS_DEFAULT
    if args.only:
        runs = [r for r in runs if r["name"] in args.only]
        log.info("restricted to runs: %s", [r["name"] for r in runs])

    all_summaries: list[dict] = []
    for run_cfg in runs:
        try:
            s = _run_one_checkpoint(
                run_cfg=run_cfg,
                subset_names=subset_names,
                label_dict=label_dict,
                train_ref_pool=train_ref_pool,
                class_resolver=class_resolver,
                output_base=out_root / run_cfg["name"],
                seed_modes=args.seed_modes,
                crf_sweep_images=args.crf_sweep_images,
                crf_workers=args.crf_workers,
                device=device,
                skip_seed_gen=args.skip_seed_gen,
                fullres_sweep=args.fullres_sweep,
                fullres_sweep_cfg_timeout_sec=args.fullres_sweep_cfg_timeout_sec,
                fullres_eval_img_timeout_sec=args.fullres_eval_img_timeout_sec,
            )
            all_summaries.append(s)
        except Exception as e:
            log.exception("run %s failed: %s", run_cfg["name"], e)

    # Aggregate Markdown.
    md_lines = [
        f"# D4 localization evaluation -- {args.subset_size} images, TTA scales {TTA_SCALES}",
        "",
        "## Conventions",
        "",
        "* **Primary metric: MICRO (pixel-pooled) DisIoU** -- what `sweep_crf_params`",
        "  and the historical 42.13% feat_chmean+CRF baseline report. Low variance,",
        "  literature-standard, directly comparable.",
        "* Secondary: MACRO (per-image mean) DisIoU + tail diagnostics",
        "  (nonempty-only mean, fraction of images with IoU>=0.3/0.5, median).",
        "* Threshold-sweep DisIoU and CRF-sweep top-1 DisIoU are ALSO micro",
        "  (from `evaluate_cam_miou` and `sweep_crf_params`).",
        "",
        "## Per-checkpoint winner (full-subset re-eval of the CRF-sweep top-1)",
        "",
        "Columns: DisIoU_micro / mIoU_micro / DisIoU_macro / macro(nonempty-only) / median / frac(>=0.3) / frac(>=0.5).",
        "",
        "| run | seed | thr | DisIoU_thr | CRF (srgb/bg/sc) | **DisIoU_micro** | mIoU_micro | DisIoU_macro | macro_nonempty | median | frac>=0.3 | frac>=0.5 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        bm = s["best_seed_mode"]
        pm = s["per_mode"][bm]
        t = pm["threshold"]
        c = pm.get("crf_best_full", {})
        if not c:
            continue
        crf_str = f"s={c['srgb']:.0f} bg={c['bg_threshold']:.2f} sc={c['scale_factor']:.0f}"
        md_lines.append(
            f"| **{s['name']}** | {bm} | {t['best_threshold']:.2f} | "
            f"{t['disease_iou']:.2f}% | {crf_str} | "
            f"**{c['disease_iou_micro']:.2f}%** | {c['mIoU_micro']:.2f}% | "
            f"{c['disease_iou_macro']:.2f}% | "
            f"{c['disease_iou_macro_nonempty']:.2f}% | "
            f"{c['median_per_img_disease_iou']:.2f}% | "
            f"{c['frac_imgs_disease_iou_ge_0.3']:.2f} | "
            f"{c['frac_imgs_disease_iou_ge_0.5']:.2f} |"
        )

    md_lines += [
        "",
        "**Baseline reference (same metric as bold column):** token feat_chmean + CRF(srgb=5, bg=0.30) = **42.13%** DisIoU_micro / 60.87% mIoU_micro (on 200 val imgs, `evaluate_feature_seeds.py`).",
        "",
        "## All (run, seed mode) pairs -- CRF-sweep top-1 on 250 imgs (MICRO metric)",
        "",
        "| run | seed mode | thr | DisIoU_thr (micro) | CRF (srgb/bg/sc) | DisIoU_micro (sweep 250) | mIoU_micro (sweep 250) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for s in all_summaries:
        for mode, pm in s["per_mode"].items():
            t = pm["threshold"]
            top = pm["crf_sweep_top3_on_subset"][0]
            crf_str = f"s={top['srgb']:.0f} bg={top['bg_threshold']:.2f} sc={top['scale_factor']:.0f}"
            marker = "**" if mode == s["best_seed_mode"] else ""
            md_lines.append(
                f"| {marker}{s['name']}{marker} | {mode} | {t['best_threshold']:.2f} | "
                f"{t['disease_iou']:.2f}% | {crf_str} | "
                f"{top['disease_iou']:.2f}% | {top['mIoU']:.2f}% |"
            )

    (out_root / "summary.md").write_text("\n".join(md_lines) + "\n")
    (out_root / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    log.info("wrote %s and %s", out_root / "summary.md", out_root / "all_summaries.json")


if __name__ == "__main__":
    main()
