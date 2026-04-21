"""Evaluate one trained seg-probe + non-trainable baselines at the same position.

For each (checkpoint, position) pair the script:

  1. Loads the trained ProbeHead from `<probe_dir>/head.pt` and runs forward
     on the full PlantSeg val set with same-class references from train.
     Output: `<probe_dir>/probe_seeds/<name>.npy` (one per val image).

  2. ALSO runs three non-trainable baselines at the same position (cheap;
     same forward pass yields the activation tensor):
        * feat_chmean       -- channel mean
        * feat_chvar        -- channel variance
        * cam_classifier_max -- only at P5 (already in correct shape) or any
          256-ch position via classifier weight projection
     Output: `<probe_dir>/baseline_<key>_seeds/<name>.npy`.

  3. For every output, runs the standard pipeline:
        * threshold sweep   (find best disease-IoU threshold)
        * CRF param sweep on 200 imgs   (per-distribution tuning)
        * full CRF eval     (final disease/bg IoU)
        * visualizations    (25 imgs + summary grid, magenta GT, teal CRF)

  4. Writes `<probe_dir>/eval.json` with the four IoUs and a
     ``probe_underperforms`` flag (true if probe_iou < max(baselines) - 2pp).

Skip-if-exists: if `eval.json` already exists, the script returns 0 without
recomputing. Pass `--force` to overwrite.

Example:
    python scripts/eval_seg_probes.py \
        --probe-dir outputs/spdnet_plantseg/seg_probe_phase1/token_n1_heavy/P3_query_merged \
        --checkpoint outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def _open_exif_safe(path, mode: str = "RGB") -> Image.Image:
    """Open a PIL image and apply EXIF rotation so dimensions match its mask.

    ~0.1% of PlantSeg train images have EXIF orientation 6 (90 deg CW);
    without this, a reference image lands in the network with wrong WxH and
    the corresponding mask won't align. ``exif_transpose`` is a no-op when no
    EXIF orientation is set.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if mode is not None:
        img = img.convert(mode)
    return img
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.refinement.crf import apply_crf
from src.wsss.spdnet._atomic_io import atomic_save_npy, prune_corrupt_seeds
from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint
from src.wsss.spdnet.class_resolver import (
    build_class_pool_from_labels,
    load_class_names,
    make_filename_class_resolver,
)
from src.wsss.spdnet.seg_probe import (
    NEEDS_REFERENCE,
    PROBE_POSITIONS,
    SPATIAL_ONLY_POSITIONS,
    ProbeHead,
    SPDNetWithProbes,
    channels_for_position,
)
from scripts.sweep_crf_params import sweep_crf_params

IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
REF_IMAGE_DIR = Path("data/plantsegv3/images/train")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
CLASS_NAMES_FILE = "outputs/plantseg_binary_mc115/labels/class_names.txt"
NUM_CLASSES = 115

CRF_SWEEP_IMAGES_DEFAULT = 200
CRF_WORKERS_DEFAULT = 8
# Hard cap (seconds) per image during the full CRF eval. pydensecrf is C++
# and does not honour Python signals; the only way out of a pathological
# inference call is to abandon waiting on it. 300 s is ~5x the slowest
# normal image we've measured (a 16 MP wheat_stripe_rust image takes ~55 s
# at srgb=5, scale_factor=1.0) so any image that hits this cap is almost
# certainly hung, not slow.
CRF_EVAL_TIMEOUT_SEC_DEFAULT = 300.0
VIZ_COUNT_DEFAULT = 25
SMOKE_VIZ_COUNT = 5
SMOKE_CRF_IMAGES = 30

GT_CONTOUR_RGB = (255, 50, 220)
GT_COLOR = np.array([0.85, 0.15, 0.85])
CRF_COLOR = np.array([0.0, 0.75, 0.75])
PRED_COLOR = np.array([0.85, 0.25, 0.15])

PROBE_KEY = "probe"
BASELINE_KEYS = ("chmean", "chvar", "cam_cls")


def _normalize(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(x)


def _resize_seed(s: np.ndarray, w: int, h: int) -> np.ndarray:
    if s.shape == (h, w):
        return s
    return np.array(
        Image.fromarray(s.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    )


def _apply_crf_to_seed(img_np, cam_dict, h, w, srgb, bg_thr, scale):
    resized = {k: _resize_seed(v, w, h) for k, v in cam_dict.items()}
    probs = apply_crf(
        img_np, resized, bg_threshold=bg_thr, t=10,
        num_cls=2, scale_factor=scale, srgb=srgb,
    )
    return np.argmax(probs, axis=0).astype(np.uint8)


def _overlay_heatmap(img, heatmap, alpha=0.55):
    hm = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]
    bl = img.astype(np.float32) / 255 * (1 - alpha) + hm_c.astype(np.float32) / 255 * alpha
    return (np.clip(bl, 0, 1) * 255).astype(np.uint8)


def _overlay_mask(img, mask, color, alpha=0.40):
    r = img.astype(np.float32) / 255
    out = r.copy()
    fg = mask > 0
    out[fg] = out[fg] * (1 - alpha) + color * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _overlay_contour(img, mask, color=GT_CONTOUR_RGB, thickness=2):
    out = img.copy()
    m8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _baseline_aggregations(
    feat: torch.Tensor,
    cam_classifier: torch.Tensor | None,
    h_orig: int,
    w_orig: int,
    classifier_weight: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Compute the three non-trainable baseline maps from a single activation.

    Returns up to 3 per-image (H, W) float32 maps, normalised to [0,1].
    `cam_cls` is computed via classifier weight projection on 256-ch maps,
    or copied from `cam_classifier` if already a CAM (P5 input).
    """
    out: dict[str, np.ndarray] = {}

    if feat.dim() == 4 and feat.shape[1] >= 1:
        chmean = feat.mean(dim=1, keepdim=True)
        chmean = F.interpolate(chmean, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        out["chmean"] = _normalize(chmean[0, 0].cpu().numpy())

        if feat.shape[1] >= 2:
            chvar = feat.var(dim=1, keepdim=True)
            chvar = F.interpolate(chvar, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            out["chvar"] = _normalize(chvar[0, 0].cpu().numpy())

    if cam_classifier is not None:
        cmax = cam_classifier.amax(dim=1, keepdim=True)
        cmax = F.interpolate(cmax, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        out["cam_cls"] = _normalize(cmax[0, 0].cpu().numpy())
    elif classifier_weight is not None and feat.dim() == 4 and feat.shape[1] == classifier_weight.shape[1]:
        cam = F.relu(torch.einsum("nc,bchw->bnhw", classifier_weight, feat))
        cmax = cam.amax(dim=1, keepdim=True)
        cmax = F.interpolate(cmax, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        out["cam_cls"] = _normalize(cmax[0, 0].cpu().numpy())

    return out


@torch.no_grad()
def generate_probe_and_baselines(
    wrapper: SPDNetWithProbes,
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
    output_dirs: dict[str, Path],
    ref_pool: dict[int, list[str]],
    ref_image_dir: Path,
    query_class_resolver,
    image_ext: str = ".jpg",
    input_size: int = 448,
    max_long: int = 0,
    device: torch.device = torch.device("cpu"),
    skip_existing: bool = True,
) -> int:
    """Run one forward pass per image; save probe + (up to) 3 baseline seeds.

    `output_dirs` maps key -> directory, with key in
    {"probe", "chmean", "chvar", "cam_cls"}. Keys absent from this dict
    are skipped.
    """
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if max_long <= 0:
        max_long = int(input_size * 1.75)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    needs_ref = wrapper.position in NEEDS_REFERENCE
    classifier_weight = wrapper.spdnet.classifier.weight.detach()

    rng = random.Random(42)
    n_processed = 0
    for name in tqdm(list(label_dict.keys()), desc=f"probe+baselines({wrapper.position})"):
        img_path = image_dir / f"{name}{image_ext}"
        if not img_path.exists():
            continue

        keys_needed = list(output_dirs.keys())
        if skip_existing:
            keys_needed = [
                k for k in keys_needed
                if not (output_dirs[k] / f"{name}.npy").exists()
            ]
            if not keys_needed:
                n_processed += 1
                continue

        ref_cls = query_class_resolver(name)
        if ref_cls is None:
            ref_cls = int(np.argmax(label_dict[name]))
        ref_names = [n for n in ref_pool.get(ref_cls, []) if n != name]
        if not ref_names:
            ref_names = [name]
        ref_name = rng.choice(ref_names)

        query_pil = _open_exif_safe(img_path)
        long_side = max(query_pil.size)
        if long_side > max_long:
            r = max_long / long_side
            query_pil = query_pil.resize(
                (round(query_pil.width * r), round(query_pil.height * r)), Image.BICUBIC,
            )
        h_orig, w_orig = query_pil.height, query_pil.width

        ref_pil = _open_exif_safe(ref_image_dir / f"{ref_name}{image_ext}")
        ls = max(ref_pil.size)
        if ls > max_long:
            r = max_long / ls
            ref_pil = ref_pil.resize(
                (round(ref_pil.width * r), round(ref_pil.height * r)), Image.BICUBIC,
            )

        # SPDNet expects fixed input_size for proper FPN merging at training res
        # but extract_probe_features can take native res for max fidelity here.
        q_t = tfm(query_pil.resize((input_size, input_size), Image.BICUBIC)).unsqueeze(0).to(device)
        r_t = tfm(ref_pil.resize((input_size, input_size), Image.BICUBIC)).unsqueeze(0).to(device)
        refs = [r_t] if needs_ref else None

        feat, feats = wrapper.extract_features_at_position(q_t, refs)

        if PROBE_KEY in keys_needed:
            seg_logit = wrapper.head(feat)
            seg_prob = torch.sigmoid(seg_logit)
            seg_prob = F.interpolate(seg_prob, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            arr = seg_prob[0, 0].cpu().numpy().astype(np.float32)
            atomic_save_npy(output_dirs[PROBE_KEY] / f"{name}.npy", {0: arr})

        cam_at_p5 = feats.get("P5_cam_classifier")
        baselines = _baseline_aggregations(
            feat, cam_at_p5 if wrapper.position == "P5_cam_classifier" else None,
            h_orig, w_orig,
            classifier_weight=classifier_weight,
        )
        for key in BASELINE_KEYS:
            if key in keys_needed and key in baselines:
                atomic_save_npy(
                    output_dirs[key] / f"{name}.npy",
                    {0: baselines[key].astype(np.float32)},
                )

        n_processed += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return n_processed


# NOTE: This worker MUST stay at module top-level so multiprocessing can
# pickle it on Linux fork-mode. Keep it dependency-free w.r.t. live process
# state (no CUDA tensors, no Hydra config, no open file handles); everything
# it needs comes through the `args` tuple.
def _crf_eval_one_image_worker(
    args: tuple[str, str, str, str, str, dict],
) -> tuple[str, float, float]:
    """Run CRF + per-class IoU for ONE image. Picklable / pool-safe.

    Returns (name, disease_iou, bg_iou) -- IoUs in [0, 1] (NOT %); the parent
    multiplies by 100 once. Raising is fine -- the parent catches it via
    apply_async.get() so a single bad image never poisons the batch.
    """
    name, seed_dir_s, image_dir_s, gt_dir_s, image_ext, crf_p = args
    seed_dir = Path(seed_dir_s)
    image_dir = Path(image_dir_s)
    gt_dir = Path(gt_dir_s)

    img = np.array(_open_exif_safe(image_dir / f"{name}{image_ext}"))
    h, w = img.shape[:2]
    gt = np.array(_open_exif_safe(gt_dir / f"{name}.png", mode=None))
    if gt.shape[:2] != (h, w):
        gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
    cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
    crf_mask = _apply_crf_to_seed(
        img, cam_dict, h, w, crf_p["srgb"], crf_p["bg_threshold"], crf_p["scale_factor"],
    )
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (crf_mask > 0).astype(np.uint8)
    inter_d = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    union_d = int(((pred_bin == 1) | (gt_bin == 1)).sum())
    inter_b = int(((pred_bin == 0) & (gt_bin == 0)).sum())
    union_b = int(((pred_bin == 0) | (gt_bin == 0)).sum())
    d_iou = float(inter_d / union_d) if union_d > 0 else 1.0
    b_iou = float(inter_b / union_b) if union_b > 0 else 1.0
    return name, d_iou, b_iou


def _full_crf_eval(
    seed_dir: Path,
    available_names: list[str],
    crf_p: dict,
    image_ext: str = ".jpg",
    num_workers: int = 1,
    per_image_timeout_sec: float = 300.0,
    _worker_fn=None,
) -> tuple[float, float, float]:
    """Run CRF on every (image, seed) pair and return (disease_iou, bg_iou, miou) %.

    Parallelism contract:
        * num_workers == 1                  -> serial loop (debuggable, deterministic).
        * num_workers > 1 and len > 1       -> multiprocessing.Pool.apply_async
                                               with per-image .get(timeout=...).

    Per-image timeout: pydensecrf is a C++ extension that releases the GIL but
    doesn't honour Python signals; on rare pathological inputs it can spin
    indefinitely (observed empirically on a 1.8 MP zucchini image with
    srgb=5, scale_factor=1.0 -- the call ran for >55 min while every other
    image in the same batch finished in <60 s). We therefore enforce a HARD
    wall-clock cap per image: when ``apply_async.get(timeout=N)`` raises
    ``multiprocessing.TimeoutError`` we abandon that future and continue with
    the rest. ``Pool.__exit__`` then calls ``terminate()`` on the way out,
    killing any still-spinning worker.

    Worst case wall-clock when at most one image hangs is therefore
    ``max(N/W * t_avg, timeout) + small overhead`` rather than unbounded;
    the average IoU is computed over the images that *did* complete, and
    the count of skipped images is logged so it shows up in master.log.

    ``_worker_fn`` is a test seam: pass an alternate top-level callable
    matching the signature of ``_crf_eval_one_image_worker`` to inject
    deterministic / slow / failing behaviour. Production code leaves it
    None so the real CRF worker is used.
    """
    worker = _worker_fn if _worker_fn is not None else _crf_eval_one_image_worker
    tasks = [
        (name, str(seed_dir), str(IMAGE_DIR), str(GT_DIR), image_ext, crf_p)
        for name in available_names
    ]
    d_ious: list[float] = []
    b_ious: list[float] = []
    n_skipped = 0
    skipped_names: list[str] = []

    use_pool = num_workers > 1 and len(tasks) > 1
    if use_pool:
        from multiprocessing import Pool
        from multiprocessing import TimeoutError as MPTimeoutError

        with Pool(num_workers) as pool:
            asyncs = [pool.apply_async(worker, (t,)) for t in tasks]
            pbar = tqdm(total=len(asyncs), desc=f"CRF {seed_dir.name}")
            for ar, t in zip(asyncs, tasks):
                try:
                    _, d, b = ar.get(timeout=per_image_timeout_sec)
                    d_ious.append(d)
                    b_ious.append(b)
                except MPTimeoutError:
                    n_skipped += 1
                    skipped_names.append(t[0])
                    print(
                        f"[CRF] {seed_dir.name}: timeout >"
                        f"{per_image_timeout_sec:.0f}s on '{t[0]}' -- skipping",
                        flush=True,
                    )
                except Exception as e:
                    n_skipped += 1
                    skipped_names.append(t[0])
                    print(
                        f"[CRF] {seed_dir.name}: worker error on '{t[0]}': "
                        f"{e!r} -- skipping",
                        flush=True,
                    )
                pbar.update(1)
            pbar.close()
            # `with Pool(...)` -> __exit__ -> terminate() will SIGKILL any
            # still-spinning worker (including the one stuck on the timed-out
            # image). This is the whole reason we use apply_async + timeout
            # instead of imap_unordered.
    else:
        for t in tqdm(tasks, desc=f"CRF {seed_dir.name}"):
            try:
                _, d, b = worker(t)
                d_ious.append(d)
                b_ious.append(b)
            except Exception as e:
                n_skipped += 1
                skipped_names.append(t[0])
                print(
                    f"[CRF] {seed_dir.name}: error on '{t[0]}': {e!r} -- skipping",
                    flush=True,
                )

    if n_skipped > 0:
        # Surface in master.log so the operator notices we skipped images;
        # downstream consumers should treat these IoUs as computed over
        # `len(d_ious)` images, not `len(tasks)`.
        preview = ", ".join(skipped_names[:5])
        more = f" (+{n_skipped - 5} more)" if n_skipped > 5 else ""
        print(
            f"[CRF] {seed_dir.name}: SKIPPED {n_skipped}/{len(tasks)} image(s) "
            f"(timeout >{per_image_timeout_sec:.0f}s or worker error). "
            f"Mean IoU computed over {len(d_ious)}/{len(tasks)} images. "
            f"Skipped: {preview}{more}",
            flush=True,
        )
    if not d_ious:
        return float("nan"), float("nan"), float("nan")
    d, b = float(np.mean(d_ious)) * 100, float(np.mean(b_ious)) * 100
    return d, b, (d + b) / 2


def _evaluate_one_seed_dir(
    seed_dir: Path,
    available_names: list[str],
    crf_sweep_images: int,
    crf_workers: int,
    skip_crf_sweep: bool = False,
    per_image_timeout_sec: float = 300.0,
) -> dict:
    """Threshold sweep + CRF param sweep + full CRF eval for a seed dir."""
    # Threshold sweep: parallelised across the 100 thresholds (was serial
    # pre 2026-04-19; took ~4 min / dir × 4 dirs / probe = ~16 min wasted
    # on each probe). Each threshold's evaluate_cam_miou call is fully
    # independent so this is embarrassingly parallel; ~Wx speedup with W
    # workers. Pure NumPy / PIL inside, no C++ extension -> no need for
    # the per-image timeout we gave the CRF parallel pool.
    sweep = evaluate_cam_threshold_sweep(
        predict_dir=str(seed_dir), gt_dir=str(GT_DIR),
        name_list=available_names, num_cls=2, optimize_metric="disease_iou",
        num_workers=crf_workers,
    )
    best_at = sweep.get("result_at_best", {})
    fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
    disease_iou_thr = best_at[fg_keys[0]] if fg_keys else 0.0
    best_thr = sweep["best_threshold"]

    if skip_crf_sweep:
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}
        crf_top5: list[dict] = []
        crf_sweep_iou = float("nan")
    else:
        crf_results = sweep_crf_params(
            seed_dir=seed_dir, image_dir=IMAGE_DIR, gt_dir=GT_DIR,
            image_ext=".jpg", num_cls=2,
            max_images=crf_sweep_images, num_workers=crf_workers,
        )
        best = crf_results[0]
        crf_p = {
            "srgb": best["srgb"], "bg_threshold": best["bg_threshold"],
            "scale_factor": best["scale_factor"],
        }
        crf_top5 = crf_results[:5]
        crf_sweep_iou = best["disease_iou"]

    # Full CRF eval: parallelised + per-image timeout. Prior to this it was
    # serial, which deadlocked the overnight orchestrator on 2026-04-19 when
    # pydensecrf hung forever on a single 1.8 MP image (zucchini_downy_mildew_
    # Bing_0120) -- 200 min wasted before the bash watchdog's heartbeat
    # surfaced the stall.
    crf_disease, crf_bg, crf_miou = _full_crf_eval(
        seed_dir, available_names, crf_p,
        num_workers=crf_workers,
        per_image_timeout_sec=per_image_timeout_sec,
    )

    return {
        "best_threshold": best_thr,
        "threshold_disease_iou": disease_iou_thr,
        "threshold_bg_iou": best_at.get("background", 0.0),
        "threshold_miou": best_at.get("mIoU", 0.0),
        "crf_params": crf_p,
        "crf_sweep_disease_iou_subset": crf_sweep_iou,
        "crf_top5": crf_top5,
        "crf_disease_iou": crf_disease,
        "crf_bg_iou": crf_bg,
        "crf_miou": crf_miou,
        "num_eval_images": len(available_names),
    }


def _generate_visualizations(
    seed_dir: Path,
    threshold: float,
    crf_p: dict,
    viz_dir: Path,
    title: str,
    n: int,
) -> None:
    rng = random.Random(42)
    names = sorted(f.stem for f in seed_dir.glob("*.npy"))
    gt_avail = {f.stem for f in GT_DIR.glob("*.png")}
    names = [n for n in names if n in gt_avail]
    selected = sorted(rng.sample(names, min(n, len(names))))

    viz_dir.mkdir(parents=True, exist_ok=True)
    panels_all = []
    for name in tqdm(selected, desc=f"viz {viz_dir.name}"):
        img = np.array(_open_exif_safe(IMAGE_DIR / f"{name}.jpg"))
        h, w = img.shape[:2]
        gt = np.array(_open_exif_safe(GT_DIR / f"{name}.png", mode=None))
        if gt.shape[:2] != (h, w):
            gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
        cam_dict = np.load(str(seed_dir / f"{name}.npy"), allow_pickle=True).item()
        seed = _normalize(_resize_seed(cam_dict[0], w, h))
        binary = (seed > threshold).astype(np.uint8)
        crf_mask = _apply_crf_to_seed(
            img, cam_dict, h, w, crf_p["srgb"], crf_p["bg_threshold"], crf_p["scale_factor"],
        )
        iou_t = float(((binary > 0) & (gt > 0)).sum() / max(((binary > 0) | (gt > 0)).sum(), 1))
        iou_c = float(((crf_mask > 0) & (gt > 0)).sum() / max(((crf_mask > 0) | (gt > 0)).sum(), 1))

        p1 = _overlay_contour(_overlay_mask(img, gt, GT_COLOR), gt)
        p2 = _overlay_heatmap(img, seed)
        p3 = _overlay_contour(_overlay_mask(img, binary, PRED_COLOR), gt)
        p4 = _overlay_contour(_overlay_mask(img, crf_mask, CRF_COLOR), gt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=120)
        for ax, panel, ttl in zip(axes, [p1, p2, p3, p4], [
            "Original + GT",
            "Seed (heatmap)",
            f"Thr={threshold:.2f}  IoU={iou_t:.1%}",
            f"CRF(srgb={crf_p['srgb']:.0f})  IoU={iou_c:.1%}",
        ]):
            ax.imshow(panel); ax.set_title(ttl, fontsize=10); ax.axis("off")
        fig.suptitle(f"{title} | {name}", fontsize=11, fontweight="bold")
        plt.tight_layout(pad=0.3)
        fig.savefig(str(viz_dir / f"{name}.png"), dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        panels_all.append({"name": name, "panels": [p1, p2, p3, p4], "iou_t": iou_t, "iou_c": iou_c})

    if not panels_all:
        return
    n_rows = min(len(panels_all), 12)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.2 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = ["Original + GT", "Seed (heatmap)",
                  f"Threshold={threshold:.2f}", f"CRF(srgb={crf_p['srgb']:.0f})"]
    for r in range(n_rows):
        e = panels_all[r]
        for c in range(4):
            axes[r, c].imshow(e["panels"][c]); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_labels[c], fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(
            f"{e['name']}\nthr={e['iou_t']:.0%} crf={e['iou_c']:.0%}",
            fontsize=7, rotation=0, labelpad=120, va="center",
        )
    mt = np.mean([e["iou_t"] for e in panels_all])
    mc = np.mean([e["iou_c"] for e in panels_all])
    fig.suptitle(f"{title} | Mean: thr={mt:.1%}, CRF={mc:.1%}", fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout(pad=0.5)
    fig.savefig(str(viz_dir / "summary_grid.png"), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


SEED_DIR_NAMES = (
    "probe_seeds",
    "baseline_chmean_seeds",
    "baseline_chvar_seeds",
    "baseline_cam_cls_seeds",
)


def _cleanup_seed_dirs(probe_dir: Path) -> dict:
    """Delete per-image seed npy directories, leaving everything else intact.

    Called after a successful eval (eval.json + viz/ both written) to keep
    persisted artefacts under ~150 MB / probe instead of ~4.6 GB. Seeds are
    fully reproducible from ``head.pt`` + source SPDNet checkpoint, so the
    overnight orchestrator drops them as it goes; ad-hoc reruns (without
    ``--cleanup-seeds``) and any post-hoc deep dive on a winning probe can
    regenerate them by re-invoking this script with ``--force``.

    Sanity-guards:
      * Refuses to do anything if ``probe_dir/eval.json`` is missing -- means
        the eval did not complete cleanly and seeds may still be needed.
      * Only touches the four well-known directory names.
    """
    probe_dir = Path(probe_dir)
    if not (probe_dir / "eval.json").exists():
        return {"removed": [], "kept_reason": "eval.json missing"}

    removed: list[str] = []
    bytes_freed = 0
    for name in SEED_DIR_NAMES:
        sd = probe_dir / name
        if not sd.is_dir():
            continue
        # rough size estimate before delete
        try:
            for f in sd.glob("*.npy"):
                bytes_freed += f.stat().st_size
        except OSError:
            pass
        shutil.rmtree(sd)
        removed.append(name)

    return {
        "removed": removed,
        "bytes_freed": int(bytes_freed),
        "mb_freed": round(bytes_freed / (1024 * 1024), 1),
    }


def _subsample_val_names(val_names: list[str], n: int, seed: int = 1234) -> list[str]:
    """Deterministically pick *n* names without replacement, sorted alphabetically.

    Independent of model/probe so every probe evaluated with the same
    ``limit_val`` sees the *exact same* val subset -- cross-probe ranks remain
    valid. Returns the original list unchanged when ``n <= 0`` or
    ``n >= len(val_names)``.
    """
    if n <= 0 or n >= len(val_names):
        return val_names
    rng = random.Random(seed)
    return sorted(rng.sample(val_names, n))


def evaluate_probe(
    probe_dir: Path,
    checkpoint: str,
    crf_sweep_images: int = CRF_SWEEP_IMAGES_DEFAULT,
    crf_workers: int = CRF_WORKERS_DEFAULT,
    viz_count: int = VIZ_COUNT_DEFAULT,
    smoke: bool = False,
    force: bool = False,
    cleanup_seeds: bool = False,
    limit_val: int = 0,
    crf_eval_timeout_sec: float = CRF_EVAL_TIMEOUT_SEC_DEFAULT,
) -> dict:
    """Run the full eval for a single trained probe in *probe_dir*.

    Reads ``head.pt`` (and reads the position from it). Returns the
    eval-summary dict and writes it to ``probe_dir/eval.json``.

    When ``cleanup_seeds`` is True, the four ``*_seeds/`` subdirectories are
    deleted at the very end (only after eval.json + viz/ are both written).
    Seeds are fully reproducible from the saved checkpoint, so this is safe
    for the overnight orchestrator and reclaims ~4.5 GB / probe.

    When ``limit_val > 0`` (and not in smoke mode), the val set is
    deterministically subsampled to that size (seed=1234). The same subset
    is used by every probe with the same ``limit_val`` so rankings remain
    comparable. The recorded ``limit_val`` is written to ``eval.json`` so
    later analyses can flag full-val vs subset numbers.

    ``crf_eval_timeout_sec`` caps the wall-clock spent on any single image
    inside the full-val CRF pass; images that exceed it are skipped from
    the IoU average and reported in master.log. See `_full_crf_eval` for
    why this is needed (one-line answer: pydensecrf can hang forever).
    """
    probe_dir = Path(probe_dir)
    eval_path = probe_dir / "eval.json"
    if eval_path.exists() and not force:
        with open(eval_path) as f:
            return json.load(f)

    head_path = probe_dir / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"No head.pt at {head_path}")

    head_blob = torch.load(head_path, map_location="cpu", weights_only=False)
    position = head_blob["position"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spdnet = load_spdnet_from_checkpoint(checkpoint, num_classes=NUM_CLASSES)
    finetuned = probe_dir / "spdnet_finetuned.pt"
    if finetuned.exists():
        ft = torch.load(finetuned, map_location="cpu", weights_only=False)
        spdnet.load_state_dict(ft["spdnet_state_dict"], strict=True)
        print(f"[eval] Loaded fine-tuned SPDNet from {finetuned}")
    spdnet.eval()
    wrapper = SPDNetWithProbes(
        spdnet=spdnet,
        position=position,
        head_hidden_dim=head_blob.get("head_hidden_dim", 64),
        target_size=(448, 448),
        freeze_backbone=True,
    )
    wrapper.head.load_state_dict(head_blob["head_state_dict"], strict=True)
    wrapper = wrapper.to(device)
    wrapper.eval()

    fpn_channels = wrapper.spdnet.fpn_channels
    in_ch = channels_for_position(wrapper.spdnet, position)
    print(f"[eval] position={position}  in_ch={in_ch}  fusion={wrapper.spdnet.fusion_mode}")

    val_names_full = sorted(f.stem for f in GT_DIR.glob("*.png"))
    print(f"[eval] {len(val_names_full)} val images on disk")

    print("[eval] Building same-class reference pool from PlantSeg train…")
    ref_pool = build_class_pool_from_labels(
        LABEL_FILE, REF_IMAGE_DIR, image_ext=".jpg",
    )
    class_names = load_class_names(CLASS_NAMES_FILE)
    class_resolver = make_filename_class_resolver(class_names)

    label_dict = {}
    for name in val_names_full:
        cls = class_resolver(name)
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        lbl[cls if cls is not None else 0] = 1.0
        label_dict[name] = lbl

    if smoke:
        val_names = val_names_full[:25]
        label_dict = {k: label_dict[k] for k in val_names}
        crf_sweep_images = SMOKE_CRF_IMAGES
        viz_count = SMOKE_VIZ_COUNT
        effective_limit_val = len(val_names)
    else:
        val_names = _subsample_val_names(val_names_full, limit_val)
        if len(val_names) < len(val_names_full):
            label_dict = {k: label_dict[k] for k in val_names}
            print(
                f"[eval] limit_val={limit_val} -> sampling {len(val_names)}"
                f" of {len(val_names_full)} val images (deterministic, seed=1234)"
            )
        effective_limit_val = limit_val if limit_val > 0 else 0

    chvar_meaningful = in_ch >= 2
    cam_meaningful = (
        position == "P5_cam_classifier"
        or in_ch == fpn_channels  # 256-ch positions can be projected via classifier
    )

    output_dirs = {
        PROBE_KEY: probe_dir / "probe_seeds",
        "chmean": probe_dir / "baseline_chmean_seeds",
    }
    if chvar_meaningful:
        output_dirs["chvar"] = probe_dir / "baseline_chvar_seeds"
    if cam_meaningful:
        output_dirs["cam_cls"] = probe_dir / "baseline_cam_cls_seeds"

    # Heal any corrupt/half-written seeds left by previous runs (atomic
    # writes prevent new ones; this handles files written before the fix
    # plus the rare case of disk hiccups on already-renamed files).
    # Without this guard, generate_probe_and_baselines(skip_existing=True)
    # would treat a truncated .npy as "done" and the eval loop would die
    # with `_pickle.UnpicklingError: pickle data was truncated`.
    pruned_total = 0
    for sd in output_dirs.values():
        pruned = prune_corrupt_seeds(sd)
        if pruned:
            print(f"[eval] pruned {len(pruned)} corrupt seed(s) from {sd.name}")
            for p in pruned[:5]:
                print(f"         - {p.name}")
            if len(pruned) > 5:
                print(f"         ... and {len(pruned) - 5} more")
            pruned_total += len(pruned)
    if pruned_total:
        print(f"[eval] total pruned corrupt seeds across all dirs: {pruned_total}")

    print(f"[eval] generating seeds for keys: {list(output_dirs.keys())}")
    t0 = time.time()
    n = generate_probe_and_baselines(
        wrapper=wrapper, label_dict=label_dict,
        image_dir=IMAGE_DIR, output_dirs=output_dirs,
        ref_pool=ref_pool, ref_image_dir=REF_IMAGE_DIR,
        query_class_resolver=class_resolver,
        device=device, skip_existing=True,
    )
    print(f"[eval] processed {n} images in {time.time() - t0:.0f}s")

    eval_results: dict[str, dict] = {}
    for key, sd in output_dirs.items():
        avail = [n for n in val_names if (sd / f"{n}.npy").exists()]
        print(f"[eval] {key}: {len(avail)} seeds  ->  threshold + CRF sweep")
        eval_results[key] = _evaluate_one_seed_dir(
            seed_dir=sd, available_names=avail,
            crf_sweep_images=crf_sweep_images,
            crf_workers=crf_workers,
            skip_crf_sweep=smoke,
            per_image_timeout_sec=crf_eval_timeout_sec,
        )

    # Decide who underperforms (probe vs best baseline, post-CRF)
    probe_iou = eval_results.get(PROBE_KEY, {}).get("crf_disease_iou", 0.0)
    baseline_ious = [
        eval_results[k]["crf_disease_iou"]
        for k in BASELINE_KEYS if k in eval_results
    ]
    max_baseline = max(baseline_ious) if baseline_ious else 0.0
    probe_underperforms = bool(
        probe_iou + 2.0 < max_baseline and len(baseline_ious) > 0
    )

    summary = {
        "probe_dir": str(probe_dir),
        "checkpoint": checkpoint,
        "position": position,
        "fusion_mode": wrapper.spdnet.fusion_mode,
        "needs_reference": position in NEEDS_REFERENCE,
        "channels_in": in_ch,
        "probe_iou": probe_iou,
        "chmean_iou": eval_results.get("chmean", {}).get("crf_disease_iou"),
        "chvar_iou": eval_results.get("chvar", {}).get("crf_disease_iou"),
        "cam_cls_iou": eval_results.get("cam_cls", {}).get("crf_disease_iou"),
        "probe_underperforms": probe_underperforms,
        "score_S": float(max(probe_iou, max_baseline)),
        # Subsampling marker -- 0 means full val, >0 means deterministic
        # subset of that size (seed=1234). Cross-probe ranks are only valid
        # when this matches between probes.
        "limit_val": int(effective_limit_val),
        "n_val_used": int(len(val_names)),
        "details": eval_results,
    }
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {eval_path}")

    print("[eval] generating visualizations…")
    viz_root = probe_dir / "viz"
    for key in [PROBE_KEY] + [k for k in BASELINE_KEYS if k in eval_results]:
        info = eval_results[key]
        sd = output_dirs[key]
        title = f"{Path(probe_dir).parts[-2]}/{position}/{key}"
        _generate_visualizations(
            seed_dir=sd, threshold=info["best_threshold"],
            crf_p=info["crf_params"], viz_dir=viz_root / key,
            title=title, n=viz_count,
        )

    if cleanup_seeds:
        info = _cleanup_seed_dirs(probe_dir)
        if info["removed"]:
            print(
                f"[eval] cleaned up seed dirs ({info['mb_freed']} MB): "
                + ", ".join(info["removed"])
            )

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate one trained seg probe.")
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--crf-sweep-images", type=int, default=CRF_SWEEP_IMAGES_DEFAULT)
    ap.add_argument("--crf-workers", type=int, default=CRF_WORKERS_DEFAULT)
    ap.add_argument("--viz-count", type=int, default=VIZ_COUNT_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--cleanup-seeds",
        action="store_true",
        help=(
            "After eval.json + viz/ are written, delete the four "
            "*_seeds/ directories (~4.5 GB / probe). Seeds are reproducible "
            "from head.pt + the source SPDNet checkpoint, so this is safe "
            "for the overnight orchestrator."
        ),
    )
    ap.add_argument(
        "--limit-val",
        type=int,
        default=0,
        help=(
            "If >0, deterministically subsample the val set to this many "
            "images (seed=1234). All steps -- threshold sweep, CRF tuning, "
            "full CRF eval, viz -- run on the subset. Use for fast probe "
            "screening (300 imgs ~5x faster than 1247, ranks stable). The "
            "value is recorded in eval.json so cross-probe comparisons can "
            "verify the same val subset was used. 0 = full val (default)."
        ),
    )
    ap.add_argument(
        "--crf-eval-timeout-sec",
        type=float,
        default=CRF_EVAL_TIMEOUT_SEC_DEFAULT,
        help=(
            "Per-image hard timeout (seconds) inside the full-val CRF pass. "
            "pydensecrf can hang forever on rare pathological images "
            "(observed 2026-04-19: zucchini_downy_mildew_Bing_0120 stuck "
            ">55 min while every other image finished in <60 s). When the "
            "timeout fires, the image is dropped from the IoU average and "
            "the count is logged. 0 = no timeout (don't use for batch jobs)."
        ),
    )
    args = ap.parse_args()

    summary = evaluate_probe(
        probe_dir=args.probe_dir,
        checkpoint=args.checkpoint,
        crf_sweep_images=args.crf_sweep_images,
        crf_workers=args.crf_workers,
        viz_count=args.viz_count,
        smoke=args.smoke,
        force=args.force,
        cleanup_seeds=args.cleanup_seeds,
        limit_val=args.limit_val,
        crf_eval_timeout_sec=args.crf_eval_timeout_sec,
    )

    print("\n" + "=" * 70)
    print(f"  Position: {summary['position']}  (in_ch={summary['channels_in']}, fusion={summary['fusion_mode']})")
    print("=" * 70)
    print(f"  Probe   IoU(CRF): {summary['probe_iou']:.2f}%")
    if summary["chmean_iou"] is not None:
        print(f"  chmean  IoU(CRF): {summary['chmean_iou']:.2f}%")
    if summary["chvar_iou"] is not None:
        print(f"  chvar   IoU(CRF): {summary['chvar_iou']:.2f}%")
    if summary["cam_cls_iou"] is not None:
        print(f"  cam_cls IoU(CRF): {summary['cam_cls_iou']:.2f}%")
    print(f"  Score S: {summary['score_S']:.2f}%   probe_underperforms={summary['probe_underperforms']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
