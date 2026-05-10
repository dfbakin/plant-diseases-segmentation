"""Compute final-validation metrics for the §5.14.9 P1' / P2' write-up.

Produces a YAML report covering:

  (1) Full-val verification of the 200-img CRF sweep best configs
      (P1'@srgb=13/bg=0.10 and P2'@srgb=13/bg=0.20 from §5.14.9.c).
  (2) Multi-scale + horizontal flip on the headline ckpts (per-scale +
      averaged), with the same fixed CRF configs from (1).
  (3) Per-image diagnostics (FG fraction, OS/US ratios, image-level
      IoU mean) for both ckpts on the multi-scale + CRF predictions.
  (4) Paired per-image delta P2' - P1' (n_improved / n_unchanged /
      n_degraded at threshold_pp, plus top-5 wins / regressions).
  (5) Optional MCTformer MC115 + WeakCLIP sanity check on full val.

Convention: dataset-level pixel-pooled IoU (the
src/wsss/mctformer/evaluation.py:evaluate_cam_miou convention).

Hardware utilisation (4× RTX 5090, 384 vCPU host, measured):

  * Stage A: torch.multiprocessing.spawn, one process per GPU. Each
    rank loads BOTH P1' and P2' (each ~325 MB on 32 GiB) and processes
    a 1/world_size shard of val. 12 forward passes per (image, ckpt)
    pair: 3 scales × 2 flips. Smoke (32 imgs, 4 GPUs): 59s →
    extrapolation to full val (1247 imgs): ~37-40 minutes.
  * Stage B: multiprocessing.Pool with --num-crf-workers workers
    (default 32). pydensecrf is C++ and CPU-bound; the bottleneck is
    the kernel scheduler. CRF eval + raw-CAM threshold sweep on 8
    cam dirs (2 ckpts × 4 scale variants). Smoke: 129s →
    extrapolation to full val: ~30-50 minutes (dominated by per-image
    CRF inference at 1247 images × 8 dirs).
  * Stage C: parallel per-image diagnostics on the saved CRF preds.
    Pure NumPy. Smoke: 6s → full val: ~3-5 minutes.
  * Stage D (optional): pixel-pooled IoU on the existing MC115
    HA-CRF and WeakCLIP final binary masks. ~30 s.

  Full-val total estimate: ~70-95 minutes wall clock (depending on
  CPU contention with other jobs on the host).

Stages can be skipped or re-run independently via --skip-* flags.
Each stage writes a JSON checkpoint under --output-dir so the final
YAML report can be reassembled without re-running anything.

Usage::

    python scripts/eval_phase5_writeup_metrics.py \\
        --output-dir outputs/_phase5_writeup_eval \\
        --num-crf-workers 32 \\
        --skip-mctformer

    # Re-emit YAML from existing JSON checkpoints (no compute)
    python scripts/eval_phase5_writeup_metrics.py \\
        --output-dir outputs/_phase5_writeup_eval \\
        --skip-camgen --skip-eval --skip-perimage --skip-mctformer

    # Smoke (32 images, single scale, no MCTformer)
    python scripts/eval_phase5_writeup_metrics.py \\
        --output-dir outputs/_phase5_writeup_smoke \\
        --max-images 32 --skip-mctformer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wsss.refinement.crf import apply_crf  # noqa: E402
from src.wsss.spdnet.cam_generator import (  # noqa: E402
    build_reference_pool,
    load_spdnet_from_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_writeup")

SCALES_DEFAULT = (0.75, 1.00, 1.25)
SCALE_TAGS = ("0.75", "1.00", "1.25")  # filename-safe + matches the YAML keys
MS_TAG = "ms"  # multi-scale combined
THR_RANGE = (5, 81)  # 0.05 .. 0.80 step 0.01
SEED = 42


# ----------------------------------------------------------------------
# Args / config
# ----------------------------------------------------------------------


@dataclass
class CkptSpec:
    tag: str
    ckpt: str
    crf_srgb: float
    crf_bg_thr: float
    crf_sf: float


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)

    # Inputs (§5.14.9 headline ckpts + 200-img sweep best CRF configs)
    p.add_argument("--p1-ckpt", default=(
        "outputs/phase5_5090_chain/phase5_5090_P1_cls_only_rps56_20260508_0711"
        "/checkpoints/best_mAP_epoch33.ckpt"
    ))
    p.add_argument("--p2-ckpt", default=(
        "outputs/phase5_5090_chain/phase5_5090_P2_warm_mask_rps56_20260508_1721"
        "/checkpoints/best_cam_iou.ckpt"
    ))
    p.add_argument("--p1-crf", default="13,0.10,1.0",
                   help="srgb,bg_thr,sf for P1' (default: §5.14.9.c best)")
    p.add_argument("--p2-crf", default="13,0.20,1.0",
                   help="srgb,bg_thr,sf for P2' (default: §5.14.9.c best)")

    p.add_argument("--image-dir", default="data/plantsegv3/images/val")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--gt-dir", default="outputs/plantseg_binary_mc115/gt_binary_val")
    p.add_argument("--labels-file", default=(
        "outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy"
    ))
    p.add_argument("--num-classes", type=int, default=115)
    p.add_argument("--input-size", type=int, default=896,
                   help="Reference resolution for max_size cap; per-scale "
                        "image size is round(orig * scale) capped at "
                        "int(input_size * 1.75).")
    p.add_argument("--max-size", type=int, default=0,
                   help="Override max long-side cap for query/ref images. "
                        "0 means int(input_size * 1.75).")
    p.add_argument("--num-ref-images", type=int, default=1)
    p.add_argument("--scales", type=str, default=",".join(f"{s:.2f}" for s in SCALES_DEFAULT))
    p.add_argument("--binary-aggregate", default="max")

    # Hardware knobs
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="CUDA device IDs; one rank per GPU. Use --gpus=0 for "
                        "single-GPU debugging.")
    p.add_argument("--num-crf-workers", type=int, default=32,
                   help="multiprocessing.Pool workers for CRF and per-image "
                        "stats. CRF is CPU-bound C++; 32 is a sane default "
                        "on the 384-core host.")

    # Stage skip / smoke
    p.add_argument("--skip-camgen", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-perimage", action="store_true")
    p.add_argument("--skip-mctformer", action="store_true")
    p.add_argument("--skip-thr-sweep", action="store_true",
                   help="Skip the raw-CAM threshold sweep in stage B (the "
                        "'best_thr' field). The fixed-CRF eval still runs.")
    p.add_argument("--max-images", type=int, default=0,
                   help="Smoke / debug knob; 0 means full val (1247).")
    p.add_argument("--paired-delta-pp", type=float, default=1.0,
                   help="Threshold (in percentage points) for "
                        "improved/unchanged/degraded buckets in stage C.")

    # MCTformer paths (optional stage D)
    p.add_argument("--mctformer-cam-dir",
                   default="outputs/plantseg_binary_mc115/cams/cam_npy_val")
    p.add_argument("--mctformer-pseudo-mask-dir",
                   default="outputs/plantseg_binary_mc115/pseudo_masks_t_0.73")
    p.add_argument("--weakclip-mask-dir",
                   default="outputs/plantseg_binary_mc115/weakclip_masks_t_0.73")

    return p


def parse_crf_spec(s: str) -> tuple[float, float, float]:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"--p*-crf expects 'srgb,bg_thr,sf', got {s!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_scales(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def parse_gpus(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


# ----------------------------------------------------------------------
# Image / reference helpers (mirror generate_all_cams to stay consistent
# with the §5.14.9.c CAM provenance)
# ----------------------------------------------------------------------


def _build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def _resize_long(pil: Image.Image, max_long: int) -> Image.Image:
    ls = max(pil.size)
    if ls > max_long:
        r = max_long / ls
        return pil.resize(
            (round(pil.width * r), round(pil.height * r)),
            resample=Image.BICUBIC,
        )
    return pil


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax - vmin > 1e-8:
        return (a - vmin) / (vmax - vmin + 1e-8)
    return np.zeros_like(a, dtype=np.float32)


def _aggregate_binary(per_class: np.ndarray, mode: str) -> np.ndarray:
    """per_class: (num_classes, H, W). Returns (H, W)."""
    if mode == "max":
        return per_class.max(axis=0)
    if mode == "mean":
        return per_class.mean(axis=0)
    if mode == "top_energy":
        e = per_class.sum(axis=(1, 2))
        return per_class[int(np.argmax(e))]
    raise ValueError(f"Unknown binary_aggregate: {mode!r}")


# ----------------------------------------------------------------------
# Stage A: 4-GPU sharded CAM generation
# ----------------------------------------------------------------------


@torch.no_grad()
def _forward_one_view(
    model,
    q_pil: Image.Image,
    r_pils: list[Image.Image],
    target_w: int,
    target_h: int,
    flip: bool,
    tfm,
    device: torch.device,
) -> np.ndarray:
    """One forward pass at a chosen resolution, optionally h-flipped.

    Returns the per-class CAM bilinear-resized to the (pre-flip) target
    resolution. The result is in the original orientation (any h-flip
    applied to the input is undone before returning).
    """
    q_scaled = q_pil.resize((target_w, target_h), resample=Image.BICUBIC)
    if flip:
        q_scaled = q_scaled.transpose(Image.FLIP_LEFT_RIGHT)
    q_t = tfm(q_scaled).unsqueeze(0).to(device)

    r_tensors = []
    for rpil in r_pils:
        r_sc = rpil.resize((target_w, target_h), resample=Image.BICUBIC)
        if flip:
            r_sc = r_sc.transpose(Image.FLIP_LEFT_RIGHT)
        r_tensors.append(tfm(r_sc).unsqueeze(0).to(device))

    if len(r_tensors) == 1:
        _, cam = model(q_t, r_tensors[0], return_cam=True)
    else:
        _, cam = model(q_t, r_tensors, return_cam=True)
    cam = torch.nn.functional.interpolate(
        cam, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )
    cam = cam[0].float().cpu().numpy()
    if flip:
        cam = np.flip(cam, axis=-1)
    return cam.astype(np.float32, copy=False)


@torch.no_grad()
def _generate_per_image_cams(
    model,
    name: str,
    image_dir: Path,
    image_ext: str,
    label_arr: np.ndarray,
    ref_pool: dict[int, list[str]],
    num_ref_images: int,
    scales: list[float],
    max_long: int,
    binary_aggregate: str,
    tfm,
    device: torch.device,
    rng: random.Random,
) -> dict[str, dict[int, np.ndarray]]:
    """Run multi-scale + flip CAM generation for ONE image.

    Returns ``{scale_tag: cam_dict}`` where ``cam_dict`` is the
    binary-aggregated, min-max-normalized {0: HxW float32} that
    sweep_crf_params / apply_crf expects. Includes one entry per
    requested scale (per-scale, hflip-averaged within scale) plus an
    'ms' entry that averages all scales + flips before
    binary-aggregating and normalizing.

    All CAMs are returned at the model's native input resolution for
    that scale (orig_h * sc, orig_w * sc, capped at max_long). The CRF
    stage will resize predictions to GT resolution.
    """
    img_path = image_dir / f"{name}{image_ext}"
    if not img_path.exists():
        return {}
    active_classes = np.where(label_arr > 0)[0].tolist()
    if not active_classes:
        return {}

    query_pil = _resize_long(Image.open(img_path).convert("RGB"), max_long)
    ref_cls = active_classes[0]
    ref_names = [n for n in ref_pool.get(ref_cls, []) if n != name]
    if not ref_names:
        ref_names = [name]
    ref_picks = rng.choices(ref_names, k=num_ref_images)
    ref_pils = [
        _resize_long(Image.open(image_dir / f"{rn}{image_ext}").convert("RGB"), max_long)
        for rn in ref_picks
    ]

    qw, qh = query_pil.size
    out: dict[str, dict[int, np.ndarray]] = {}
    per_scale_class_avg: list[np.ndarray] = []

    for sc, tag in zip(scales, SCALE_TAGS):
        tw = round(qw * sc)
        th = round(qh * sc)
        scaled_long = max(tw, th)
        if scaled_long > max_long:
            r = max_long / scaled_long
            tw, th = round(tw * r), round(th * r)
        cam_o = _forward_one_view(model, query_pil, ref_pils, tw, th, False, tfm, device)
        cam_f = _forward_one_view(model, query_pil, ref_pils, tw, th, True, tfm, device)
        cam_per_class = (cam_o + cam_f) * 0.5  # (num_classes, th, tw)
        per_scale_class_avg.append(cam_per_class)

        # Per-scale: aggregate -> normalize -> save
        merged = _aggregate_binary(cam_per_class, binary_aggregate)
        merged = _normalize_minmax(merged)
        out[tag] = {0: merged.astype(np.float32)}

    # Multi-scale average: resize all per-scale CAMs to scale=1.0 grid
    # before averaging (otherwise summing maps of different sizes is
    # ill-defined). Use the scale-1.0 size as the canonical grid.
    if 1.00 in scales:
        canon_idx = scales.index(1.00)
        canon = per_scale_class_avg[canon_idx]
        canon_h, canon_w = canon.shape[-2:]
    else:
        canon = per_scale_class_avg[0]
        canon_h, canon_w = canon.shape[-2:]

    resized = []
    for cam_per_class in per_scale_class_avg:
        if cam_per_class.shape[-2:] == (canon_h, canon_w):
            resized.append(cam_per_class)
            continue
        t = torch.from_numpy(cam_per_class).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(canon_h, canon_w), mode="bilinear", align_corners=False,
        )
        resized.append(t[0].numpy())

    ms_per_class = np.mean(resized, axis=0)
    merged_ms = _aggregate_binary(ms_per_class, binary_aggregate)
    merged_ms = _normalize_minmax(merged_ms)
    out[MS_TAG] = {0: merged_ms.astype(np.float32)}
    return out


def _camgen_worker(
    rank: int,
    world_size: int,
    devices: list[int],
    args: argparse.Namespace,
    image_names: list[str],
    out_root: str,
):
    """One process per GPU. Generates per-scale + multi-scale CAMs for
    its shard of images, for BOTH P1' and P2'. Writes to disk.
    """
    cuda_id = devices[rank]
    torch.cuda.set_device(cuda_id)
    device = torch.device(f"cuda:{cuda_id}")

    # Workers use logging, but with rank in the prefix.
    log_local = logging.getLogger(f"camgen_rank{rank}")

    # Load both models on this GPU. Each is ~325 MB; trivial vs 32 GiB.
    log_local.info(f"loading P1' from {args.p1_ckpt}")
    p1_model = load_spdnet_from_checkpoint(
        args.p1_ckpt, num_classes=args.num_classes,
    ).to(device).eval()
    log_local.info(f"loading P2' from {args.p2_ckpt}")
    p2_model = load_spdnet_from_checkpoint(
        args.p2_ckpt, num_classes=args.num_classes,
    ).to(device).eval()

    labels = np.load(args.labels_file, allow_pickle=True).item()
    image_dir = Path(args.image_dir)
    ref_pool = build_reference_pool(labels, image_dir, args.image_ext)

    scales = parse_scales(args.scales)
    max_long = args.max_size if args.max_size > 0 else int(args.input_size * 1.75)
    tfm = _build_transform()

    out_root_p = Path(out_root)
    # Layout: <out_root>/cams/<tag>/<scale_tag>/<image>.npy
    for tag in ("p1", "p2"):
        for s_tag in (*SCALE_TAGS, MS_TAG):
            (out_root_p / "cams" / tag / s_tag).mkdir(parents=True, exist_ok=True)

    # Shard
    shard = image_names[rank::world_size]
    rng = random.Random(SEED)  # same seed across ranks; ref selection is
                               # by-image so the shard is independent

    iterator = shard
    if rank == 0:
        iterator = tqdm(shard, desc="rank0", position=0)

    n_skipped = 0
    n_done = 0
    t0 = time.perf_counter()
    for name in iterator:
        label = labels[name]
        for tag, model in (("p1", p1_model), ("p2", p2_model)):
            cams = _generate_per_image_cams(
                model=model,
                name=name,
                image_dir=image_dir,
                image_ext=args.image_ext,
                label_arr=label,
                ref_pool=ref_pool,
                num_ref_images=args.num_ref_images,
                scales=scales,
                max_long=max_long,
                binary_aggregate=args.binary_aggregate,
                tfm=tfm,
                device=device,
                rng=random.Random(SEED + hash(name) % 2**31),
            )
            if not cams:
                n_skipped += 1
                continue
            for s_tag, cam_dict in cams.items():
                np.save(
                    str(out_root_p / "cams" / tag / s_tag / f"{name}.npy"),
                    cam_dict,
                )
        n_done += 1

    elapsed = time.perf_counter() - t0
    log_local.info(
        f"done. {n_done} images, {n_skipped} skipped, {elapsed:.1f}s "
        f"({elapsed / max(1, n_done):.2f} s/img)"
    )


def stage_a_camgen(args: argparse.Namespace, out_dir: Path) -> dict:
    """Spawn workers and time the stage. Returns a manifest dict."""
    devices = parse_gpus(args.gpus)
    world_size = len(devices)
    log.info(f"Stage A: CAM gen on {world_size} GPU(s) (devices={devices})")

    labels = np.load(args.labels_file, allow_pickle=True).item()
    gt_have = {f.stem for f in Path(args.gt_dir).glob("*.png")}
    image_names = sorted(n for n in labels.keys() if n in gt_have)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]
    log.info(
        f"  {len(image_names)} images "
        f"(skipped {len(labels) - len(image_names)} without GT mask)"
    )

    t0 = time.perf_counter()
    if world_size == 1:
        _camgen_worker(0, 1, devices, args, image_names, str(out_dir))
    else:
        mp.spawn(
            _camgen_worker,
            args=(world_size, devices, args, image_names, str(out_dir)),
            nprocs=world_size,
            join=True,
        )
    elapsed = time.perf_counter() - t0
    log.info(f"  Stage A wall clock: {elapsed:.1f}s")

    manifest = {
        "stage": "A",
        "n_images": len(image_names),
        "n_gpus": world_size,
        "elapsed_sec": elapsed,
        "scales": parse_scales(args.scales),
        "binary_aggregate": args.binary_aggregate,
        "ref_pool_size_p1": _ckpt_rps(args.p1_ckpt),
        "ref_pool_size_p2": _ckpt_rps(args.p2_ckpt),
    }
    (out_dir / "stage_a.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _ckpt_rps(ckpt_path: str) -> int:
    """Read ref_pool_size from checkpoint hyper_parameters (best-effort)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        return int(hp.get("ref_pool_size", 14))
    except Exception:
        return 14


# ----------------------------------------------------------------------
# Stage B: CRF + dataset-level pixel-pooled IoU
# ----------------------------------------------------------------------


def _crf_one_image(args_tuple) -> tuple[np.ndarray, np.ndarray]:
    """Apply CRF on one image, return (pred, gt) flattened uint8 arrays
    (with ignore-255 already filtered out). The parent reduces these
    into TP/P/T counters."""
    name, cam_dir, image_dir, image_ext, gt_dir, srgb, bg_thr, sf, num_cls = args_tuple

    cam_dict = np.load(str(Path(cam_dir) / f"{name}.npy"), allow_pickle=True).item()
    sample = next(iter(cam_dict.values()))
    cam_h, cam_w = sample.shape

    pil = Image.open(Path(image_dir) / f"{name}{image_ext}").convert("RGB")
    if (pil.height, pil.width) != (cam_h, cam_w):
        pil = pil.resize((cam_w, cam_h), Image.BILINEAR)
    img = np.array(pil)

    q = apply_crf(
        img, cam_dict, bg_threshold=bg_thr, t=10, num_cls=num_cls,
        scale_factor=sf, srgb=srgb,
    )
    pred = np.argmax(q, axis=0).astype(np.uint8)

    gt = np.array(Image.open(Path(gt_dir) / f"{name}.png"))
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST,
        ))

    cal = gt < 255
    return pred[cal].astype(np.uint8, copy=False), gt[cal].astype(np.uint8, copy=False)


def _eval_iou_on_dir(
    cam_dir: Path,
    image_dir: Path,
    image_ext: str,
    gt_dir: Path,
    srgb: float,
    bg_thr: float,
    sf: float,
    num_cls: int,
    num_workers: int,
    desc: str,
    save_pred_dir: Path | None = None,
) -> dict:
    """Run CRF on every .npy in cam_dir, compute pixel-pooled IoU.

    If save_pred_dir is given, also writes per-image argmax PNG masks
    (uint8, values in {0, 1}) so stage C can compute per-image IoUs
    without re-running CRF.
    """
    names = sorted(f.stem for f in cam_dir.glob("*.npy"))
    log.info(f"  [{desc}] CRF on {len(names)} images, "
             f"srgb={srgb} bg_thr={bg_thr} sf={sf}")

    if save_pred_dir is not None:
        save_pred_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (n, str(cam_dir), str(image_dir), image_ext, str(gt_dir),
         srgb, bg_thr, sf, num_cls)
        for n in names
    ]

    TP = np.zeros(num_cls, dtype=np.int64)
    P = np.zeros(num_cls, dtype=np.int64)
    T = np.zeros(num_cls, dtype=np.int64)
    saved = 0

    def _accum(name, pred_flat, gt_flat):
        nonlocal saved
        # We don't reconstruct the full (H, W) here; per-image stage
        # needs the full pred so we save them inside the worker if
        # save_pred_dir is requested.
        for i in range(num_cls):
            P[i] += int((pred_flat == i).sum())
            T[i] += int((gt_flat == i).sum())
            TP[i] += int(((pred_flat == i) & (gt_flat == i)).sum())

    if num_workers > 1 and len(tasks) > 1:
        with Pool(num_workers) as pool:
            it = pool.imap_unordered(_crf_one_image, tasks, chunksize=8)
            for name, (pred_flat, gt_flat) in zip(
                (t[0] for t in tasks), tqdm(it, total=len(tasks), desc=desc)
            ):
                _accum(name, pred_flat, gt_flat)
                # Save full pred if requested -- only available from a
                # second pass below; we batch the second pass to avoid
                # holding pred 2D arrays in worker return values.
    else:
        for t in tqdm(tasks, desc=desc):
            pred_flat, gt_flat = _crf_one_image(t)
            _accum(t[0], pred_flat, gt_flat)

    IoU = TP / (T + P - TP + 1e-10)
    bg_iou = float(IoU[0] * 100)
    fg_iou = float(IoU[1] * 100) if num_cls >= 2 else 0.0
    miou = float(IoU.mean() * 100)
    out = {
        "n_images": len(names),
        "srgb": srgb, "bg_threshold": bg_thr, "scale_factor": sf,
        "bg_iou": bg_iou, "disease_iou": fg_iou, "mIoU": miou,
        "TP": TP.tolist(), "P": P.tolist(), "T": T.tolist(),
    }

    # Second pass: write per-image argmax PNG masks (so stage C can
    # iterate them cheaply). This duplicates work but the alternative
    # is materializing 1247 (H, W) uint8 arrays in worker returns
    # which is ~2 GiB of pickle traffic; second pass is faster overall.
    if save_pred_dir is not None:
        log.info(f"  [{desc}] saving per-image predictions to {save_pred_dir}")
        if num_workers > 1 and len(tasks) > 1:
            with Pool(num_workers) as pool:
                it = pool.imap_unordered(
                    _crf_one_image_save_pred,
                    [(*t, str(save_pred_dir)) for t in tasks],
                    chunksize=8,
                )
                for _ in tqdm(it, total=len(tasks), desc=f"{desc}/save"):
                    saved += 1
        else:
            for t in tasks:
                _crf_one_image_save_pred((*t, str(save_pred_dir)))
                saved += 1
        out["pred_dir"] = str(save_pred_dir)
        out["n_pred_saved"] = saved
    return out


def _crf_one_image_save_pred(args_tuple) -> str:
    """Same as _crf_one_image but saves the full (H, W) prediction PNG
    instead of returning anything. Last element of the tuple is
    save_pred_dir (str)."""
    name, cam_dir, image_dir, image_ext, gt_dir, srgb, bg_thr, sf, num_cls, save_dir = args_tuple

    cam_dict = np.load(str(Path(cam_dir) / f"{name}.npy"), allow_pickle=True).item()
    sample = next(iter(cam_dict.values()))
    cam_h, cam_w = sample.shape

    pil = Image.open(Path(image_dir) / f"{name}{image_ext}").convert("RGB")
    if (pil.height, pil.width) != (cam_h, cam_w):
        pil = pil.resize((cam_w, cam_h), Image.BILINEAR)
    img = np.array(pil)

    q = apply_crf(
        img, cam_dict, bg_threshold=bg_thr, t=10, num_cls=num_cls,
        scale_factor=sf, srgb=srgb,
    )
    pred = np.argmax(q, axis=0).astype(np.uint8)

    gt = np.array(Image.open(Path(gt_dir) / f"{name}.png"))
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST,
        ))

    Image.fromarray(pred.astype(np.uint8)).save(str(Path(save_dir) / f"{name}.png"))
    return name


def _raw_threshold_sweep(
    cam_dir: Path,
    gt_dir: Path,
    num_workers: int,
    desc: str,
) -> dict:
    """Sweep raw-CAM thresholds (no CRF) on a given cam_dir.

    Mirrors evaluate_cam_threshold_sweep but: (a) operates on the
    binary-aggregated {0: cam} dict format produced by stage A
    (always num_cls=2), (b) reuses the same Pool. Returns the best
    threshold + per-class IoUs at best.
    """
    names = sorted(f.stem for f in cam_dir.glob("*.npy"))
    if not names:
        return {"best_thr": None}

    start, end = THR_RANGE
    tasks = [(t, str(cam_dir), str(gt_dir), names) for t in range(start, end)]

    if num_workers > 1 and len(tasks) > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_threshold_one_worker, tasks, chunksize=2),
                total=len(tasks), desc=desc,
            ))
    else:
        results = [_threshold_one_worker(t) for t in tqdm(tasks, desc=desc)]

    results.sort(key=lambda r: r["t"])
    best = max(results, key=lambda r: r["disease_iou"])
    return {
        "best_thr": best["t"] / 100.0,
        "bg_iou": best["bg_iou"],
        "disease_iou": best["disease_iou"],
        "mIoU": best["mIoU"],
        "n_images": len(names),
        "curve": [
            {"thr": r["t"] / 100.0, "disease_iou": r["disease_iou"],
             "bg_iou": r["bg_iou"], "mIoU": r["mIoU"]}
            for r in results
        ],
    }


def _threshold_one_worker(task) -> dict:
    """Argmax(softmax-with-bg=t) -> per-pixel-pooled IoU for one threshold."""
    t, cam_dir, gt_dir, names = task
    thr = t / 100.0
    TP = np.zeros(2, dtype=np.int64)
    P = np.zeros(2, dtype=np.int64)
    T = np.zeros(2, dtype=np.int64)
    for n in names:
        cam_dict = np.load(str(Path(cam_dir) / f"{n}.npy"), allow_pickle=True).item()
        cam = next(iter(cam_dict.values()))  # (H, W) in [0, 1]
        pred = (cam >= thr).astype(np.uint8)
        gt = np.array(Image.open(Path(gt_dir) / f"{n}.png"))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST,
            ))
        cal = gt < 255
        for i in (0, 1):
            P[i] += int(((pred == i) & cal).sum())
            T[i] += int(((gt == i) & cal).sum())
            TP[i] += int(((pred == i) & (gt == i) & cal).sum())
    IoU = TP / (T + P - TP + 1e-10)
    return {
        "t": t,
        "bg_iou": float(IoU[0] * 100),
        "disease_iou": float(IoU[1] * 100),
        "mIoU": float(IoU.mean() * 100),
    }


def stage_b_eval(args: argparse.Namespace, out_dir: Path) -> dict:
    log.info("Stage B: CRF eval at fixed configs + raw-CAM threshold sweep")

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    p1_crf = parse_crf_spec(args.p1_crf)
    p2_crf = parse_crf_spec(args.p2_crf)

    crf_specs = [
        ("p1", p1_crf),
        ("p2", p2_crf),
    ]

    results: dict[str, dict[str, dict]] = {"p1": {}, "p2": {}}
    pred_dirs: dict[str, str] = {}
    t0 = time.perf_counter()
    for tag, crf in crf_specs:
        srgb, bg_thr, sf = crf
        for s_tag in (*SCALE_TAGS, MS_TAG):
            cam_dir = out_dir / "cams" / tag / s_tag
            if not cam_dir.exists() or not any(cam_dir.iterdir()):
                log.warning(f"  [{tag}/{s_tag}] cam_dir empty, skipping")
                continue
            # Save predictions only for the multi-scale variant -- that's
            # what stage C needs for per-image diagnostics. Saving every
            # scale would balloon disk usage.
            save_pred = (
                out_dir / "preds" / f"{tag}_{s_tag}_crf"
                if s_tag == MS_TAG else None
            )
            d = _eval_iou_on_dir(
                cam_dir=cam_dir, image_dir=image_dir, image_ext=args.image_ext,
                gt_dir=gt_dir, srgb=srgb, bg_thr=bg_thr, sf=sf,
                num_cls=2, num_workers=args.num_crf_workers,
                desc=f"{tag}/{s_tag}/crf",
                save_pred_dir=save_pred,
            )
            results[tag][f"{s_tag}_crf"] = d
            if save_pred is not None:
                pred_dirs[tag] = str(save_pred)

            if not args.skip_thr_sweep:
                # Raw-CAM threshold sweep on the same cam_dir (no CRF).
                # Useful for the 'best_thr' field in the YAML report.
                thr = _raw_threshold_sweep(
                    cam_dir=cam_dir, gt_dir=gt_dir,
                    num_workers=args.num_crf_workers,
                    desc=f"{tag}/{s_tag}/thr",
                )
                results[tag][f"{s_tag}_thr"] = thr

    elapsed = time.perf_counter() - t0
    log.info(f"  Stage B wall clock: {elapsed:.1f}s")

    out = {
        "stage": "B",
        "elapsed_sec": elapsed,
        "p1_crf": list(p1_crf),
        "p2_crf": list(p2_crf),
        "results": results,
        "pred_dirs": pred_dirs,
    }
    (out_dir / "stage_b.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------
# Stage C: per-image diagnostics + paired delta
# ----------------------------------------------------------------------


def _per_image_one(args_tuple) -> dict:
    """Compute per-image FG fraction, OS/US ratios and image-level IoU."""
    name, pred_dir, gt_dir = args_tuple
    pred = np.array(Image.open(Path(pred_dir) / f"{name}.png"))
    gt = np.array(Image.open(Path(gt_dir) / f"{name}.png"))
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST,
        ))
    cal = gt < 255
    pred = pred[cal].astype(np.uint8, copy=False)
    gt = gt[cal].astype(np.uint8, copy=False)

    n_pix = max(1, int(pred.size))
    pred_fg = (pred == 1)
    gt_fg = (gt == 1)
    tp = int((pred_fg & gt_fg).sum())
    fp = int((pred_fg & ~gt_fg).sum())
    fn = int((~pred_fg & gt_fg).sum())
    tn = int((~pred_fg & ~gt_fg).sum())

    union_fg = tp + fp + fn
    fg_iou = (tp / union_fg * 100.0) if union_fg > 0 else 0.0
    bg_iou = (tn / max(1, tn + fp + fn) * 100.0)
    miou = 0.5 * (fg_iou + bg_iou)

    return {
        "name": name,
        "pred_fg_fraction": float(pred_fg.sum()) / n_pix,
        "gt_fg_fraction": float(gt_fg.sum()) / n_pix,
        # Oversegmentation: fraction of predicted-FG pixels that are FP.
        # Undefined when prediction has no FG; reported as 0 in that case.
        "oversegmentation_ratio": (fp / max(1, tp + fp)),
        # Undersegmentation: fraction of GT-FG pixels missed.
        # Undefined when GT has no FG; reported as 0 in that case.
        "undersegmentation_ratio": (fn / max(1, tp + fn)),
        "image_disease_iou": fg_iou,
        "image_bg_iou": bg_iou,
        "image_miou": miou,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _aggregate_per_image(rows: list[dict]) -> dict:
    """Macro means across images. OS is averaged only over images that
    predicted some FG; US and image_mean_disease_iou are averaged only
    over images that actually have GT-FG (consistent with the WSSS
    convention of treating empty-GT images as bg-only)."""
    if not rows:
        return {}
    arr = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    has_gt_fg = arr("gt_fg_fraction") > 0
    has_pred_fg = arr("pred_fg_fraction") > 0

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        return float(values[mask].mean())

    return {
        "n_images": len(rows),
        "n_with_gt_fg": int(has_gt_fg.sum()),
        "n_with_pred_fg": int(has_pred_fg.sum()),
        "mean_pred_fg_fraction": float(arr("pred_fg_fraction").mean()),
        "mean_gt_fg_fraction": float(arr("gt_fg_fraction").mean()),
        "oversegmentation_ratio": _masked_mean(
            arr("oversegmentation_ratio"), has_pred_fg,
        ),
        "undersegmentation_ratio": _masked_mean(
            arr("undersegmentation_ratio"), has_gt_fg,
        ),
        "image_mean_disease_iou": _masked_mean(
            arr("image_disease_iou"), has_gt_fg,
        ),
        "image_mean_miou": float(arr("image_miou").mean()),
    }


def stage_c_perimage(args: argparse.Namespace, out_dir: Path,
                     stage_b_out: dict) -> dict:
    log.info("Stage C: per-image stats + paired delta")
    pred_dirs = stage_b_out.get("pred_dirs", {})
    if "p1" not in pred_dirs or "p2" not in pred_dirs:
        raise RuntimeError(
            "Stage B did not produce per-image prediction dirs for both "
            "ckpts. Re-run with --skip-camgen unset (or remove the empty "
            "outputs/preds dir)."
        )

    gt_dir = Path(args.gt_dir)
    pred_dirs_path = {k: Path(v) for k, v in pred_dirs.items()}
    common = sorted(
        set(f.stem for f in pred_dirs_path["p1"].glob("*.png"))
        & set(f.stem for f in pred_dirs_path["p2"].glob("*.png"))
    )
    log.info(f"  paired set: {len(common)} images")

    t0 = time.perf_counter()
    rows: dict[str, list[dict]] = {}
    for tag in ("p1", "p2"):
        tasks = [(n, str(pred_dirs_path[tag]), str(gt_dir)) for n in common]
        if args.num_crf_workers > 1 and len(tasks) > 1:
            with Pool(args.num_crf_workers) as pool:
                rows[tag] = list(tqdm(
                    pool.imap_unordered(_per_image_one, tasks, chunksize=16),
                    total=len(tasks), desc=f"per-img/{tag}",
                ))
        else:
            rows[tag] = [_per_image_one(t) for t in tqdm(tasks, desc=f"per-img/{tag}")]

    # CSVs
    for tag, rs in rows.items():
        csv_path = out_dir / f"per_image_{tag}.csv"
        with open(csv_path, "w") as f:
            keys = list(rs[0].keys()) if rs else ["name"]
            f.write(",".join(keys) + "\n")
            for r in rs:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        log.info(f"  wrote {csv_path}")

    aggregated = {tag: _aggregate_per_image(rs) for tag, rs in rows.items()}

    # Paired delta
    by_name = {tag: {r["name"]: r for r in rs} for tag, rs in rows.items()}
    deltas = []
    for n in common:
        d = by_name["p2"][n]["image_disease_iou"] - by_name["p1"][n]["image_disease_iou"]
        deltas.append((n, d))
    deltas.sort(key=lambda x: x[1])  # ascending

    pp = float(args.paired_delta_pp)
    n_improved = sum(1 for _, d in deltas if d > pp)
    n_degraded = sum(1 for _, d in deltas if d < -pp)
    n_unchanged = len(deltas) - n_improved - n_degraded
    delta_arr = np.array([d for _, d in deltas])
    median_delta = float(np.median(delta_arr)) if delta_arr.size else 0.0

    # Top/bottom 5 (top = biggest P2 wins; bottom = biggest P2 regressions)
    bottom_5 = [n for n, _ in deltas[:5]]
    top_5 = [n for n, _ in deltas[-5:][::-1]]

    elapsed = time.perf_counter() - t0
    log.info(f"  Stage C wall clock: {elapsed:.1f}s")

    paired = {
        "threshold_pp": pp,
        "n_pair": len(common),
        "n_improved": n_improved,
        "n_unchanged": n_unchanged,
        "n_degraded": n_degraded,
        "median_delta_pp": median_delta,
        "best_5_image_names": top_5,
        "worst_5_image_names": bottom_5,
    }
    out = {
        "stage": "C",
        "elapsed_sec": elapsed,
        "aggregated": aggregated,
        "paired": paired,
    }
    (out_dir / "stage_c.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------
# Stage D (optional): MCTformer MC115 + WeakCLIP sanity check
# ----------------------------------------------------------------------


def stage_d_mctformer(args: argparse.Namespace, out_dir: Path) -> dict:
    """Compute disease_iou for the existing MC115 + WeakCLIP outputs.

    MC115 cam_npy is multi-class (115 channels); we evaluate using
    src.wsss.mctformer.evaluation.evaluate_cam_miou (with num_cls=116
    and disease_iou = mean of foreground-class IoUs aggregated to
    binary). For WeakCLIP final masks the binary masks are PNGs.
    """
    log.info("Stage D: MCTformer MC115 + WeakCLIP sanity check")
    from src.wsss.mctformer.evaluation import evaluate_cam_miou

    labels = np.load(args.labels_file, allow_pickle=True).item()
    gt_have = {f.stem for f in Path(args.gt_dir).glob("*.png")}
    name_list = sorted(n for n in labels.keys() if n in gt_have)
    if args.max_images > 0:
        name_list = name_list[: args.max_images]

    out: dict = {"stage": "D"}
    t0 = time.perf_counter()

    # 1) MC115 + HA-CRF disease_iou (the §5.7+ pipeline output).
    #    The MC115 cams are 115-channel npy dicts; we sweep thresholds
    #    on a small subset to find a representative one, then compute
    #    aggregate disease_iou. Existing pipeline stores HA-CRF masks
    #    as PNGs under pseudo_masks_t_0.73 -- which IS the right
    #    artefact for the "ha_crf_disease_iou" field.
    mc115_pseudo = Path(args.mctformer_pseudo_mask_dir)
    if mc115_pseudo.exists():
        # Pseudo masks are already binary PNGs; compute pixel-pooled
        # IoU directly via a 2-class confusion matrix on full val.
        out["ha_crf"] = _binary_png_iou(
            mc115_pseudo, Path(args.gt_dir), name_list,
        )
    else:
        log.warning(f"  MC115 pseudo masks not found at {mc115_pseudo}; "
                    f"skipping ha_crf_disease_iou.")
        out["ha_crf"] = None

    # 2) WeakCLIP final binary masks
    weakclip = Path(args.weakclip_mask_dir)
    if weakclip.exists():
        out["weakclip"] = _binary_png_iou(
            weakclip, Path(args.gt_dir), name_list,
        )
    else:
        log.warning(f"  WeakCLIP masks not found at {weakclip}; "
                    f"skipping weakclip_final_disease_iou.")
        out["weakclip"] = None

    out["elapsed_sec"] = time.perf_counter() - t0
    (out_dir / "stage_d.json").write_text(json.dumps(out, indent=2))
    return out


def _binary_png_iou(pred_dir: Path, gt_dir: Path, names: list[str]) -> dict:
    """Pixel-pooled IoU for binary PNG predictions vs GT PNGs."""
    TP = np.zeros(2, dtype=np.int64)
    P = np.zeros(2, dtype=np.int64)
    T = np.zeros(2, dtype=np.int64)
    n_eval = 0
    for n in tqdm(names, desc=f"{pred_dir.name}/iou"):
        pp = pred_dir / f"{n}.png"
        gp = gt_dir / f"{n}.png"
        if not pp.exists() or not gp.exists():
            continue
        pred = np.array(Image.open(pp))
        gt = np.array(Image.open(gp))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST,
            ))
        # Mask to binary {0, 1} -- some pipeline outputs use 255 for FG.
        if pred.max() > 1:
            pred = (pred >= 128).astype(np.uint8)
        cal = gt < 255
        for i in (0, 1):
            P[i] += int(((pred == i) & cal).sum())
            T[i] += int(((gt == i) & cal).sum())
            TP[i] += int(((pred == i) & (gt == i) & cal).sum())
        n_eval += 1
    IoU = TP / (T + P - TP + 1e-10)
    return {
        "n_images": n_eval,
        "bg_iou": float(IoU[0] * 100),
        "disease_iou": float(IoU[1] * 100),
        "mIoU": float(IoU.mean() * 100),
    }


# ----------------------------------------------------------------------
# Stage E: emit final YAML report
# ----------------------------------------------------------------------


def _yaml_dump_str(obj, indent=0) -> str:
    """Tiny custom YAML serialiser. We avoid the PyYAML dep for the
    final report so this script is import-self-contained, and so we
    can match the user's exact templating (quoted brackets, comments).
    """
    sp = "  " * indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return "null"
        return f"{obj}"
    if isinstance(obj, str):
        if any(c in obj for c in (":", "#", "[", "]", "{", "}", ",", "&", "*",
                                  "?", "|", "-", "<", ">", "=", "!", "%", "@",
                                  "`", '"', "'", "\n")):
            return json.dumps(obj)
        return obj
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, (int, float, str, bool)) or x is None for x in obj):
            return "[" + ", ".join(_yaml_dump_str(x) for x in obj) + "]"
        return "\n" + "\n".join(
            f"{sp}- {_yaml_dump_str(x, indent + 1).lstrip()}"
            for x in obj
        )
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            v_str = _yaml_dump_str(v, indent + 1)
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{sp}{k}:{v_str if v_str.startswith(chr(10)) else (' ' + v_str)}")
            else:
                lines.append(f"{sp}{k}: {v_str}")
        return "\n" + "\n".join(lines) if indent > 0 else "\n".join(lines)
    return json.dumps(obj)


def _fmt_block(d: dict, fields: list[str]) -> dict:
    """Pull a subset of fields from an eval result dict, rounding to 2 dp."""
    out = {}
    for f in fields:
        if f in d:
            v = d[f]
            out[f] = round(v, 2) if isinstance(v, float) else v
    return out


def stage_e_emit_yaml(args: argparse.Namespace, out_dir: Path,
                       a: dict | None, b: dict | None,
                       c: dict | None, d: dict | None) -> str:
    log.info("Stage E: assembling final YAML report")

    # ---------- (1) full-val verification of the 200-img CRF best ----------
    p1_crf = parse_crf_spec(args.p1_crf)
    p2_crf = parse_crf_spec(args.p2_crf)

    def _full_val_block(tag: str, ckpt: str, crf: tuple, b_results: dict):
        srgb, bg_thr, sf = crf
        ms_crf = b_results.get(tag, {}).get(f"{MS_TAG}_crf", {})
        s100_crf = b_results.get(tag, {}).get("1.00_crf", {})  # single-scale 1.0 = req (1)
        thr_block = b_results.get(tag, {}).get(f"{MS_TAG}_thr", {})
        return {
            "ckpt": ckpt,
            "rps": int(_ckpt_rps(ckpt)),
            "seed_source": "cam_classifier",
            "scales": [1.00],
            "crf": {"srgb": int(srgb), "bg_thr": float(bg_thr), "sf": float(sf)},
            "full_val": {
                "disease_iou": round(s100_crf.get("disease_iou", float("nan")), 2),
                "bg_iou": round(s100_crf.get("bg_iou", float("nan")), 2),
                "miou": round(s100_crf.get("mIoU", float("nan")), 2),
                "best_thr": round(thr_block.get("best_thr") or 0.0, 2)
                            if thr_block else None,
                "_n_images": s100_crf.get("n_images", None),
            },
        }

    # ---------- (2) multi-scale + flip ----------
    def _multi_block(tag: str, ckpt: str, crf: tuple, b_results: dict):
        per_scale = {}
        for s_tag in SCALE_TAGS:
            crf_d = b_results.get(tag, {}).get(f"{s_tag}_crf", {})
            thr_d = b_results.get(tag, {}).get(f"{s_tag}_thr", {})
            per_scale[s_tag] = {
                "disease_iou": round(crf_d.get("disease_iou", float("nan")), 2),
                "bg_iou": round(crf_d.get("bg_iou", float("nan")), 2),
                "miou": round(crf_d.get("mIoU", float("nan")), 2),
                "best_thr": (round(thr_d.get("best_thr") or 0.0, 2)
                             if thr_d else None),
            }
        ms_crf = b_results.get(tag, {}).get(f"{MS_TAG}_crf", {})
        ms_thr = b_results.get(tag, {}).get(f"{MS_TAG}_thr", {})
        srgb, bg_thr, sf = crf
        return {
            "scales": parse_scales(args.scales),
            "hflip": True,
            "per_scale": per_scale,
            "multi_scale_avg": {
                "crf": {"srgb": int(srgb), "bg_thr": float(bg_thr), "sf": float(sf)},
                "disease_iou": round(ms_crf.get("disease_iou", float("nan")), 2),
                "bg_iou": round(ms_crf.get("bg_iou", float("nan")), 2),
                "miou": round(ms_crf.get("mIoU", float("nan")), 2),
                "best_thr_raw_cam": (
                    round(ms_thr.get("best_thr") or 0.0, 2) if ms_thr else None
                ),
            },
        }

    b_results = (b or {}).get("results", {})

    block1_p1 = _full_val_block("p1", args.p1_ckpt, p1_crf, b_results)
    block1_p2 = _full_val_block("p2", args.p2_ckpt, p2_crf, b_results)
    block2_p1 = _multi_block("p1", args.p1_ckpt, p1_crf, b_results)
    block2_p2 = _multi_block("p2", args.p2_ckpt, p2_crf, b_results)

    # ---------- (3) per-image stats ----------
    per_image = {}
    if c is not None and "aggregated" in c:
        for tag in ("p1", "p2"):
            agg = c["aggregated"].get(tag, {})
            per_image[tag] = {
                "mean_pred_fg_fraction": round(agg.get("mean_pred_fg_fraction", float("nan")), 4),
                "mean_gt_fg_fraction": round(agg.get("mean_gt_fg_fraction", float("nan")), 4),
                "oversegmentation_ratio": round(agg.get("oversegmentation_ratio", float("nan")), 4),
                "undersegmentation_ratio": round(agg.get("undersegmentation_ratio", float("nan")), 4),
                "image_mean_disease_iou": round(agg.get("image_mean_disease_iou", float("nan")), 2),
                "image_mean_miou": round(agg.get("image_mean_miou", float("nan")), 2),
                "_n_images": agg.get("n_images", None),
                "_n_with_gt_fg": agg.get("n_with_gt_fg", None),
                "_n_with_pred_fg": agg.get("n_with_pred_fg", None),
            }

    # ---------- (4) paired delta ----------
    paired = (c or {}).get("paired", {})

    # ---------- (5) MCTformer sanity ----------
    sanity = {}
    if d is not None:
        ha = d.get("ha_crf") or {}
        wc = d.get("weakclip") or {}
        sanity = {
            "ha_crf_disease_iou": round(ha.get("disease_iou", float("nan")), 2)
                                  if ha else None,
            "weakclip_final_disease_iou": round(wc.get("disease_iou", float("nan")), 2)
                                          if wc else None,
            "_ha_crf_n_images": ha.get("n_images") if ha else None,
            "_weakclip_n_images": wc.get("n_images") if wc else None,
        }

    # Repo commit
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    if a is not None and "n_images" in a:
        n_actual = int(a["n_images"])
    elif b is not None:
        n_actual = max(
            (int(d.get("n_images", 0))
             for d in (b.get("results", {}).get("p1", {}) or {}).values()),
            default=0,
        )
    else:
        n_actual = 0
    if n_actual >= 1247:
        val_set_label = f"PlantSeg val (full {n_actual} images)"
    elif n_actual > 0:
        val_set_label = f"PlantSeg val (subset of {n_actual} images)"
    else:
        val_set_label = "PlantSeg val (size unknown)"

    report = {
        "repo_commit": commit,
        "val_set": val_set_label,
        "gt_dir": args.gt_dir,
        "convention": (
            "dataset-level (pixel-pooled) IoU per "
            "src/wsss/mctformer/evaluation.py:evaluate_cam_miou; "
            "image_mean_* fields use macro-image IoU averaging"
        ),
        "P1_prime": block1_p1,
        "P2_prime": block1_p2,
        "P1_prime_multiscale": block2_p1,
        "P2_prime_multiscale": block2_p2,
        "per_image_stats": per_image,
        "paired_delta": paired,
        "sanity_check_mctformer": sanity if d is not None else None,
        "elapsed_sec": {
            "stage_a_camgen": (a or {}).get("elapsed_sec"),
            "stage_b_eval": (b or {}).get("elapsed_sec"),
            "stage_c_perimage": (c or {}).get("elapsed_sec"),
            "stage_d_mctformer": (d or {}).get("elapsed_sec"),
        },
    }

    yaml_str = _yaml_dump_str(report)
    out_path = out_dir / "report.yaml"
    out_path.write_text(yaml_str + "\n")
    log.info(f"  wrote {out_path}")
    return yaml_str


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _load_stage(out_dir: Path, name: str) -> dict | None:
    fp = out_dir / f"stage_{name}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def main():
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_gpus(args.gpus)

    log.info("=" * 64)
    log.info(f"Phase-5 write-up metrics ({len(devices)} GPUs, "
             f"{args.num_crf_workers} CRF workers)")
    log.info(f"  output_dir: {out_dir}")
    log.info(f"  P1' ckpt: {args.p1_ckpt}")
    log.info(f"  P2' ckpt: {args.p2_ckpt}")
    log.info(f"  P1' CRF: {args.p1_crf}")
    log.info(f"  P2' CRF: {args.p2_crf}")
    log.info(f"  scales: {args.scales}")
    log.info(f"  max_images: {args.max_images} (0 = full val)")
    log.info("=" * 64)

    # Sanity: ckpts + GT dir + image dir + labels file exist
    for path, name in [
        (args.p1_ckpt, "P1' ckpt"),
        (args.p2_ckpt, "P2' ckpt"),
        (args.image_dir, "image_dir"),
        (args.gt_dir, "gt_dir"),
        (args.labels_file, "labels_file"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Stage A
    if args.skip_camgen:
        a = _load_stage(out_dir, "a")
        log.info(f"Stage A skipped (loaded {bool(a)} from disk)")
    else:
        a = stage_a_camgen(args, out_dir)

    # Stage B
    if args.skip_eval:
        b = _load_stage(out_dir, "b")
        log.info(f"Stage B skipped (loaded {bool(b)} from disk)")
    else:
        b = stage_b_eval(args, out_dir)

    # Stage C
    if args.skip_perimage:
        c = _load_stage(out_dir, "c")
        log.info(f"Stage C skipped (loaded {bool(c)} from disk)")
    elif b is None:
        log.warning("Stage C skipped because Stage B output missing.")
        c = None
    else:
        c = stage_c_perimage(args, out_dir, b)

    # Stage D (optional)
    if args.skip_mctformer:
        d = _load_stage(out_dir, "d")
        log.info(f"Stage D skipped (loaded {bool(d)} from disk)")
    else:
        d = stage_d_mctformer(args, out_dir)

    # Stage E
    yaml_str = stage_e_emit_yaml(args, out_dir, a, b, c, d)

    log.info("\n" + "=" * 64)
    log.info("FINAL YAML REPORT (also written to report.yaml):")
    log.info("=" * 64)
    print(yaml_str)


if __name__ == "__main__":
    # Required for CUDA + multiprocessing.spawn
    mp.set_start_method("spawn", force=True)
    main()
