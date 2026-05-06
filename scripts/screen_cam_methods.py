"""Cheap screening of CAM-generation methods on a single SPDNet checkpoint.

Purpose
-------
Detect whether a gradient-based CAM method (LayerCAM / GradCAM++ /
XGradCAM) is materially better than the current ``cam_max`` baseline on
a WSSS localization subset, without paying the full
``eval_d4_localization.py`` cost. Only methods that beat the baseline
by a meaningful margin should be promoted to the expensive 1000-img
full-resolution pipeline.

Trade-offs vs. ``eval_d4_localization.py --fullres_sweep``:
  * **Subset size:** 250 imgs (default) instead of 1000 (full-res final).
  * **TTA:** just scale=1.0 + hflip (2 augs / image) instead of
    4 scales x 2 flips (8 augs / image).
  * **CRF sweep:** CAM-resolution (fast) with a narrow 8-config grid
    instead of the 72-config fullres grid. Uses ``sweep_crf_params``.
  * **Final eval:** skipped. The script reports the CAM-res CRF sweep's
    top-1 micro DisIoU as its decision signal. That is known to be a
    lower bound on the real full-res micro DisIoU (we have seen CAM-res
    be ~6--8 pp OVER the full-res number on this project's data due to
    nearest-neighbour upsampling smoothing, so prefer it for ranking
    rather than absolute comparison).

Output
------
Per mode: ``<out>/<mode>/screen.json`` with fields:

* ``mode``, ``target_layer``, ``n_images``, ``elapsed_s``
* ``threshold_best``, ``threshold_disease_iou``
* ``crf_top1_disease_iou``, ``crf_top1_params``, ``crf_top3``

After all modes finish, a compact ranking table is printed to stdout.

Skip-if-exists: if ``screen.json`` already exists for a mode, the script
skips that mode unless ``--force`` is passed.

Example
-------
::

    uv run python scripts/screen_cam_methods.py \\
        --checkpoint outputs/spdnet_plantseg/d4_ac_safe/checkpoints/best.ckpt \\
        --seed-modes cam_max,feat_chvar,layercam,gradcam_pp,xgradcam \\
        --out outputs/phase5/a1_screen_d4ac
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

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
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
from src.wsss.spdnet.gradient_cam_methods import is_gradient_cam_mode
from src.wsss.spdnet.online_loc_metric import select_deterministic_subset
from scripts.sweep_crf_params import sweep_crf_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("screen_cam")


IMAGE_DIR = Path("data/plantsegv3/images/val")
GT_DIR = Path("outputs/plantseg_binary_mc115/gt_binary_val")
REF_IMAGE_DIR = Path("data/plantsegv3/images/train")
LABEL_FILE = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"
CLASS_NAMES_FILE = "outputs/plantseg_binary_mc115/labels/class_names.txt"
NUM_CLASSES = 115
IMAGE_EXT = ".jpg"

# Narrow grid for screening -- 8 configs vs the 72 in eval_d4_localization.
# All values chosen from earlier CAM-res CRF sweeps on this dataset.
SCREEN_CRF_SRGB = [3, 5, 8, 13]
SCREEN_CRF_BG_THR = [0.1, 0.3]
SCREEN_CRF_SCALE = [1.0]

# Cheap TTA: scale=1.0 only; hflip is added implicitly below by mirroring
# each entry. Drops multi-scale -- the screening is about ranking, not
# absolute quality.
SCREEN_SCALES: list[float] = [1.0]

DEFAULT_SUBSET_SIZE = 250
DEFAULT_SUBSET_SEED = 1234  # matches run_seg_probes_phase1.sh


# Seed modes known to the existing ``generate_all_seeds`` / gradient
# dispatcher PLUS the classifier-CAM path. Screening supports all of
# them so the same invocation can calibrate and compare.
ALLOWED_MODES: tuple[str, ...] = (
    "cam_max",
    "feat_chmean", "feat_chvar", "feat_chmax", "feat_neg_chmean", "feat_l2norm",
    "fused_chmean", "fused_chvar", "fused_chmax", "fused_neg_chmean", "fused_l2norm",
    "spatial_proto", "attn_map", "attn_max",
    "layercam", "gradcam_pp", "xgradcam",
)


def _select_subset(gt_dir: Path, subset_size: int, seed: int) -> list[str]:
    all_names = sorted(f.stem for f in gt_dir.glob("*.png"))
    if subset_size >= len(all_names):
        return all_names
    return select_deterministic_subset(all_names, subset_size, seed=seed)


def _build_label_dict(
    names: list[str], class_resolver, num_classes: int,
) -> dict[str, np.ndarray]:
    """Multi-hot labels from the filename-class resolver."""
    out: dict[str, np.ndarray] = {}
    for n in names:
        cls = class_resolver(n)
        lbl = np.zeros(num_classes, dtype=np.float32)
        lbl[cls if cls is not None else 0] = 1.0
        out[n] = lbl
    return out


def _generate_seeds_for_mode(
    mode: str,
    model,
    label_dict: dict[str, np.ndarray],
    ref_pool: dict,
    class_resolver,
    seed_dir: Path,
    device: torch.device,
    target_layer: str,
    max_classes_per_image: int,
) -> list[str]:
    """Run the correct seed generator for the mode and return image names processed."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    if mode == "cam_max":
        return generate_all_cams(
            model=model, label_dict=label_dict, image_dir=IMAGE_DIR,
            output_dir=seed_dir, image_ext=IMAGE_EXT, scales=SCREEN_SCALES,
            input_size=448, num_ref_images=1, binary_aggregate="max",
            device=device, ref_pool=ref_pool, ref_image_dir=REF_IMAGE_DIR,
            query_class_resolver=class_resolver,
        )
    return generate_all_seeds(
        model=model, label_dict=label_dict, image_dir=IMAGE_DIR,
        output_dir=seed_dir, image_ext=IMAGE_EXT, scales=SCREEN_SCALES,
        input_size=448, num_ref_images=1, seed_mode=mode,
        device=device, ref_pool=ref_pool, ref_image_dir=REF_IMAGE_DIR,
        query_class_resolver=class_resolver,
        target_layer=target_layer,
        max_classes_per_image=max_classes_per_image,
    )


def _screen_one_mode(
    mode: str,
    model,
    label_dict: dict[str, np.ndarray],
    ref_pool: dict,
    class_resolver,
    out_dir: Path,
    device: torch.device,
    target_layer: str,
    max_classes_per_image: int,
    crf_workers: int,
    skip_seed_gen: bool,
    skip_crf: bool = False,
) -> dict:
    """Run the full screen for one seed mode: generate + threshold + CRF sweep."""
    mode_dir = out_dir / mode
    seed_dir = mode_dir / "seeds"
    screen_json = mode_dir / "screen.json"

    mode_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in seed_dir.glob("*.npy")} if seed_dir.exists() else set()
    want = set(label_dict.keys())
    missing = sorted(want - existing)

    t0 = time.time()
    if skip_seed_gen and not missing:
        log.info("[%s] %d seeds already present -- skipping gen", mode, len(existing))
    else:
        log.info(
            "[%s] generating seeds for %d imgs (scales=%s, target_layer=%s)",
            mode, len(missing) if skip_seed_gen else len(label_dict),
            SCREEN_SCALES, target_layer,
        )
        target = {n: label_dict[n] for n in (missing if skip_seed_gen else label_dict)}
        _generate_seeds_for_mode(
            mode=mode, model=model, label_dict=target, ref_pool=ref_pool,
            class_resolver=class_resolver, seed_dir=seed_dir, device=device,
            target_layer=target_layer, max_classes_per_image=max_classes_per_image,
        )

    avail = sorted(n for n in label_dict if (seed_dir / f"{n}.npy").exists())
    log.info("[%s] %d/%d seeds available for eval", mode, len(avail), len(label_dict))
    if not avail:
        raise RuntimeError(
            f"[{mode}] no seeds generated under {seed_dir} -- aborting screen"
        )

    # Threshold sweep (micro DisIoU), parallelised across the 100
    # thresholds via multiprocessing.Pool. Historically serial; drops
    # from ~2 min to ~10--15 s on the 250-img subset at crf_workers=16.
    log.info("[%s] threshold sweep (disease_iou)...", mode)
    sweep = evaluate_cam_threshold_sweep(
        predict_dir=str(seed_dir), gt_dir=str(GT_DIR),
        name_list=avail, num_cls=2, optimize_metric="disease_iou",
        num_workers=crf_workers,
    )
    best_at = sweep.get("result_at_best", {})
    fg_keys = [k for k in best_at if k not in ("mIoU", "background")]
    thr_disease_iou = float(best_at[fg_keys[0]]) if fg_keys else 0.0
    best_threshold = float(sweep["best_threshold"])
    thr_bg = float(best_at.get("background", 0))
    thr_miou = float(best_at.get("mIoU", 0))

    # CAM-res CRF sweep (narrow grid).
    if skip_crf:
        crf_top3: list[dict] = []
        top1_disease = 0.0
        top1_params: dict = {}
    else:
        log.info(
            "[%s] CRF sweep [CAM-res, %d configs] on %d imgs...",
            mode,
            len(SCREEN_CRF_SRGB) * len(SCREEN_CRF_BG_THR) * len(SCREEN_CRF_SCALE),
            len(avail),
        )
        crf_results = sweep_crf_params(
            seed_dir=seed_dir, image_dir=IMAGE_DIR, gt_dir=GT_DIR,
            image_ext=IMAGE_EXT, num_cls=2,
            srgb_values=SCREEN_CRF_SRGB,
            bg_thr_values=SCREEN_CRF_BG_THR,
            scale_values=SCREEN_CRF_SCALE,
            max_images=0,  # use all `avail`
            num_workers=crf_workers,
        )
        crf_top3 = crf_results[:3]
        top1 = crf_results[0] if crf_results else {}
        top1_disease = float(top1.get("disease_iou", 0.0))
        top1_params = {
            "srgb": float(top1.get("srgb", 0.0)),
            "bg_threshold": float(top1.get("bg_threshold", 0.0)),
            "scale_factor": float(top1.get("scale_factor", 0.0)),
        }

    elapsed = time.time() - t0

    result = {
        "mode": mode,
        "target_layer": target_layer if is_gradient_cam_mode(mode) else None,
        "n_images": len(avail),
        "n_images_requested": len(label_dict),
        "scales": SCREEN_SCALES,
        "elapsed_s": round(elapsed, 1),
        "threshold_best": best_threshold,
        "threshold_disease_iou": thr_disease_iou,
        "threshold_bg_iou": thr_bg,
        "threshold_miou": thr_miou,
        "crf_top1_disease_iou": top1_disease,
        "crf_top1_params": top1_params,
        "crf_top3": crf_top3,
    }
    screen_json.write_text(json.dumps(result, indent=2))
    log.info(
        "[%s] DONE in %.0fs  |  thr=%.2f dis=%.2f%%  CRF top-1 dis=%.2f%% "
        "(srgb=%.0f bg=%.2f sc=%.1f)",
        mode, elapsed, best_threshold, thr_disease_iou,
        top1_disease, top1_params.get("srgb", 0.0),
        top1_params.get("bg_threshold", 0.0), top1_params.get("scale_factor", 0.0),
    )
    return result


def _print_ranking(results: list[dict]) -> None:
    if not results:
        print("(no results)")
        return
    results_sorted = sorted(
        results, key=lambda r: r["crf_top1_disease_iou"], reverse=True,
    )
    print("\n" + "=" * 78)
    print(
        f"{'rank':>4} {'mode':>18} {'thr_dis':>9} {'CRF_top1_dis':>13} "
        f"{'srgb':>5} {'bg':>5} {'n':>5} {'time_s':>7}"
    )
    print("-" * 78)
    for i, r in enumerate(results_sorted):
        p = r["crf_top1_params"]
        print(
            f"{i + 1:>4} {r['mode']:>18} "
            f"{r['threshold_disease_iou']:>9.2f} "
            f"{r['crf_top1_disease_iou']:>13.2f} "
            f"{p.get('srgb', 0.0):>5.0f} {p.get('bg_threshold', 0.0):>5.2f} "
            f"{r['n_images']:>5d} {r['elapsed_s']:>7.1f}"
        )
    print("=" * 78)
    print(
        "[ranking metric: CAM-resolution CRF sweep top-1 disease IoU (MICRO %), "
        f"{len(SCREEN_CRF_SRGB) * len(SCREEN_CRF_BG_THR) * len(SCREEN_CRF_SCALE)} configs]"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument(
        "--seed-modes",
        required=True,
        help="Comma-separated list of seed modes to screen. Choose from: "
        + ", ".join(ALLOWED_MODES),
    )
    p.add_argument("--out", required=True, type=Path,
                    help="Output root directory (per-mode subdirs live here).")
    p.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    p.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    p.add_argument("--crf-workers", type=int, default=8)
    p.add_argument(
        "--target-layer", default="query_merged",
        choices=["query_merged", "fused", "layer4"],
        help="Target layer for gradient-CAM modes (ignored by others).",
    )
    p.add_argument("--max-classes-per-image", type=int, default=4)
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if <out>/<mode>/screen.json already exists.",
    )
    p.add_argument(
        "--skip-crf", action="store_true",
        help="Skip the CRF sweep (threshold-only screen). Fast.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without loading the checkpoint.",
    )
    args = p.parse_args()

    modes = [m.strip() for m in args.seed_modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in ALLOWED_MODES]
    if unknown:
        print(f"ERROR: unknown seed modes: {unknown}", file=sys.stderr)
        print(f"Allowed: {', '.join(ALLOWED_MODES)}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    subset_names = _select_subset(GT_DIR, args.subset_size, args.subset_seed)

    if args.dry_run:
        print("[dry-run] would screen:")
        print(f"  checkpoint       : {args.checkpoint}")
        print(f"  output           : {args.out}")
        print(f"  modes            : {modes}")
        print(f"  subset size/seed : {len(subset_names)} / {args.subset_seed}")
        print(f"  scales (TTA)     : {SCREEN_SCALES}")
        print(
            "  crf grid         : "
            f"srgb={SCREEN_CRF_SRGB}, bg={SCREEN_CRF_BG_THR}, sc={SCREEN_CRF_SCALE}"
        )
        print(f"  target_layer     : {args.target_layer}")
        print(f"  max_classes/img  : {args.max_classes_per_image}")
        return 0

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # Build shared ref pool + label dict.
    class_names = load_class_names(CLASS_NAMES_FILE)
    class_resolver = make_filename_class_resolver(class_names)
    ref_pool = build_class_pool_from_labels(
        LABEL_FILE, REF_IMAGE_DIR, image_ext=IMAGE_EXT,
    )
    log.info(
        "selected %d val images (seed=%d); ref pool covers %d/%d classes",
        len(subset_names), args.subset_seed, len(ref_pool), NUM_CLASSES,
    )
    label_dict = _build_label_dict(subset_names, class_resolver, NUM_CLASSES)

    log.info("loading checkpoint: %s", args.checkpoint)
    model = load_spdnet_from_checkpoint(str(args.checkpoint), NUM_CLASSES).to(device)
    model.eval()
    log.info(
        "fusion_mode=%s  num_classes=%d  device=%s",
        model.fusion_mode, NUM_CLASSES, device,
    )

    results: list[dict] = []
    for mode in modes:
        screen_json = args.out / mode / "screen.json"
        if screen_json.exists() and not args.force:
            log.info("[%s] screen.json exists -- skipping (use --force to override)", mode)
            results.append(json.loads(screen_json.read_text()))
            continue
        try:
            r = _screen_one_mode(
                mode=mode, model=model, label_dict=label_dict,
                ref_pool=ref_pool, class_resolver=class_resolver,
                out_dir=args.out, device=device,
                target_layer=args.target_layer,
                max_classes_per_image=args.max_classes_per_image,
                crf_workers=args.crf_workers,
                skip_seed_gen=True,
                skip_crf=args.skip_crf,
            )
            results.append(r)
        except Exception as e:
            log.exception("[%s] screen failed: %s", mode, e)

    _print_ranking(results)

    # Aggregate JSON for downstream consumers.
    agg_path = args.out / "screen_all.json"
    agg_path.write_text(json.dumps({"results": results}, indent=2))
    log.info("wrote aggregate %s", agg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
