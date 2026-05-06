"""Idempotent Phase 5 summary merger.

Scans ``outputs/phase5/`` for the well-known artefact layout produced by
the Phase 5 stages (screening, full eval, probes, smoke) and writes a
single ``outputs/phase5/summary.json`` consolidating the numbers. Safe
to re-run at any point -- it rebuilds ``summary.json`` from scratch on
each invocation, picking up whichever stages have completed so far.

Expected on-disk layout
-----------------------

::

    outputs/phase5/
      c_smoke/
        smoke.json                             # from smoke_test_spdnet_highres.py
      a1_screen_d4ac/
        screen_all.json                        # aggregate from screen_cam_methods.py
        <mode>/screen.json                     # per-mode
      a2_full_d4ac/
        all_summaries.json                     # from eval_d4_localization.py
        <run>/summary.json
      b_probe_d4ac/
        <tag>/<position>/eval.json             # from eval_seg_probes.py
      d1_screen_highres/
        screen_all.json
        <mode>/screen.json
      d2_full_highres/
        all_summaries.json
        <run>/summary.json
      d3_probe_highres/
        <tag>/<position>/eval.json

Output
------

``outputs/phase5/summary.json`` with four top-level stage keys:
``{"c_smoke": {...}, "cam_methods": {...}, "probes": {...}, "headline": {...}}``.
The ``cam_methods`` and ``probes`` entries are further keyed by
checkpoint tag (``d4_ac_safe`` / ``highres_896``) so the report
generator can produce a single table with both rows.

Example
-------
::

    uv run python scripts/phase5_update_summary.py
    uv run python scripts/phase5_update_summary.py --stage probes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("outputs/phase5")

# Canonical checkpoint-tag mapping used across all stages. Kept in one
# place so changes to the launch-guide directory names propagate here.
CHECKPOINT_TAGS = ("d4_ac_safe", "highres_896")

# Which subdirectory feeds which phase of the summary. One tuple per
# stage type so ``--stage`` can filter down to the parts that actually
# changed.
SCREEN_DIRS = {
    "d4_ac_safe": "a1_screen_d4ac",
    "highres_896": "d1_screen_highres",
}
FULL_EVAL_DIRS = {
    "d4_ac_safe": "a2_full_d4ac",
    "highres_896": "d2_full_highres",
}
PROBE_DIRS = {
    "d4_ac_safe": "b_probe_d4ac",
    "highres_896": "d3_probe_highres",
}
SMOKE_DIR = "c_smoke"

VALID_STAGES = ("cam_methods", "probes", "smoke", "headline", "all")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"[warn] malformed JSON at {path}: {e}", file=sys.stderr)
        return None


def _collect_screen(root: Path, tag: str) -> dict[str, Any]:
    """Collect per-mode screening JSONs under the tag's screen directory."""
    subdir = root / SCREEN_DIRS[tag]
    if not subdir.exists():
        return {}
    modes: dict[str, dict] = {}
    for mode_dir in sorted(d for d in subdir.iterdir() if d.is_dir()):
        sj = mode_dir / "screen.json"
        data = _read_json_if_exists(sj)
        if data is not None:
            modes[mode_dir.name] = {
                "mode": data.get("mode", mode_dir.name),
                "n_images": data.get("n_images"),
                "threshold_disease_iou": data.get("threshold_disease_iou"),
                "crf_top1_disease_iou": data.get("crf_top1_disease_iou"),
                "crf_top1_params": data.get("crf_top1_params", {}),
                "elapsed_s": data.get("elapsed_s"),
                "target_layer": data.get("target_layer"),
            }
    return modes


def _collect_full_eval(root: Path, tag: str) -> dict[str, Any]:
    """Collect per-run summary JSONs written by eval_d4_localization."""
    subdir = root / FULL_EVAL_DIRS[tag]
    if not subdir.exists():
        return {}
    runs: dict[str, dict] = {}
    for run_dir in sorted(d for d in subdir.iterdir() if d.is_dir()):
        sj = run_dir / "summary.json"
        data = _read_json_if_exists(sj)
        if data is None:
            continue
        per_mode = data.get("per_mode", {})
        best_mode = data.get("best_seed_mode")
        best_full = per_mode.get(best_mode, {}).get("crf_best_full", {})
        runs[run_dir.name] = {
            "name": data.get("name", run_dir.name),
            "label": data.get("label"),
            "checkpoint": data.get("checkpoint"),
            "n_images_subset": data.get("n_images_subset"),
            "best_seed_mode": best_mode,
            "disease_iou_micro": best_full.get("disease_iou_micro"),
            "bg_iou_micro": best_full.get("bg_iou_micro"),
            "mIoU_micro": best_full.get("mIoU_micro"),
            "disease_iou_macro": best_full.get("disease_iou_macro"),
            "median_per_img_disease_iou": best_full.get("median_per_img_disease_iou"),
            "frac_imgs_disease_iou_ge_0.3": best_full.get("frac_imgs_disease_iou_ge_0.3"),
            "frac_imgs_disease_iou_ge_0.5": best_full.get("frac_imgs_disease_iou_ge_0.5"),
            "crf_params": {
                "srgb": best_full.get("srgb"),
                "bg_threshold": best_full.get("bg_threshold"),
                "scale_factor": best_full.get("scale_factor"),
            },
        }
    return runs


def _collect_probes(root: Path, tag: str) -> dict[str, Any]:
    """Collect per-position eval.json files from eval_seg_probes.py."""
    subdir = root / PROBE_DIRS[tag]
    if not subdir.exists():
        return {}
    out: dict[str, dict] = {}
    # Accept two layouts:
    #   <tag>/<position>/eval.json         (run_seg_probes_phase1.sh output)
    #   <position>/eval.json               (flat layout from manual runs)
    for sub in sorted(subdir.rglob("eval.json")):
        data = _read_json_if_exists(sub)
        if data is None:
            continue
        position = data.get("position") or sub.parent.name
        out[position] = {
            "position": position,
            "probe_iou": data.get("probe_iou"),
            "chmean_iou": data.get("chmean_iou"),
            "chvar_iou": data.get("chvar_iou"),
            "cam_cls_iou": data.get("cam_cls_iou"),
            "score_S": data.get("score_S"),
            "limit_val": data.get("limit_val"),
            "n_val_used": data.get("n_val_used"),
            "checkpoint": data.get("checkpoint"),
        }
    return out


def _collect_smoke(root: Path) -> dict[str, Any]:
    sj = root / SMOKE_DIR / "smoke.json"
    data = _read_json_if_exists(sj)
    if data is None:
        return {}
    return {
        "image_size": data.get("image_size"),
        "batch_size": data.get("batch_size"),
        "peak_vram_gib": data.get("peak_vram_gib"),
        "fwd_bwd_seconds": data.get("fwd_bwd_seconds"),
        "loss_components": data.get("loss_components"),
        "use_aux": data.get("use_aux"),
    }


def _compute_headline(
    cam_methods: dict[str, dict],
    probes: dict[str, dict],
) -> dict[str, Any]:
    """Single-line summary: best DisIoU across all stages."""
    def _best_micro_for(tag: str) -> dict:
        tag_runs = cam_methods.get(tag, {}).get("full_eval", {})
        if not tag_runs:
            return {}
        best_name, best_row = max(
            tag_runs.items(),
            key=lambda kv: (kv[1].get("disease_iou_micro") or -1.0),
        )
        return {
            "run": best_name,
            "seed_mode": best_row.get("best_seed_mode"),
            "disease_iou_micro": best_row.get("disease_iou_micro"),
            "mIoU_micro": best_row.get("mIoU_micro"),
            "crf_params": best_row.get("crf_params"),
        }

    def _best_probe_for(tag: str) -> dict:
        pos_map = probes.get(tag, {})
        if not pos_map:
            return {}
        best_pos, best_row = max(
            pos_map.items(),
            key=lambda kv: (kv[1].get("probe_iou") or -1.0),
        )
        return {
            "position": best_pos,
            "probe_iou": best_row.get("probe_iou"),
        }

    return {
        tag: {
            "best_cam_method": _best_micro_for(tag),
            "best_probe": _best_probe_for(tag),
        }
        for tag in CHECKPOINT_TAGS
    }


def build_summary(root: Path, stages: set[str]) -> dict[str, Any]:
    """Build the merged summary dict for the requested stages.

    When a stage is not requested, its prior value in ``summary.json``
    is preserved (incremental updates).
    """
    prior: dict[str, Any] = _read_json_if_exists(root / "summary.json") or {}
    summary: dict[str, Any] = {}
    summary["root"] = str(root)

    # smoke
    if "smoke" in stages or "all" in stages:
        summary["c_smoke"] = _collect_smoke(root)
    else:
        summary["c_smoke"] = prior.get("c_smoke", {})

    # cam_methods
    if "cam_methods" in stages or "all" in stages:
        cam_out: dict[str, dict] = {}
        for tag in CHECKPOINT_TAGS:
            cam_out[tag] = {
                "screen": _collect_screen(root, tag),
                "full_eval": _collect_full_eval(root, tag),
            }
        summary["cam_methods"] = cam_out
    else:
        summary["cam_methods"] = prior.get("cam_methods", {})

    # probes
    if "probes" in stages or "all" in stages:
        probe_out: dict[str, dict] = {}
        for tag in CHECKPOINT_TAGS:
            probe_out[tag] = _collect_probes(root, tag)
        summary["probes"] = probe_out
    else:
        summary["probes"] = prior.get("probes", {})

    # headline (always recomputed from the above)
    summary["headline"] = _compute_headline(
        summary.get("cam_methods", {}),
        summary.get("probes", {}),
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Phase 5 outputs root (default: {DEFAULT_ROOT}).")
    p.add_argument(
        "--stage", choices=VALID_STAGES, default="all",
        help="Filter which stage(s) to re-scan (others preserved from prior summary.json).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Explicit output path. Defaults to <root>/summary.json.",
    )
    args = p.parse_args()

    root = args.root
    out = args.out or (root / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    stages = {args.stage}
    summary = build_summary(root=root, stages=stages)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[phase5] wrote {out} (stage filter: {args.stage})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
