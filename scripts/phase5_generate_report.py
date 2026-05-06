"""Render ``reports/notes/phase5_highres_results.md`` from ``summary.json``.

Reads the JSON written by ``scripts/phase5_update_summary.py`` and emits
three markdown tables + a short interpretation stub so the operator has
a single place to review Phase 5 outcomes at a glance.

Tables
------
  1. CAM methods ranking -- screening top-1 micro DisIoU per (checkpoint,
     mode) + full-eval micro DisIoU when available.
  2. Probe IoU vs existing baselines -- per (checkpoint, position).
  3. Headline -- best DisIoU per checkpoint across the 4 CAM methods.

Usage
-----
::

    uv run python scripts/phase5_generate_report.py
    uv run python scripts/phase5_generate_report.py \\
        --summary outputs/phase5/summary.json \\
        --out reports/notes/phase5_highres_results.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_SUMMARY = Path("outputs/phase5/summary.json")
DEFAULT_OUT = Path("reports/notes/phase5_highres_results.md")

# Baseline row copied from reports/notes/d4_ablation_localization_v2.md
# (token feat_chmean + CRF(srgb=5, bg=0.30) -> 42.13% DisIoU_micro).
BASELINE_DISEASE_IOU_MICRO = 42.13
BASELINE_MIOU_MICRO = 60.87
BASELINE_LABEL = "token feat_chmean + CRF(srgb=5, bg=0.30) [historical]"


def _fmt_float(v, fmt: str = ".2f") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):{fmt}}"
    except (TypeError, ValueError):
        return str(v)


def _section_header(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}\n"


def _render_screen_and_full_table(cam_methods: dict[str, Any]) -> list[str]:
    lines = [
        _section_header("1. CAM methods ranking (Phase A + D)"),
        "",
        "Columns: DisIoU_thr = threshold-sweep best disease IoU (no CRF). "
        "DisIoU_crf_top1 = best of the CAM-resolution CRF sweep (ranking "
        "signal, fast). DisIoU_micro_full = full-resolution CRF final eval "
        "(strict, only present for promoted methods).",
        "",
        "| checkpoint | mode | target | DisIoU_thr (screen) | DisIoU_crf_top1 (screen) | CRF (srgb/bg/sc) | DisIoU_micro_full | mIoU_micro_full | n_screen |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for tag, blob in cam_methods.items():
        screen = blob.get("screen", {}) or {}
        full = blob.get("full_eval", {}) or {}
        # Build union set: mode name from screen PLUS best_seed_mode from full.
        modes = set(screen.keys())
        for run_row in full.values():
            bsm = run_row.get("best_seed_mode")
            if bsm:
                modes.add(bsm)
        modes_sorted = sorted(modes)

        for mode in modes_sorted:
            s = screen.get(mode, {}) or {}
            # Find matching full-eval row (where best_seed_mode==mode).
            full_row: dict = {}
            for row in full.values():
                if row.get("best_seed_mode") == mode:
                    full_row = row
                    break

            s_params = s.get("crf_top1_params", {}) or {}
            s_crf = (
                f"s={_fmt_float(s_params.get('srgb'), '.0f')} "
                f"bg={_fmt_float(s_params.get('bg_threshold'), '.2f')} "
                f"sc={_fmt_float(s_params.get('scale_factor'), '.1f')}"
            ) if s_params else "-"
            lines.append(
                f"| {tag} | {mode} | {s.get('target_layer') or '-'} | "
                f"{_fmt_float(s.get('threshold_disease_iou'))} | "
                f"{_fmt_float(s.get('crf_top1_disease_iou'))} | "
                f"{s_crf} | "
                f"{_fmt_float(full_row.get('disease_iou_micro'))} | "
                f"{_fmt_float(full_row.get('mIoU_micro'))} | "
                f"{s.get('n_images') or '-'} |"
            )
    lines.append("")
    lines.append(
        f"**Baseline reference:** {BASELINE_LABEL} = "
        f"**{BASELINE_DISEASE_IOU_MICRO:.2f}%** DisIoU_micro / "
        f"{BASELINE_MIOU_MICRO:.2f}% mIoU_micro."
    )
    lines.append("")
    return lines


def _render_probe_table(probes: dict[str, Any]) -> list[str]:
    lines = [
        _section_header("2. Probe IoU (Phase B + D-probe)"),
        "",
        "Shallow segmentation probe (trained) and three non-trainable "
        "baselines at the same position. ``probe_iou`` is the full-val "
        "disease IoU (%); larger = more localization signal in the "
        "frozen SPDNet features at that position.",
        "",
        "| checkpoint | position | probe_iou | chmean_iou | chvar_iou | cam_cls_iou | score_S | n_val |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, pos_map in probes.items():
        if not pos_map:
            continue
        for pos in sorted(pos_map.keys()):
            row = pos_map[pos]
            lines.append(
                f"| {tag} | {pos} | "
                f"{_fmt_float(row.get('probe_iou'))} | "
                f"{_fmt_float(row.get('chmean_iou'))} | "
                f"{_fmt_float(row.get('chvar_iou'))} | "
                f"{_fmt_float(row.get('cam_cls_iou'))} | "
                f"{_fmt_float(row.get('score_S'))} | "
                f"{row.get('n_val_used') or '-'} |"
            )
    lines.append("")
    return lines


def _render_headline_table(headline: dict[str, Any]) -> list[str]:
    lines = [
        _section_header("3. Headline DisIoU per checkpoint"),
        "",
        "Single best number per checkpoint (DisIoU_micro of the "
        "cross-mode winner from the full-res eval), plus the best "
        "probe IoU for context.",
        "",
        "| checkpoint | best_mode | DisIoU_micro | mIoU_micro | best_probe_pos | probe_iou |",
        "|---|---|---:|---:|---|---:|",
    ]
    for tag, row in headline.items():
        bcam = row.get("best_cam_method", {}) or {}
        bprobe = row.get("best_probe", {}) or {}
        lines.append(
            f"| {tag} | {bcam.get('seed_mode') or '-'} | "
            f"{_fmt_float(bcam.get('disease_iou_micro'))} | "
            f"{_fmt_float(bcam.get('mIoU_micro'))} | "
            f"{bprobe.get('position') or '-'} | "
            f"{_fmt_float(bprobe.get('probe_iou'))} |"
        )
    lines.append("")
    return lines


def _render_smoke_block(smoke: dict[str, Any]) -> list[str]:
    if not smoke:
        return []
    return [
        _section_header("0. 896^2 smoke preflight"),
        "",
        f"- image_size: {smoke.get('image_size')}",
        f"- batch_size: {smoke.get('batch_size')}",
        f"- peak VRAM: {_fmt_float(smoke.get('peak_vram_gib'))} GiB",
        f"- fwd+bwd: {_fmt_float(smoke.get('fwd_bwd_seconds'))} s",
        f"- loss components: {json.dumps(smoke.get('loss_components') or {}, sort_keys=True)}",
        "",
    ]


def render(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 5 -- CAM-method screening, highres training, seg-probe sweep\n",
        "",
        (
            "Generated from `outputs/phase5/summary.json` by "
            "`scripts/phase5_generate_report.py`. Rebuild after every stage "
            "by running `phase5_update_summary.py` then this script."
        ),
        "",
    ]
    lines.extend(_render_smoke_block(summary.get("c_smoke", {})))
    lines.extend(_render_screen_and_full_table(summary.get("cam_methods", {})))
    lines.extend(_render_probe_table(summary.get("probes", {})))
    lines.extend(_render_headline_table(summary.get("headline", {})))
    lines.append("")
    lines.append(
        "_Interpretation goes here manually after each stage completes; "
        "keep it short. See `reports/notes/phase5_launch_guide.md` for the "
        "per-stage decision points._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.summary.exists():
        print(f"ERROR: summary file not found: {args.summary}", file=sys.stderr)
        print(
            "Run scripts/phase5_update_summary.py first to generate it.",
            file=sys.stderr,
        )
        return 2

    summary = json.loads(args.summary.read_text())
    text = render(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[phase5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
