#!/usr/bin/env python
"""Decision-gate helpers for the SPDNet seg-probe pipeline.

Two subcommands:

    phase1  Read every per-position eval.json under
            outputs/spdnet_plantseg/seg_probe_phase1/<ckpt>/<pos>/eval.json,
            compute the composite score S = max(probe, chmean, chvar, cam_cls),
            pick top-3 per ckpt, force-include any S >= 30, and (for spatial)
            at least one fused position. Write SUMMARY.md + selected.json.

    phase2  Read every per-(ckpt, position, lambda) eval.json under
            outputs/spdnet_plantseg/seg_probe_phase2/<ckpt>/<pos>/seg<L>_cls<C>/eval.json,
            pick the single best by probe_iou, write SUMMARY.md + chosen.json
            for Phase 3 to consume.

Also exposes a simple `--print-table` helper used by the overnight master
script for final reporting.

Run:
    python scripts/seg_probe_decisions.py phase1 \\
        --root outputs/spdnet_plantseg/seg_probe_phase1
    python scripts/seg_probe_decisions.py phase2 \\
        --root outputs/spdnet_plantseg/seg_probe_phase2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_ROOT_PHASE1 = Path("outputs/spdnet_plantseg/seg_probe_phase1")
DEFAULT_ROOT_PHASE2 = Path("outputs/spdnet_plantseg/seg_probe_phase2")
TOP_N = 3
FORCE_INCLUDE_THRESHOLD = 30.0  # S >= this in percent => keep regardless of rank
FUSED_POSITIONS = ("P4_fused", "P5_cam_classifier")
SPATIAL_CKPT_HINT = "spatial"  # substring match used to flag spatial ckpts


@dataclass
class P1Row:
    ckpt: str
    position: str
    probe_iou: float
    chmean_iou: float | None
    chvar_iou: float | None
    cam_cls_iou: float | None
    score_S: float
    probe_underperforms: bool
    fusion_mode: str
    # 0 = absent (eval.json predates --limit-val) OR full val.
    # >0 = deterministic subset of that size used for ranking.
    limit_val: int = 0
    # Total val images used in the eval (after limit_val applied).
    # Defaults to 0 when absent in the eval.json.
    n_val_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class P2Row:
    ckpt: str
    position: str
    seg_loss_weight: float
    cls_loss_weight: float
    probe_iou: float
    chmean_iou: float | None
    chvar_iou: float | None
    cam_cls_iou: float | None
    score_S: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_phase1_rows(root: Path) -> list[P1Row]:
    rows: list[P1Row] = []
    for ckpt_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pos_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            ej = pos_dir / "eval.json"
            if not ej.exists():
                continue
            with open(ej) as f:
                blob = json.load(f)
            row = P1Row(
                ckpt=ckpt_dir.name,
                position=pos_dir.name,
                probe_iou=_safe_get(blob, "probe_iou", 0.0) or 0.0,
                chmean_iou=_safe_get(blob, "chmean_iou"),
                chvar_iou=_safe_get(blob, "chvar_iou"),
                cam_cls_iou=_safe_get(blob, "cam_cls_iou"),
                score_S=_safe_get(blob, "score_S", 0.0) or 0.0,
                probe_underperforms=bool(blob.get("probe_underperforms", False)),
                fusion_mode=str(blob.get("fusion_mode", "")),
                limit_val=int(blob.get("limit_val", 0) or 0),
                n_val_used=int(blob.get("n_val_used", 0) or 0),
            )
            rows.append(row)
    return rows


def _select_phase2(rows: list[P1Row]) -> dict[str, list[str]]:
    """Apply the force-include rule per ckpt.

    Returns {ckpt: [positions...]}.
    """
    out: dict[str, list[str]] = {}
    by_ckpt: dict[str, list[P1Row]] = {}
    for r in rows:
        by_ckpt.setdefault(r.ckpt, []).append(r)

    for ckpt, ckpt_rows in by_ckpt.items():
        ckpt_rows = sorted(ckpt_rows, key=lambda x: -x.score_S)
        selected: list[str] = []

        for r in ckpt_rows[:TOP_N]:
            selected.append(r.position)

        for r in ckpt_rows:
            if r.score_S >= FORCE_INCLUDE_THRESHOLD and r.position not in selected:
                selected.append(r.position)

        is_spatial = ckpt_rows and ckpt_rows[0].fusion_mode == "spatial"
        if not is_spatial:
            is_spatial = SPATIAL_CKPT_HINT in ckpt
        if is_spatial:
            fused_present = [r for r in ckpt_rows if r.position in FUSED_POSITIONS]
            if fused_present and not any(p in selected for p in FUSED_POSITIONS):
                fused_best = max(fused_present, key=lambda x: x.score_S)
                selected.append(fused_best.position)

        seen: set[str] = set()
        ordered: list[str] = []
        for p in selected:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        out[ckpt] = ordered
    return out


def _format_iou(v: float | None) -> str:
    if v is None:
        return "  -  "
    return f"{v:5.2f}"


def _phase1_summary_md(rows: list[P1Row], selected: dict[str, list[str]]) -> str:
    lines = []
    lines.append("# Phase 1 — Frozen Probe Summary")
    lines.append("")
    lines.append(f"Selected positions for Phase 2 (top-{TOP_N} by composite score "
                 f"S = max(probe, chmean, chvar, cam_cls); force-include S >= "
                 f"{FORCE_INCLUDE_THRESHOLD}%; spatial ckpt force-includes 1 fused).")
    lines.append("")
    by_ckpt: dict[str, list[P1Row]] = {}
    for r in rows:
        by_ckpt.setdefault(r.ckpt, []).append(r)

    for ckpt, ckpt_rows in by_ckpt.items():
        ckpt_rows = sorted(ckpt_rows, key=lambda x: -x.score_S)
        lines.append(f"## {ckpt}  (fusion={ckpt_rows[0].fusion_mode if ckpt_rows else '?'})")
        lines.append("")
        lines.append("| Position | n_val | Probe IoU | chmean | chvar | cam_cls | **Score S** | probe_under | selected |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
        sel = set(selected.get(ckpt, []))
        for r in ckpt_rows:
            star = "[YES]" if r.position in sel else " "
            under = "[!]" if r.probe_underperforms else " "
            # Show n_val and append a "*" when this row was screened on a
            # subset (limit_val > 0) -- helps the reader spot apples vs
            # oranges when a few legacy rows from a prior full-val run
            # are mixed in with a screening sweep.
            if r.n_val_used > 0:
                nval_str = f"{r.n_val_used}{'*' if r.limit_val > 0 else ''}"
            else:
                nval_str = "full?"  # absent in eval.json => predates flag
            lines.append(
                f"| {r.position} | {nval_str} | {_format_iou(r.probe_iou)} | "
                f"{_format_iou(r.chmean_iou)} | {_format_iou(r.chvar_iou)} | "
                f"{_format_iou(r.cam_cls_iou)} | **{r.score_S:5.2f}** | "
                f"{under} | {star} |"
            )
        lines.append("")
        lines.append(f"**Picked for Phase 2:** {', '.join(selected.get(ckpt, [])) or '<none>'}")
        lines.append("")
    lines.append("Legend: `n_val` = number of val images used in the eval. "
                 "Asterisk (`*`) marks subset-mode rows (`--limit-val`). "
                 "`full?` means the eval.json predates the flag (treat as full val).")
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_phase1(args) -> int:
    root = Path(args.root)
    rows = _load_phase1_rows(root)
    if not rows:
        print(f"[phase1] no eval.json found under {root}")
        return 1

    selected = _select_phase2(rows)
    summary_md = _phase1_summary_md(rows, selected)
    (root / "SUMMARY.md").write_text(summary_md)
    (root / "selected.json").write_text(
        json.dumps(
            {
                "selected_per_ckpt": selected,
                "rows": [r.to_dict() for r in rows],
            },
            indent=2,
        )
    )
    print(summary_md)
    print(f"[phase1] wrote {root}/SUMMARY.md and selected.json")
    return 0


def _load_phase2_rows(root: Path) -> list[P2Row]:
    """Phase 2 layout: <root>/<ckpt>/<pos>/seg<L>_cls<C>/eval.json"""
    rows: list[P2Row] = []
    for ckpt_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pos_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            for lam_dir in sorted(p for p in pos_dir.iterdir() if p.is_dir()):
                ej = lam_dir / "eval.json"
                if not ej.exists():
                    continue
                if not lam_dir.name.startswith("seg"):
                    continue
                try:
                    seg = float(lam_dir.name.split("seg")[1].split("_")[0])
                    cls = float(lam_dir.name.split("cls")[1])
                except (IndexError, ValueError):
                    continue
                with open(ej) as f:
                    blob = json.load(f)
                rows.append(P2Row(
                    ckpt=ckpt_dir.name,
                    position=pos_dir.name,
                    seg_loss_weight=seg,
                    cls_loss_weight=cls,
                    probe_iou=_safe_get(blob, "probe_iou", 0.0) or 0.0,
                    chmean_iou=_safe_get(blob, "chmean_iou"),
                    chvar_iou=_safe_get(blob, "chvar_iou"),
                    cam_cls_iou=_safe_get(blob, "cam_cls_iou"),
                    score_S=_safe_get(blob, "score_S", 0.0) or 0.0,
                ))
    return rows


def _phase2_summary_md(rows: list[P2Row], best: P2Row | None) -> str:
    lines = []
    lines.append("# Phase 2 — Targeted Unfrozen Fine-Tune Summary")
    lines.append("")
    if best is None:
        lines.append("No phase-2 results found.")
        return "\n".join(lines) + "\n"
    lines.append(f"**Best (ckpt, position, lambda): "
                 f"{best.ckpt} / {best.position} / seg={best.seg_loss_weight} cls={best.cls_loss_weight}** "
                 f"-- probe IoU = {best.probe_iou:.2f}%, S = {best.score_S:.2f}%")
    lines.append("")
    by_ckpt_pos: dict[tuple[str, str], list[P2Row]] = {}
    for r in rows:
        by_ckpt_pos.setdefault((r.ckpt, r.position), []).append(r)
    for (ckpt, pos), trios in sorted(by_ckpt_pos.items()):
        lines.append(f"## {ckpt} / {pos}")
        lines.append("")
        lines.append("| seg | cls | Probe IoU | chmean | chvar | cam_cls | S |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(trios, key=lambda x: x.seg_loss_weight):
            lines.append(
                f"| {r.seg_loss_weight:.2f} | {r.cls_loss_weight:.2f} | "
                f"{_format_iou(r.probe_iou)} | {_format_iou(r.chmean_iou)} | "
                f"{_format_iou(r.chvar_iou)} | {_format_iou(r.cam_cls_iou)} | "
                f"{_format_iou(r.score_S)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_phase2(args) -> int:
    root = Path(args.root)
    rows = _load_phase2_rows(root)
    if not rows:
        print(f"[phase2] no eval.json found under {root}")
        return 1
    best = max(rows, key=lambda r: r.probe_iou)
    summary_md = _phase2_summary_md(rows, best)
    (root / "SUMMARY.md").write_text(summary_md)
    (root / "chosen.json").write_text(json.dumps(best.to_dict(), indent=2))
    print(summary_md)
    print(f"[phase2] wrote {root}/SUMMARY.md and chosen.json")
    return 0


def cmd_table(args) -> int:
    """Print a single combined table of P1+P2+P3 results from the master log."""
    root = Path(args.root)
    if not root.exists():
        print(f"[table] no root: {root}")
        return 1
    print(f"\n=== Final summary ({root}) ===\n")
    for phase_root_name in ("seg_probe_phase1", "seg_probe_phase2", "seg_probe_phase3"):
        phase_root = root / phase_root_name
        if not phase_root.exists():
            continue
        sm = phase_root / "SUMMARY.md"
        if sm.exists():
            print(f"\n--- {phase_root_name}/SUMMARY.md ---\n")
            print(sm.read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("phase1")
    sp1.add_argument("--root", type=str, default=str(DEFAULT_ROOT_PHASE1))
    sp1.set_defaults(func=cmd_phase1)

    sp2 = sub.add_parser("phase2")
    sp2.add_argument("--root", type=str, default=str(DEFAULT_ROOT_PHASE2))
    sp2.set_defaults(func=cmd_phase2)

    spt = sub.add_parser("table")
    spt.add_argument("--root", type=str, default="outputs/spdnet_plantseg")
    spt.set_defaults(func=cmd_table)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
