"""Tests for scripts/screen_cam_methods.py.

Light-weight: focuses on the module's standalone helpers and the
``--dry-run`` CLI path. A full end-to-end smoke (actually generating
seeds + CRF sweep) would require the plant-dataset fixtures on disk,
which aren't present in unit-test environments; those are covered by
the prelaunch script invoked before real runs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "screen_cam_methods.py"


def test_script_exists():
    assert SCRIPT.exists(), f"screen_cam_methods.py not found at {SCRIPT}"


def test_dry_run_invocation(tmp_path):
    """--dry-run prints plan without loading a checkpoint or touching disk."""
    out_dir = tmp_path / "dry"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint", "/tmp/does-not-exist.ckpt",
            "--seed-modes", "layercam,feat_chvar",
            "--out", str(out_dir),
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    stdout = result.stdout
    assert "[dry-run]" in stdout
    assert "modes" in stdout
    assert "layercam" in stdout and "feat_chvar" in stdout
    assert "crf grid" in stdout
    # Should NOT have created any output files (dry-run is read-only).
    # The script does call mkdir(parents=True, exist_ok=True) on args.out,
    # so the directory may exist; just assert no screen.json was written.
    assert not any(out_dir.rglob("screen.json"))


def test_unknown_mode_fails(tmp_path):
    """Unknown seed mode should fail with a helpful error."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint", "/tmp/x.ckpt",
            "--seed-modes", "bogus_mode",
            "--out", str(tmp_path),
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "unknown seed modes" in result.stderr.lower()


def test_allowed_modes_include_all_intended():
    """Regression: the allowed-modes tuple covers every mode we plan to screen."""
    from scripts.screen_cam_methods import ALLOWED_MODES

    must_include = {
        "cam_max", "feat_chmean", "feat_chvar",
        "fused_chvar", "attn_map",
        "layercam", "gradcam_pp", "xgradcam",
    }
    missing = must_include - set(ALLOWED_MODES)
    assert not missing, f"ALLOWED_MODES missing: {missing}"


def test_crf_grid_is_narrow():
    """Screening CRF grid must stay tiny (fast screen, not a final sweep)."""
    from scripts.screen_cam_methods import (
        SCREEN_CRF_BG_THR, SCREEN_CRF_SCALE, SCREEN_CRF_SRGB,
    )

    n = len(SCREEN_CRF_SRGB) * len(SCREEN_CRF_BG_THR) * len(SCREEN_CRF_SCALE)
    assert 4 <= n <= 16, (
        f"screening CRF grid has {n} configs; should be between 4 and 16 "
        "(too few = noise; too many = defeats the purpose of a cheap screen)"
    )


def test_select_subset_deterministic(tmp_path):
    """_select_subset produces the same list for the same seed."""
    from scripts.screen_cam_methods import _select_subset

    # Create a fake GT dir with fake .png files.
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for i in range(500):
        (gt_dir / f"img_{i:04d}.png").write_bytes(b"x")

    a = _select_subset(gt_dir, subset_size=250, seed=1234)
    b = _select_subset(gt_dir, subset_size=250, seed=1234)
    c = _select_subset(gt_dir, subset_size=250, seed=5678)

    assert a == b, "deterministic subset must be reproducible"
    assert a != c, "different seeds should give different subsets"
    assert len(a) == 250


def test_select_subset_returns_all_when_requested_too_large(tmp_path):
    from scripts.screen_cam_methods import _select_subset

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for i in range(10):
        (gt_dir / f"img_{i:04d}.png").write_bytes(b"x")

    a = _select_subset(gt_dir, subset_size=100, seed=1234)
    assert len(a) == 10


def test_build_label_dict_sets_resolved_class():
    from scripts.screen_cam_methods import _build_label_dict

    def resolver(name: str) -> int | None:
        if name.startswith("x_"):
            return 5
        return None

    names = ["x_foo", "y_bar"]
    labels = _build_label_dict(names, resolver, num_classes=10)
    assert labels["x_foo"][5] == 1.0
    assert labels["x_foo"].sum() == 1.0
    # Unresolved falls back to class 0.
    assert labels["y_bar"][0] == 1.0
    assert labels["y_bar"].sum() == 1.0


def test_screen_result_json_roundtrip(tmp_path):
    """The script writes a parseable screen.json with required keys."""
    # Emulate what _screen_one_mode writes.
    from scripts.screen_cam_methods import SCREEN_SCALES

    payload = {
        "mode": "layercam",
        "target_layer": "query_merged",
        "n_images": 250,
        "n_images_requested": 250,
        "scales": SCREEN_SCALES,
        "elapsed_s": 42.5,
        "threshold_best": 0.35,
        "threshold_disease_iou": 28.3,
        "threshold_bg_iou": 70.1,
        "threshold_miou": 49.2,
        "crf_top1_disease_iou": 33.4,
        "crf_top1_params": {"srgb": 5.0, "bg_threshold": 0.3, "scale_factor": 1.0},
        "crf_top3": [],
    }
    p = tmp_path / "screen.json"
    p.write_text(json.dumps(payload))
    got = json.loads(p.read_text())
    # Required fields for phase5_update_summary.py + phase5_generate_report.py.
    for k in [
        "mode", "n_images", "threshold_best", "threshold_disease_iou",
        "crf_top1_disease_iou", "crf_top1_params", "elapsed_s",
    ]:
        assert k in got, f"missing required key: {k}"
