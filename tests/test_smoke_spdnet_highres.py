"""Pytest wrapper for scripts/smoke_test_spdnet_highres.py.

The script is designed to be invoked manually before the big training
run on a real GPU. This test wrapper runs it in ``--allow-cpu-fallback``
mode (128^2, no VRAM cap) so the CI machine can still exercise:

  * that the script imports cleanly,
  * that all aux-loss code paths execute at a non-default resolution,
  * that the JSON diagnostics have the shape Phase 5 consumers expect.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "smoke_test_spdnet_highres.py"


def test_script_exists():
    assert SCRIPT.exists(), f"missing {SCRIPT}"


def test_cpu_fallback_smoke(tmp_path):
    """Run the smoke script on CPU at 128^2, assert success + diagnostics JSON."""
    out_json = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--image-size", "128",
            "--batch-size", "2",
            "--num-classes", "5",
            "--lambda-ac", "0.1",
            "--lambda-mask", "0.1",
            "--out", str(out_json),
            "--allow-cpu-fallback",
        ],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert out_json.exists(), "smoke.json was not written"

    info = json.loads(out_json.read_text())
    for k in [
        "image_size", "batch_size", "num_classes", "device",
        "use_aux", "loss_total", "loss_components",
        "query_merged_shape", "fused_shape", "logits_shape",
        "fwd_bwd_seconds", "peak_vram_gib", "total_params",
    ]:
        assert k in info, f"missing required field: {k}"

    assert info["image_size"] == 128
    assert info["batch_size"] == 2
    assert info["use_aux"] is True
    assert "L_cls" in info["loss_components"]
    assert "L_ac" in info["loss_components"]
    assert "L_mask" in info["loss_components"]
    assert info["logits_shape"] == [2, 5]


def test_no_aux_flag(tmp_path):
    """--no-aux disables L_ac and L_mask."""
    out_json = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--image-size", "128",
            "--batch-size", "2",
            "--num-classes", "5",
            "--out", str(out_json),
            "--allow-cpu-fallback",
            "--no-aux",
        ],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    info = json.loads(out_json.read_text())
    assert info["use_aux"] is False
    assert "L_cls" in info["loss_components"]
    assert "L_ac" not in info["loss_components"]
    assert "L_mask" not in info["loss_components"]


def test_no_cuda_errors_without_fallback(tmp_path):
    """Without --allow-cpu-fallback the script must refuse to run on CPU."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available; can't test CPU-refusal path here")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--image-size", "128",
            "--batch-size", "2",
            "--num-classes", "5",
        ],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 1
    assert "CUDA is not available" in result.stderr
