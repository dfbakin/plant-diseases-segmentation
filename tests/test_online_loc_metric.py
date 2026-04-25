"""Unit tests for ``src.wsss.spdnet.online_loc_metric``.

Each test name is bound to an invariant ID from the SPDNet aux-losses
spec (`spdnet_auxiliary_spatial_losses_*.plan.md` Phase C / RESEARCH_CONTEXT.md
§5.11). Two test classes:

* ``TestSweepFunctions``  -> pure-function correctness on synthetic CAMs.
* ``TestOnlineCAMIoU``    -> end-to-end on a temp-dir mini PlantSeg layout.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.wsss.spdnet.model import SPDNet
from src.wsss.spdnet.online_loc_metric import (
    DEFAULT_THRESHOLDS,
    OnlineCAMIoU,
    compute_iou_sweep,
    first_per_class_references,
    select_deterministic_subset,
    summarize_iou_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gt_circle(H: int, W: int, cx: float, cy: float, r: float) -> torch.Tensor:
    """Return a binary {0,1} mask of a disk centred at ``(cy, cx)``."""
    yy, xx = torch.meshgrid(
        torch.arange(H).float(), torch.arange(W).float(), indexing="ij",
    )
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestSweepFunctions:
    """Synthetic-CAM tests for ``compute_iou_sweep`` / ``summarize_iou_sweep``."""

    def test_perfect_prediction_gives_iou_best_1(self) -> None:
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=12).unsqueeze(0)
        cam = gt.clone()
        ious = compute_iou_sweep(cam, gt)
        s = summarize_iou_sweep(ious)
        assert math.isclose(s["cam_iou_best"], 1.0, abs_tol=1e-6)

    def test_zero_prediction_is_zero_against_nontrivial_gt(self) -> None:
        """All-zero CAM, non-empty GT: every threshold > 0 gives IoU=0
        (pred is empty, GT non-empty -> intersection=0, union>0). At
        threshold=0 the prediction is "all foreground", which gives a
        nonzero (but < 1) IoU. ``cam_iou_best`` is the MAX over the sweep,
        so it's the all-foreground IoU, NOT zero.

        The right invariant is: ``cam_iou_best_thr == 0`` (the only "good"
        threshold is the trivial all-fg one); see also
        ``test_diffuse_activation_low_optimal_thr``.
        """
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=12).unsqueeze(0)
        cam = torch.zeros_like(gt)
        ious = compute_iou_sweep(cam, gt)
        s = summarize_iou_sweep(ious)
        # iou at tau=0: pred=ones, intersection=|GT|, union=H*W => IoU=|GT|/(H*W)
        n_fg = gt.sum().item()
        expected_iou_at_0 = n_fg / (H * W)
        assert math.isclose(ious[0].item(), expected_iou_at_0, abs_tol=1e-6)
        # Every threshold > 0 -> empty pred, IoU=0.
        for ti in range(1, ious.numel()):
            assert ious[ti].item() == 0.0
        assert s["cam_iou_best_thr"] == 0.0

    def test_sharp_peak_high_optimal_thr(self) -> None:
        """CAM with a clean 0/1 step on GT -> every threshold in (0, 1]
        gives IoU=1. The tie-break in ``summarize_iou_sweep`` returns the
        HIGHEST tied threshold, which here is 1.0 (the sharpness diagnostic
        signal: "all activation is concentrated above tau=1.0", i.e. fully
        sharp)."""
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=12).unsqueeze(0)
        cam = gt.clone()
        ious = compute_iou_sweep(cam, gt)
        s = summarize_iou_sweep(ious)
        assert math.isclose(s["cam_iou_best"], 1.0, abs_tol=1e-6)
        assert s["cam_iou_best_thr"] >= 0.5, (
            f"sharp peaks should give high optimal threshold, got "
            f"{s['cam_iou_best_thr']}"
        )

    def test_diffuse_activation_low_optimal_thr(self) -> None:
        """CAM whose peak is OUTSIDE the GT (e.g. centred elsewhere) is
        diffuse w.r.t. the target -- the only way to overlap the GT is to
        threshold near zero (predict everything as foreground), so
        ``cam_iou_best_thr`` collapses to 0."""
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=10).unsqueeze(0)
        # Activation is a Gaussian centred in the corner, far from GT centre.
        yy, xx = torch.meshgrid(
            torch.arange(H).float(), torch.arange(W).float(), indexing="ij",
        )
        d = torch.sqrt((yy - 5) ** 2 + (xx - 5) ** 2)
        cam = torch.exp(-d / 8.0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        ious = compute_iou_sweep(cam, gt)
        s = summarize_iou_sweep(ious)
        assert s["cam_iou_best_thr"] <= 0.5, (
            f"diffuse-w.r.t.-GT activation should give low optimal threshold, "
            f"got {s['cam_iou_best_thr']}"
        )

    def test_auc_in_unit_interval(self) -> None:
        """Any non-degenerate CAM -> ``cam_iou_auc`` in [0, 1]."""
        torch.manual_seed(0)
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=12).unsqueeze(0)
        cam = torch.rand(1, H, W)
        ious = compute_iou_sweep(cam, gt)
        s = summarize_iou_sweep(ious)
        assert 0.0 <= s["cam_iou_auc"] <= 1.0

    def test_auc_monotone_in_quality(self) -> None:
        """Perfect prediction + increasing Gaussian noise -> AUC decreases
        monotonically (within Gaussian-noise tolerance)."""
        torch.manual_seed(0)
        H = W = 64
        gt = _gt_circle(H, W, cx=32, cy=32, r=12).unsqueeze(0)
        # Two-level CAM (1.0 in GT, 0.0 outside) is what an "ideal" model
        # would produce after min-max normalisation.
        ideal = gt.clone()
        aucs = []
        for sigma in [0.0, 0.2, 0.4, 0.6]:
            torch.manual_seed(int(sigma * 100))
            noisy = ideal + sigma * torch.randn_like(ideal)
            noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
            ious = compute_iou_sweep(noisy, gt)
            s = summarize_iou_sweep(ious)
            aucs.append(s["cam_iou_auc"])
        # Monotone non-increasing.
        for i in range(len(aucs) - 1):
            assert aucs[i] >= aucs[i + 1] - 0.02, (
                f"AUC not monotone-non-increasing across noise levels: {aucs}"
            )

    def test_subset_deterministic(self) -> None:
        """Same seed -> same subset across two independent calls."""
        names = [f"img_{i:04d}" for i in range(500)]
        a = select_deterministic_subset(names, subset_size=100, seed=1234)
        b = select_deterministic_subset(names, subset_size=100, seed=1234)
        assert a == b
        assert len(a) == 100
        assert len(set(a)) == 100  # no duplicates
        # Different seed -> usually different subset.
        c = select_deterministic_subset(names, subset_size=100, seed=42)
        assert a != c

    def test_first_per_class_references_alphabetical(self) -> None:
        names = ["a_apple", "b_banana", "c_cherry"]
        cti = {0: [2, 0], 1: [1, 0]}  # class 0 in cherry+apple; class 1 in banana+apple
        out = first_per_class_references(cti, names, num_classes=3)
        assert out == {0: "a_apple", 1: "a_apple"}  # class 2 absent
        assert 2 not in out

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="must agree"):
            compute_iou_sweep(torch.zeros(2, 4, 4), torch.zeros(2, 4, 5))
        with pytest.raises(ValueError, match=r"\(N, H, W\)"):
            compute_iou_sweep(torch.zeros(2, 4, 4, 4), torch.zeros(2, 4, 4, 4))
        with pytest.raises(ValueError, match="must agree"):
            summarize_iou_sweep(torch.zeros(5), torch.zeros(6))
        with pytest.raises(ValueError, match=">= 2 thresholds"):
            summarize_iou_sweep(torch.zeros(1), torch.zeros(1))


# ---------------------------------------------------------------------------
# OnlineCAMIoU end-to-end (mini synthetic PlantSeg layout)
# ---------------------------------------------------------------------------


def _make_mini_plantseg(
    root: Path,
    n_train: int,
    n_val: int,
    num_classes: int,
    image_size: int = 32,
) -> Path:
    """Create a tiny synthetic PlantSeg layout with random images +
    multi-class masks."""
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        mask_dir = root / "annotations" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            stem = f"{split}_{i:04d}"
            img = (rng.integers(0, 256, size=(image_size, image_size, 3))).astype(
                np.uint8,
            )
            Image.fromarray(img).save(img_dir / f"{stem}.jpg")
            # Each image gets exactly one foreground class (rotating).
            cls = (i % num_classes) + 1  # 1-indexed (PlantSeg convention)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            mask[image_size // 4 : 3 * image_size // 4,
                 image_size // 4 : 3 * image_size // 4] = cls
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")
    # Binary GT masks for val (a separate directory in the real layout).
    bin_dir = root.parent / "gt_binary_val"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_val):
        stem = f"val_{i:04d}"
        bin_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        bin_mask[image_size // 4 : 3 * image_size // 4,
                 image_size // 4 : 3 * image_size // 4] = 255
        Image.fromarray(bin_mask).save(bin_dir / f"{stem}.png")
    return bin_dir


@pytest.fixture
def mini_plantseg(tmp_path: Path) -> tuple[Path, Path]:
    """Mini PlantSeg layout: 16 train + 12 val, 4 classes, 32x32."""
    root = tmp_path / "plantseg"
    bin_dir = _make_mini_plantseg(
        root, n_train=16, n_val=12, num_classes=4, image_size=32,
    )
    return root, bin_dir


class TestOnlineCAMIoU:
    """End-to-end against a tiny synthetic dataset on disk."""

    def test_subset_deterministic_across_inits(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        root, bin_dir = mini_plantseg
        a = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=8, image_size=32, eval_batch_size=4,
        )
        b = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=8, image_size=32, eval_batch_size=4,
        )
        assert a.query_names == b.query_names
        assert len(a.query_names) == 8

    def test_subset_filters_to_existing_gt(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        """Drop a few binary GT files and verify they're filtered out."""
        root, bin_dir = mini_plantseg
        for missing in ("val_0000", "val_0001"):
            (bin_dir / f"{missing}.png").unlink()
        m = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=12, image_size=32,
        )
        assert "val_0000" not in m.query_names
        assert "val_0001" not in m.query_names

    def test_kill_switch_disables_metric(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        """``enabled=False`` -> no I/O, no subset, no eval."""
        root, bin_dir = mini_plantseg
        m = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=8, image_size=32, enabled=False,
        )
        assert m.query_names == []
        assert not m.should_run(0)
        assert not m.should_run(99)
        # A non-existent path would have raised during __init__ if I/O happened.
        bad = OnlineCAMIoU(
            plantseg_root=Path("/does/not/exist"),
            gt_binary_dir=Path("/does/not/exist"),
            enabled=False,
        )
        assert bad.evaluate(model=None, device=torch.device("cpu")) == {}  # type: ignore[arg-type]

    def test_every_n_epochs_cadence(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        root, bin_dir = mini_plantseg
        m = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=4, image_size=32, every_n_epochs=3,
        )
        # Cadence is ``(epoch + 1) % every_n_epochs == 0``: epochs 2, 5, 8, ...
        assert m.should_run(2)
        assert m.should_run(5)
        assert not m.should_run(0)
        assert not m.should_run(1)
        assert not m.should_run(3)
        assert not m.should_run(4)

    def test_evaluate_returns_logged_keys_and_finite_values(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        """Run ``evaluate`` against a real (untrained) SPDNet on CPU and
        check the three expected keys are emitted with finite values."""
        root, bin_dir = mini_plantseg
        m = OnlineCAMIoU(
            plantseg_root=root, gt_binary_dir=bin_dir, num_classes=4,
            subset_size=8, image_size=32, eval_batch_size=4,
        )
        torch.manual_seed(0)
        model = SPDNet(
            num_classes=4, fpn_channels=32, pretrained=False, fusion_mode="spatial",
        )
        device = torch.device("cpu")
        model.to(device)
        out = m.evaluate(model, device)
        for key in ("cam_iou_best", "cam_iou_best_thr", "cam_iou_auc"):
            assert key in out, f"missing key {key!r}"
            assert math.isfinite(out[key]), f"{key} not finite: {out[key]}"
        assert 0.0 <= out["cam_iou_best"] <= 1.0
        assert 0.0 <= out["cam_iou_best_thr"] <= 1.0
        assert 0.0 <= out["cam_iou_auc"] <= 1.0

    def test_evaluate_empty_when_disabled(
        self, mini_plantseg: tuple[Path, Path],
    ) -> None:
        m = OnlineCAMIoU(
            plantseg_root=mini_plantseg[0], gt_binary_dir=mini_plantseg[1],
            num_classes=4, subset_size=4, image_size=32, enabled=False,
        )
        out = m.evaluate(model=None, device=torch.device("cpu"))  # type: ignore[arg-type]
        assert out == {}
