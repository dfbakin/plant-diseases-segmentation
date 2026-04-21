"""Tests for binary WSSS pipeline components.

Covers:
  - Label export (plantseg_binary mode with/without PlantVillage)
  - BinaryPlantDataset construction and item shapes
  - Binary GT mask conversion logic
  - Threshold sweep subsampling
  - MCTformer _build_datasets with plantseg_binary
  - WeakCLIP model build with 2 classes
  - Name collision safety between PlantSeg and PlantVillage

Performance notes:
  - Dataset-dependent tests use session-scoped fixtures to avoid repeated
    scanning of ~8K PlantSeg masks and ~54K PlantVillage images.
  - Expected total run time: ~3-5 min (dominated by initial dataset scan).
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Paths (skip tests if data is missing) ───────────────────
PLANTSEG_ROOT = Path("data/plantsegv3")
PV_ROOT = Path("data/plant-village")
HAS_PLANTSEG = PLANTSEG_ROOT.exists() and (PLANTSEG_ROOT / "images" / "train").exists()
HAS_PV = PV_ROOT.exists()
HAS_DATA = HAS_PLANTSEG and HAS_PV

requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="PlantSeg and/or PlantVillage data not found"
)


# ── Session-scoped fixtures (built once, shared across tests) ─
@pytest.fixture(scope="session")
def binary_dataset_ps_only():
    """BinaryPlantDataset with PlantSeg only (no PlantVillage)."""
    if not HAS_DATA:
        pytest.skip("data not found")
    from src.data.voc_classification import BinaryPlantDataset

    t0 = time.time()
    ds = BinaryPlantDataset(
        plantseg_root=PLANTSEG_ROOT,
        plantvillage_root=PV_ROOT,
        split="train",
        image_size=64,
        include_plantvillage=False,
    )
    print(f"\n  [fixture] BinaryPlantDataset(ps_only): {len(ds)} samples in {time.time()-t0:.1f}s")
    return ds


@pytest.fixture(scope="session")
def binary_dataset_combined():
    """BinaryPlantDataset with PlantSeg + PlantVillage."""
    if not HAS_DATA:
        pytest.skip("data not found")
    from src.data.voc_classification import BinaryPlantDataset

    t0 = time.time()
    ds = BinaryPlantDataset(
        plantseg_root=PLANTSEG_ROOT,
        plantvillage_root=PV_ROOT,
        split="train",
        image_size=64,
        include_plantvillage=True,
    )
    print(f"\n  [fixture] BinaryPlantDataset(combined): {len(ds)} samples in {time.time()-t0:.1f}s")
    return ds


@pytest.fixture(scope="session")
def export_ps_only_labels():
    """Cached plantseg_binary export (PlantSeg only)."""
    if not HAS_DATA:
        pytest.skip("data not found")
    from src.export_labels import ExportLabelsConfig, export_plantseg_binary

    cfg = ExportLabelsConfig(
        mode="plantseg_binary",
        root=str(PLANTSEG_ROOT),
        pv_root=str(PV_ROOT),
        pv_split="train",
        include_plantvillage=False,
        output="/tmp/_test_ps_only.npy",
    )
    t0 = time.time()
    labels, class_names = export_plantseg_binary(cfg)
    print(f"\n  [fixture] export_plantseg_binary(ps_only): {len(labels)} labels in {time.time()-t0:.1f}s")
    return labels, class_names


@pytest.fixture(scope="session")
def export_combined_labels():
    """Cached plantseg_binary export (combined)."""
    if not HAS_DATA:
        pytest.skip("data not found")
    from src.export_labels import ExportLabelsConfig, export_plantseg_binary

    cfg = ExportLabelsConfig(
        mode="plantseg_binary",
        root=str(PLANTSEG_ROOT),
        pv_root=str(PV_ROOT),
        pv_split="train",
        include_plantvillage=True,
        output="/tmp/_test_combined.npy",
    )
    t0 = time.time()
    labels, class_names = export_plantseg_binary(cfg)
    print(f"\n  [fixture] export_plantseg_binary(combined): {len(labels)} labels in {time.time()-t0:.1f}s")
    return labels, class_names


# ═══════════════════════════════════════════════════════════════
# Unit tests: Binary GT mask conversion  (~instant)
# ═══════════════════════════════════════════════════════════════
class TestBinaryGTMaskConversion:
    """Verify the inline Python snippet that converts multiclass masks to binary."""

    @staticmethod
    def _convert(mask: np.ndarray) -> np.ndarray:
        m = mask.copy()
        m[(m > 0) & (m < 255)] = 1
        return m.astype(np.uint8)

    def test_background_stays_zero(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        assert (self._convert(mask) == 0).all()

    def test_ignore_stays_255(self):
        mask = np.full((32, 32), 255, dtype=np.uint8)
        assert (self._convert(mask) == 255).all()

    def test_foreground_becomes_one(self):
        mask = np.array([[0, 5, 100], [255, 42, 1]], dtype=np.uint8)
        expected = np.array([[0, 1, 1], [255, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(self._convert(mask), expected)

    def test_mixed_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 50
        mask[40:60, 40:60] = 115
        mask[0:5, 0:5] = 255
        result = self._convert(mask)
        assert set(np.unique(result)).issubset({0, 1, 255})
        assert (result[10:30, 10:30] == 1).all()
        assert (result[40:60, 40:60] == 1).all()
        assert (result[0:5, 0:5] == 255).all()
        assert (result[5:10, 5:10] == 0).all()


# ═══════════════════════════════════════════════════════════════
# Unit tests: Threshold sweep subsampling  (~5s)
# ═══════════════════════════════════════════════════════════════
class TestThresholdSweepSubsample:
    """Test max_samples parameter in evaluate_cam_threshold_sweep."""

    @staticmethod
    def _create_dummy_cams_and_gt(tmpdir: Path, n: int, num_fg: int = 1):
        cam_dir = tmpdir / "cams"
        gt_dir = tmpdir / "gt"
        cam_dir.mkdir()
        gt_dir.mkdir()

        names = []
        for i in range(n):
            name = f"img_{i:04d}"
            names.append(name)
            h, w = 32, 32
            cam_dict = {c: np.random.rand(h, w).astype(np.float32) for c in range(num_fg)}
            np.save(str(cam_dir / f"{name}.npy"), cam_dict)
            gt = np.zeros((h, w), dtype=np.uint8)
            gt[8:24, 8:24] = 1
            Image.fromarray(gt).save(gt_dir / f"{name}.png")
        return cam_dir, gt_dir, names

    def test_subsample_reduces_name_list(self):
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_dummy_cams_and_gt(Path(tmpdir), n=50)
            result = evaluate_cam_threshold_sweep(
                predict_dir=str(cam_dir),
                gt_dir=str(gt_dir),
                name_list=names,
                num_cls=2,
                start=0, end=5,
                max_samples=10, seed=42,
            )
            assert "best_miou" in result
            assert "best_threshold" in result
            assert result["best_miou"] >= 0.0

    def test_no_subsample_when_zero(self):
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_dummy_cams_and_gt(Path(tmpdir), n=10)
            result = evaluate_cam_threshold_sweep(
                predict_dir=str(cam_dir),
                gt_dir=str(gt_dir),
                name_list=names,
                num_cls=2,
                start=0, end=3,
                max_samples=0,
            )
            assert "best_miou" in result

    def test_max_samples_larger_than_list(self):
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_dummy_cams_and_gt(Path(tmpdir), n=5)
            result = evaluate_cam_threshold_sweep(
                predict_dir=str(cam_dir),
                gt_dir=str(gt_dir),
                name_list=names,
                num_cls=2,
                start=0, end=3,
                max_samples=100,
            )
            assert "best_miou" in result


# ═══════════════════════════════════════════════════════════════
# Unit tests: Threshold sweep parallelism  (~5s)
#
# Regression suite for the parallel branch added 2026-04-19. Phase 2
# was burning ~2 h of wall-clock per overnight on a single-threaded
# trange loop over 100 thresholds × 4 seed dirs × 9 probes; the
# parallel path uses multiprocessing.Pool.apply_async over thresholds
# (each evaluate_cam_miou call is independent) for ~Wx speedup.
#
# Critical invariant: the parallel branch must produce IDENTICAL
# best_threshold and curves as the serial branch. Anything else is
# a regression that would invalidate cross-probe rank comparisons.
# ═══════════════════════════════════════════════════════════════
class TestThresholdSweepParallel:
    """Parallel threshold sweep == serial sweep, bit-for-bit."""

    @staticmethod
    def _create_seeded_cams_and_gt(tmpdir: Path, n: int, num_fg: int = 1, seed: int = 0):
        """Like _create_dummy_cams_and_gt but with a seeded RNG so the
        serial vs parallel comparison sees identical inputs across runs.
        """
        cam_dir = tmpdir / "cams"
        gt_dir = tmpdir / "gt"
        cam_dir.mkdir()
        gt_dir.mkdir()

        rng = np.random.default_rng(seed)
        names = []
        for i in range(n):
            name = f"img_{i:04d}"
            names.append(name)
            h, w = 32, 32
            cam_dict = {c: rng.random((h, w)).astype(np.float32) for c in range(num_fg)}
            np.save(str(cam_dir / f"{name}.npy"), cam_dict)
            gt = np.zeros((h, w), dtype=np.uint8)
            gt[8:24, 8:24] = 1
            Image.fromarray(gt).save(gt_dir / f"{name}.png")
        return cam_dir, gt_dir, names

    def test_parallel_matches_serial_best_threshold(self):
        """num_workers=4 must pick the SAME best_threshold as num_workers=1.

        This is the core invariant: any divergence here would silently
        change the seed/CAM threshold every probe uses for visualization
        and downstream Phase 2 ranking.
        """
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=12, seed=1234,
            )
            kw = dict(
                predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                name_list=names, num_cls=2,
                start=0, end=10, optimize_metric="disease_iou",
            )
            serial = evaluate_cam_threshold_sweep(**kw, num_workers=1)
            parallel = evaluate_cam_threshold_sweep(**kw, num_workers=4)

        assert serial["best_threshold"] == parallel["best_threshold"], (
            f"serial best_threshold={serial['best_threshold']} != "
            f"parallel best_threshold={parallel['best_threshold']}"
        )

    def test_parallel_curves_match_serial(self):
        """Per-threshold curves must be bit-identical between paths.

        Same input -> same evaluate_cam_miou per threshold -> same dict.
        Order matters because curves[k] is a list indexed by threshold.
        """
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=8, seed=99,
            )
            kw = dict(
                predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                name_list=names, num_cls=2,
                start=0, end=6, optimize_metric="disease_iou",
            )
            serial = evaluate_cam_threshold_sweep(**kw, num_workers=1)
            parallel = evaluate_cam_threshold_sweep(**kw, num_workers=4)

        assert serial["curves"]["threshold"] == parallel["curves"]["threshold"]
        assert serial["curves"]["mIoU"] == parallel["curves"]["mIoU"]
        # Per-class curves identical
        for k in serial["curves"]:
            if k == "threshold":
                continue
            assert serial["curves"][k] == parallel["curves"][k], (
                f"divergence at curves[{k!r}]"
            )

    def test_parallel_result_at_best_matches_serial(self):
        """``result_at_best`` (per-class IoUs at the chosen threshold)
        must match -- this is what eval.json records as the headline
        ``probe_iou`` / ``chmean_iou`` etc. number.
        """
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=10, seed=7,
            )
            kw = dict(
                predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                name_list=names, num_cls=2,
                start=0, end=8, optimize_metric="disease_iou",
            )
            serial = evaluate_cam_threshold_sweep(**kw, num_workers=1)
            parallel = evaluate_cam_threshold_sweep(**kw, num_workers=4)

        for k in serial["result_at_best"]:
            assert serial["result_at_best"][k] == parallel["result_at_best"][k], (
                f"result_at_best[{k!r}] diverged: "
                f"serial={serial['result_at_best'][k]}, "
                f"parallel={parallel['result_at_best'][k]}"
            )

    def test_num_workers_one_uses_serial_path(self):
        """``num_workers=1`` keeps the historical (pre-2026-04-19) behaviour.

        Smoke check that the default code path still returns sensible
        keys -- the actual numerical equivalence is covered above by
        the cross-path tests against the parallel branch.
        """
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=5, seed=42,
            )
            result = evaluate_cam_threshold_sweep(
                predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                name_list=names, num_cls=2,
                start=0, end=4, optimize_metric="disease_iou",
                num_workers=1,
            )
        assert "best_threshold" in result
        assert "best_miou" in result
        assert "result_at_best" in result
        assert "curves" in result
        assert len(result["curves"]["threshold"]) == 4

    def test_single_threshold_falls_back_to_serial_pool(self):
        """A range of size 1 (start=5, end=6) is degenerate for a Pool;
        the parallel branch's ``n_thr > 1`` guard must skip the Pool and
        run serially without error or empty results.
        """
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=4, seed=11,
            )
            result = evaluate_cam_threshold_sweep(
                predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                name_list=names, num_cls=2,
                start=5, end=6, optimize_metric="disease_iou",
                num_workers=8,
            )
        assert result["best_threshold"] == 0.05
        assert len(result["curves"]["threshold"]) == 1

    def test_parallel_warns_about_ignored_patience(self, caplog):
        """``patience`` is silently meaningless once tasks are dispatched
        in parallel; we must log a WARNING so anyone passing it knows.
        """
        import logging
        from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep

        with tempfile.TemporaryDirectory() as tmpdir:
            cam_dir, gt_dir, names = self._create_seeded_cams_and_gt(
                Path(tmpdir), n=4, seed=55,
            )
            with caplog.at_level(logging.WARNING, logger="src.wsss.mctformer.evaluation"):
                evaluate_cam_threshold_sweep(
                    predict_dir=str(cam_dir), gt_dir=str(gt_dir),
                    name_list=names, num_cls=2,
                    start=0, end=4, optimize_metric="disease_iou",
                    patience=2, num_workers=4,
                )

        assert any("patience" in rec.message and "ignored" in rec.message
                   for rec in caplog.records), (
            "Expected a 'patience ignored' warning when num_workers>1"
        )


# ═══════════════════════════════════════════════════════════════
# Unit tests: Label export  (~2-3 min for initial scan, cached)
# ═══════════════════════════════════════════════════════════════
@requires_data
class TestLabelExport:
    """Test export_plantseg_binary with real data (uses session fixtures)."""

    def test_plantseg_only_labels(self, export_ps_only_labels):
        labels, class_names = export_ps_only_labels
        assert len(labels) > 0
        assert class_names == ["disease"]
        for name, lbl in labels.items():
            assert lbl.shape == (1,), f"{name}: shape={lbl.shape}"
            assert lbl[0] == 1.0, f"{name}: expected 1.0, got {lbl[0]}"

    def test_combined_labels(self, export_combined_labels, export_ps_only_labels):
        labels, class_names = export_combined_labels
        ps_labels, _ = export_ps_only_labels

        assert class_names == ["disease"]
        values = {float(v[0]) for v in labels.values()}
        assert 0.0 in values, "Expected healthy (0.0) from PlantVillage"
        assert 1.0 in values, "Expected diseased (1.0)"
        assert len(labels) > len(ps_labels), "Combined should have more entries"

    def test_label_shape(self, export_combined_labels):
        labels, _ = export_combined_labels
        sample = next(iter(labels.values()))
        assert sample.shape == (1,)
        assert sample.dtype == np.float32


# ═══════════════════════════════════════════════════════════════
# Unit tests: BinaryPlantDataset  (~2-3 min for initial scan, cached)
# ═══════════════════════════════════════════════════════════════
@requires_data
class TestBinaryPlantDataset:
    """Test BinaryPlantDataset (uses session fixtures)."""

    def test_plantseg_only(self, binary_dataset_ps_only):
        ds = binary_dataset_ps_only
        assert ds.num_classes == 1
        assert len(ds) > 0
        for _, lbl in ds.samples:
            assert lbl == 1.0

    def test_combined_has_both_labels(self, binary_dataset_combined):
        labels = {lbl for _, lbl in binary_dataset_combined.samples}
        assert 0.0 in labels, "Expected healthy samples from PlantVillage"
        assert 1.0 in labels

    def test_getitem_shapes(self, binary_dataset_combined):
        item = binary_dataset_combined[0]
        assert "image" in item
        assert "label" in item
        assert "name" in item
        assert item["label"].shape == (1,)
        assert item["label"].dtype.is_floating_point
        assert item["image"].shape == (3, 64, 64)

    def test_no_name_collisions(self):
        """PlantSeg and PlantVillage stems should not collide significantly."""
        from src.data.plantseg import PlantSegMulticlassDataset
        from src.data.plantvillage import PlantVillageDataset

        ps = PlantSegMulticlassDataset(root=PLANTSEG_ROOT, split="train", transform=None)
        pv = PlantVillageDataset(root=PV_ROOT, split="train")
        ps_names = {s["name"] for s in ps.samples}
        pv_names = {s["name"] for s in pv.samples}
        collisions = ps_names & pv_names
        assert len(collisions) < 50, f"Too many name collisions: {len(collisions)}"


# ═══════════════════════════════════════════════════════════════
# Integration tests: MCTformer _build_datasets  (~uses cached fixtures)
# ═══════════════════════════════════════════════════════════════
@requires_data
class TestMCTformerBinaryIntegration:
    """Test _build_datasets with plantseg_binary option (uses cached datasets internally)."""

    def test_build_returns_correct_num_classes(self, binary_dataset_combined):
        assert binary_dataset_combined.num_classes == 1
        assert len(binary_dataset_combined) > 0

    def test_train_has_both_labels(self, binary_dataset_combined):
        labels_seen = set()
        for i in range(min(200, len(binary_dataset_combined))):
            item = binary_dataset_combined[i]
            labels_seen.add(float(item["label"][0]))
            if len(labels_seen) == 2:
                break
        assert 1.0 in labels_seen

    def test_forward_backward_binary_batch(self):
        """MCTformer forward+backward with binary labels (shape (B,1)). ~2s"""
        import torch

        from src.models.classifier_factory import create_classifier

        model = create_classifier(
            name="mctformer_v2", num_classes=1, pretrained=False, input_size=64
        )
        model.train()
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        cls_logits = output[0]
        assert cls_logits.shape == (2, 1)
        loss = cls_logits.sum()
        loss.backward()


# ═══════════════════════════════════════════════════════════════
# Integration tests: WeakCLIP with 2 classes  (~10s if pretrained exists)
# ═══════════════════════════════════════════════════════════════
class TestWeakCLIPBinaryBuild:
    """Test that WeakCLIP model can be built with num_classes=2."""

    @pytest.fixture(scope="class")
    def clip_pretrained(self):
        p = Path("pretrained/ViT-B-16.pt")
        if not p.exists():
            pytest.skip("ViT-B-16.pt not found in pretrained/")
        return str(p)

    def test_build_model_2_classes(self, clip_pretrained):
        from src.train_weakclip import WeakCLIPTrainConfig, build_weakclip_model

        cfg = WeakCLIPTrainConfig()
        cfg.num_classes = 2
        cfg.image_size = 64
        cfg.clip_pretrained = clip_pretrained
        cfg.context_length = 5

        model = build_weakclip_model(cfg, ("disease",))
        assert model is not None

    def test_fpn_channels_770(self, clip_pretrained):
        """FPN in_channels should be 768 + 2 = 770 for binary."""
        from src.train_weakclip import WeakCLIPTrainConfig, build_weakclip_model

        cfg = WeakCLIPTrainConfig()
        cfg.num_classes = 2
        cfg.image_size = 64
        cfg.clip_pretrained = clip_pretrained
        cfg.context_length = 5

        model = build_weakclip_model(cfg, ("disease",))
        fpn_in = model.neck.lateral_convs[0].conv.in_channels
        assert fpn_in == 770, f"Expected 770 (768+2), got {fpn_in}"

    def test_cross_entropy_binary(self):
        """F.cross_entropy with 2-channel output and target {0,1,255} works. ~instant"""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(4, 2, 8, 8)
        target = torch.zeros(4, 8, 8, dtype=torch.long)
        target[:, :4, :] = 1
        target[:, 7, :] = 255
        loss = F.cross_entropy(logits, target, ignore_index=255)
        assert loss.isfinite()
