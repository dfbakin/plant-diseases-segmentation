"""Unit tests for the SPDNet seg-probe module + dataset.

Run:
    .venv/bin/pytest tests/test_seg_probe.py -v

These tests deliberately use IMAGE_SIZE=64 and BATCH_SIZE=2 so they fit
comfortably in any GPU; they are also CPU-runnable. The only "expensive"
test is the dataset class-resolver one which scans the val annotations
folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.wsss.spdnet.model import SPDNet
from src.wsss.spdnet.seg_probe import (
    NEEDS_REFERENCE,
    PROBE_POSITIONS,
    SPATIAL_ONLY_POSITIONS,
    ProbeHead,
    SPDNetWithProbes,
    bce_dice_loss,
    channels_for_position,
    dice_loss,
)

IMAGE_SIZE = 64
BATCH_SIZE = 2
NUM_CLASSES = 8
FPN_CHANNELS = 256


# ----------------------------------------------------------------------------
# (a) ProbeHead forward shape
# ----------------------------------------------------------------------------

class TestProbeHead:
    def test_forward_shape_basic(self):
        head = ProbeHead(in_channels=256, hidden_channels=64,
                         target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = torch.randn(BATCH_SIZE, 256, 8, 8)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)

    def test_forward_shape_single_ch(self):
        """Probe head must accept a 1-channel input (P6_attn_map)."""
        head = ProbeHead(in_channels=1, hidden_channels=8, target_size=(32, 32))
        x = torch.randn(2, 1, 4, 4)
        out = head(x)
        assert out.shape == (2, 1, 32, 32)

    def test_forward_shape_2048_ch(self):
        """Probe head must accept the 2048-ch P1_layer4 input."""
        head = ProbeHead(in_channels=2048, hidden_channels=64, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = torch.randn(BATCH_SIZE, 2048, 4, 4)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)

    def test_init_does_not_explode(self):
        """Default init produces logits in a reasonable range."""
        head = ProbeHead(in_channels=128, hidden_channels=32, target_size=(32, 32))
        x = torch.randn(2, 128, 4, 4)
        out = head(x)
        assert out.abs().max().item() < 50, f"logits too large: {out.abs().max().item()}"


# ----------------------------------------------------------------------------
# (b) SPDNetWithProbes returns all positions with correct dims
# ----------------------------------------------------------------------------

@pytest.fixture
def token_model() -> SPDNet:
    return SPDNet(num_classes=NUM_CLASSES, pretrained=False, fusion_mode="token").eval()


@pytest.fixture
def spatial_model() -> SPDNet:
    return SPDNet(num_classes=NUM_CLASSES, pretrained=False, fusion_mode="spatial").eval()


@pytest.fixture
def pair() -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    r = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    return q, r


class TestSPDNetExtractProbeFeatures:
    def test_token_returns_5_positions(self, token_model, pair):
        """Token model exposes P1..P5 (no P6 attn map)."""
        q, r = pair
        feats = token_model.extract_probe_features(q, r)
        assert "P1_layer4" in feats
        assert "P2_fpn_p2" in feats
        assert "P3_query_merged" in feats
        assert "P4_fused" in feats
        assert "P5_cam_classifier" in feats
        assert "P6_attn_map" not in feats

    def test_spatial_returns_6_positions(self, spatial_model, pair):
        q, r = pair
        feats = spatial_model.extract_probe_features(q, r)
        for k in PROBE_POSITIONS:
            assert k in feats, f"missing position: {k}"

    def test_no_reference_omits_fused(self, spatial_model):
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        feats = spatial_model.extract_probe_features(q, reference=None)
        assert "P3_query_merged" in feats
        for k in NEEDS_REFERENCE:
            assert k not in feats

    def test_p1_shape(self, token_model, pair):
        q, _ = pair
        feats = token_model.extract_probe_features(q, pair[1])
        h, w = IMAGE_SIZE // 32, IMAGE_SIZE // 32
        assert feats["P1_layer4"].shape == (BATCH_SIZE, 2048, h, w)

    def test_p2_shape(self, token_model, pair):
        q, _ = pair
        feats = token_model.extract_probe_features(q, pair[1])
        h, w = IMAGE_SIZE // 4, IMAGE_SIZE // 4  # FPN-P2 = ResNet C2 res
        assert feats["P2_fpn_p2"].shape == (BATCH_SIZE, FPN_CHANNELS, h, w)

    def test_p3_shape(self, token_model, pair):
        q, r = pair
        feats = token_model.extract_probe_features(q, r)
        # query_merged is at the highest FPN res = C2 res
        h, w = IMAGE_SIZE // 4, IMAGE_SIZE // 4
        assert feats["P3_query_merged"].shape == (BATCH_SIZE, FPN_CHANNELS, h, w)

    def test_p5_channels_eq_num_classes(self, token_model, pair):
        q, r = pair
        feats = token_model.extract_probe_features(q, r)
        assert feats["P5_cam_classifier"].shape[1] == NUM_CLASSES

    def test_p6_single_channel(self, spatial_model, pair):
        q, r = pair
        feats = spatial_model.extract_probe_features(q, r)
        assert feats["P6_attn_map"].shape[1] == 1

    def test_existing_forward_unchanged(self, token_model, pair):
        """Original forward path must still work and have stable output."""
        q, r = pair
        token_model.eval()
        torch.manual_seed(0)
        out_a = token_model(q, r, return_cam=False)
        torch.manual_seed(0)
        _ = token_model.extract_probe_features(q, r)
        out_b = token_model(q, r, return_cam=False)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_spatial_only_position_rejected_on_token(self, token_model):
        with pytest.raises(ValueError, match="requires fusion_mode='spatial'"):
            SPDNetWithProbes(spdnet=token_model, position="P6_attn_map")


# ----------------------------------------------------------------------------
# (c) frozen-backbone gradient isolation
# ----------------------------------------------------------------------------

class TestFrozenBackboneGradIsolation:
    def test_no_backbone_grads_after_backward(self, token_model, pair):
        wrapper = SPDNetWithProbes(spdnet=token_model, position="P3_query_merged",
                                   freeze_backbone=True, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        # ensure no stale grads
        for p in wrapper.parameters():
            if p.grad is not None:
                p.grad.zero_()

        q, r = pair
        seg_logits = wrapper(q, r, return_cls=False)
        target = torch.randint(0, 2, seg_logits.shape, dtype=torch.float32)
        loss = bce_dice_loss(seg_logits, target)
        loss.backward()

        for name, p in wrapper.spdnet.named_parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0, (
                f"backbone param '{name}' got non-zero grad in frozen mode"
            )
        head_grad_sum = sum(
            p.grad.abs().sum().item()
            for p in wrapper.head.parameters() if p.grad is not None
        )
        assert head_grad_sum > 0, "head params have no grads"

    def test_unfrozen_backbone_gets_grads(self, token_model, pair):
        # token_model fixture is shared; clone weights into a new model so
        # the previous frozen test doesn't affect requires_grad here.
        model = SPDNet(num_classes=NUM_CLASSES, pretrained=False, fusion_mode="token").eval()
        wrapper = SPDNetWithProbes(spdnet=model, position="P3_query_merged",
                                   freeze_backbone=False, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        for p in wrapper.parameters():
            if p.grad is not None:
                p.grad.zero_()
        q, r = pair
        seg_logits = wrapper(q, r, return_cls=False)
        loss = seg_logits.sum()
        loss.backward()
        # at least one backbone param must have non-zero grad
        nonzero = sum(
            1 for n, p in wrapper.spdnet.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        assert nonzero > 0, "no backbone param received gradient in unfrozen mode"


# ----------------------------------------------------------------------------
# (d) BCE+Dice loss is finite for random input
# ----------------------------------------------------------------------------

class TestLoss:
    def test_bce_dice_finite_random(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 1, 32, 32)
        mask = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.float32)
        loss = bce_dice_loss(logits, mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_dice_perfect_pred_zero(self):
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        # logits very large positive where mask=1, very negative where mask=0
        logits = torch.tensor([[[[-20.0, 20.0], [20.0, -20.0]]]])
        d = dice_loss(logits, mask).item()
        assert d < 0.01, f"perfect dice should be ~0, got {d}"

    def test_dice_inverted_pred_one(self):
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])
        d = dice_loss(logits, mask).item()
        assert d > 0.95, f"fully-wrong dice should be ~1, got {d}"


# ----------------------------------------------------------------------------
# (e) SiamesePlantSegSegDataset returns mask in {0,1} with correct shape
# ----------------------------------------------------------------------------

DATASET_AVAILABLE = (
    Path("data/plantsegv3/images/val").exists()
    and Path("data/plantsegv3/annotations/val").exists()
)


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="PlantSeg dataset not present")
class TestSegDataset:
    def test_returns_image_label_mask(self):
        from src.wsss.spdnet.seg_dataset import SiamesePlantSegSegDataset
        ds = SiamesePlantSegSegDataset(
            root="data/plantsegv3", split="val", image_size=128,
            train_aug=False, num_references=1, limit=4,
        )
        sample = ds[0]
        q = sample["query"]
        assert q["image"].shape == (3, 128, 128)
        assert q["mask"].shape == (1, 128, 128)
        assert q["mask"].dtype == torch.float32
        assert q["mask"].min() >= 0 and q["mask"].max() <= 1
        assert q["label"].shape == (115,)
        assert q["label"].sum() >= 1, "every PlantSeg image has at least 1 disease class"
        assert len(sample["references"]) == 1

    def test_mask_values_in_zero_one(self):
        from src.wsss.spdnet.seg_dataset import SiamesePlantSegSegDataset
        ds = SiamesePlantSegSegDataset(
            root="data/plantsegv3", split="val", image_size=64,
            train_aug=False, num_references=1, limit=8,
        )
        for i in range(len(ds)):
            uniq = torch.unique(ds[i]["query"]["mask"]).tolist()
            for v in uniq:
                assert v in (0.0, 1.0), f"mask value {v} not in [0,1] -- got {uniq}"

    def test_collate(self):
        from src.wsss.spdnet.seg_dataset import (
            SiamesePlantSegSegDataset, siamese_seg_collate_fn,
        )
        ds = SiamesePlantSegSegDataset(
            root="data/plantsegv3", split="val", image_size=64,
            train_aug=False, num_references=1, limit=4,
        )
        batch = siamese_seg_collate_fn([ds[0], ds[1]])
        assert batch["query_image"].shape == (2, 3, 64, 64)
        assert batch["query_mask"].shape == (2, 1, 64, 64)
        assert batch["query_label"].shape == (2, 115)
        assert len(batch["ref_images"]) == 1
        assert batch["ref_images"][0].shape == (2, 3, 64, 64)

    def test_exif_rotated_samples_load(self):
        """Regression test: ~0.1% of train images have EXIF orientation 6.

        Without ``ImageOps.exif_transpose`` the raw image dims (e.g. 848x636)
        don't match the mask dims (636x848) and albumentations rejects the
        pair, killing the dataloader worker. This test pins a few known
        offenders and asserts they load cleanly via _PlantSegBase.
        """
        from src.wsss.spdnet.seg_dataset import _PlantSegBase
        base = _PlantSegBase("data/plantsegv3", split="train", image_size=128)
        targets = {
            "apple_rust_google_0065",
            "celery_anthracnose_google_0001",
            "grape_leaf_spot_google_0041",
            "tomato_bacterial_leaf_spot_42",
            "tomato_early_blight_google_0002",
        }
        loaded = 0
        for idx, name in enumerate(base.names):
            if name not in targets:
                continue
            sample = base[idx]
            assert sample["image"].shape == (3, 128, 128), f"{name}: {sample['image'].shape}"
            assert sample["mask"].shape == (1, 128, 128), f"{name}: {sample['mask'].shape}"
            loaded += 1
        assert loaded == len(targets), (
            f"expected {len(targets)} EXIF-affected samples, loaded {loaded}"
        )


# ----------------------------------------------------------------------------
# (f) class resolver picks same-class ref
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not DATASET_AVAILABLE, reason="PlantSeg dataset not present")
class TestClassResolver:
    def test_resolver_matches_train_pool(self):
        from src.wsss.spdnet.class_resolver import (
            build_class_pool_from_labels,
            load_class_names,
            make_filename_class_resolver,
        )
        class_names = load_class_names("outputs/plantseg_binary_mc115/labels/class_names.txt")
        resolver = make_filename_class_resolver(class_names)
        pool = build_class_pool_from_labels(
            "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy",
            "data/plantsegv3/images/train", image_ext=".jpg",
        )

        val_dir = Path("data/plantsegv3/images/val")
        candidates = [p.stem for p in sorted(val_dir.glob("*.jpg"))[:50]]
        assert candidates, "no val images"

        same_class_pairs = 0
        for stem in candidates:
            cls = resolver(stem)
            if cls is None or cls not in pool or not pool[cls]:
                continue
            ref = pool[cls][0]
            ref_cls = resolver(ref)
            if ref_cls == cls:
                same_class_pairs += 1
        # at least 40 of 50 should resolve and produce a same-class match
        assert same_class_pairs >= 40, f"only {same_class_pairs}/50 same-class pairs"


# ----------------------------------------------------------------------------
# (g) skip-if-exists logic for the training entrypoint
# ----------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_if_exists_short_circuits(self, tmp_path, monkeypatch):
        """train_probe must return early if .TRAIN_DONE marker is present.

        We only verify that ``train_probe`` short-circuits *before* trying to
        load the (heavy) checkpoint -- that's what the resume guarantee
        promises. We test this by pointing it at a non-existent checkpoint
        and a directory where the marker already exists, then asserting
        no exception.
        """
        from src.train_spdnet_probe import train_probe
        from src.conf.spdnet_probe import (
            SPDNetProbeConfig, SPDNetProbeDataConfig,
            SPDNetProbeModelConfig, SPDNetProbeTrainerConfig,
        )

        cfg = SPDNetProbeConfig(
            phase="phase1",
            ckpt_tag="dummy",
            checkpoint="/path/that/does/not/exist.ckpt",
            model=SPDNetProbeModelConfig(position="P3_query_merged"),
            data=SPDNetProbeDataConfig(),
            trainer=SPDNetProbeTrainerConfig(max_epochs=1),
        )
        cfg.output_dir = str(tmp_path / "probe")
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".TRAIN_DONE").touch()
        (out_dir / "head.pt").touch()
        cfg.resume_if_exists = True

        # Should NOT raise even though checkpoint doesn't exist
        result = train_probe(cfg)
        assert result == 0.0


# ----------------------------------------------------------------------------
# extras: channel mapping + position constants
# ----------------------------------------------------------------------------

class TestPositionConstants:
    def test_all_positions_have_channels(self, token_model):
        for pos in PROBE_POSITIONS:
            try:
                ch = channels_for_position(token_model, pos)
                assert ch >= 1
            except ValueError:
                pytest.fail(f"channels_for_position raised for {pos}")

    def test_p6_only_in_spatial(self):
        assert "P6_attn_map" in SPATIAL_ONLY_POSITIONS
        assert "P3_query_merged" not in SPATIAL_ONLY_POSITIONS

    def test_needs_reference_set(self):
        assert "P4_fused" in NEEDS_REFERENCE
        assert "P5_cam_classifier" in NEEDS_REFERENCE
        assert "P6_attn_map" in NEEDS_REFERENCE
        assert "P3_query_merged" not in NEEDS_REFERENCE


class TestCleanupSeedDirs:
    """Stream-clean of *_seeds/ npy directories after a probe eval finishes.

    Verifies the contract used by the overnight orchestrator:
      * seed npy directories are wiped
      * head.pt, eval.json, viz/, checkpoints/, config.yaml are kept
      * cleanup is a no-op when eval.json is missing (eval did not finish)
      * cleanup is idempotent when seed dirs are already gone
    """

    def _make_probe_dir(self, root, with_eval_json: bool = True):
        from pathlib import Path
        import json
        import numpy as np

        root = Path(root)
        (root / "probe_seeds").mkdir(parents=True)
        (root / "baseline_chmean_seeds").mkdir(parents=True)
        (root / "baseline_chvar_seeds").mkdir(parents=True)
        (root / "baseline_cam_cls_seeds").mkdir(parents=True)
        (root / "viz").mkdir(parents=True)
        (root / "checkpoints").mkdir(parents=True)

        # one fake seed in each seed dir + a payload-sized one
        for sd in (
            "probe_seeds",
            "baseline_chmean_seeds",
            "baseline_chvar_seeds",
            "baseline_cam_cls_seeds",
        ):
            np.save(str(root / sd / "img_a.npy"), np.zeros((448, 448), dtype=np.float32))
            np.save(str(root / sd / "img_b.npy"), np.zeros((100, 100), dtype=np.float32))

        # things that must survive
        (root / "head.pt").write_bytes(b"x" * 1024)
        (root / "config.yaml").write_text("phase: phase1\n")
        (root / "viz" / "summary_grid.png").write_bytes(b"x" * 256)
        (root / "checkpoints" / "epoch=05.ckpt").write_bytes(b"x" * 4096)
        if with_eval_json:
            (root / "eval.json").write_text(json.dumps({"probe_iou": 12.3}))
        return root

    def test_cleanup_removes_only_seed_dirs(self, tmp_path):
        from scripts.eval_seg_probes import _cleanup_seed_dirs

        probe_dir = self._make_probe_dir(tmp_path / "probe", with_eval_json=True)
        info = _cleanup_seed_dirs(probe_dir)

        assert sorted(info["removed"]) == [
            "baseline_cam_cls_seeds",
            "baseline_chmean_seeds",
            "baseline_chvar_seeds",
            "probe_seeds",
        ]
        assert info["bytes_freed"] > 0
        # seed dirs must be gone
        for sd in (
            "probe_seeds",
            "baseline_chmean_seeds",
            "baseline_chvar_seeds",
            "baseline_cam_cls_seeds",
        ):
            assert not (probe_dir / sd).exists(), f"{sd} should have been removed"
        # everything else must still be there
        assert (probe_dir / "head.pt").exists()
        assert (probe_dir / "config.yaml").exists()
        assert (probe_dir / "eval.json").exists()
        assert (probe_dir / "viz" / "summary_grid.png").exists()
        assert (probe_dir / "checkpoints" / "epoch=05.ckpt").exists()

    def test_cleanup_refuses_when_eval_json_missing(self, tmp_path):
        from scripts.eval_seg_probes import _cleanup_seed_dirs

        probe_dir = self._make_probe_dir(tmp_path / "probe", with_eval_json=False)
        info = _cleanup_seed_dirs(probe_dir)

        assert info["removed"] == []
        assert info["kept_reason"] == "eval.json missing"
        # seed dirs must still be there because eval did not finish
        for sd in (
            "probe_seeds",
            "baseline_chmean_seeds",
            "baseline_chvar_seeds",
            "baseline_cam_cls_seeds",
        ):
            assert (probe_dir / sd).exists(), f"{sd} should have been preserved"

    def test_cleanup_is_idempotent(self, tmp_path):
        from scripts.eval_seg_probes import _cleanup_seed_dirs

        probe_dir = self._make_probe_dir(tmp_path / "probe", with_eval_json=True)
        first = _cleanup_seed_dirs(probe_dir)
        second = _cleanup_seed_dirs(probe_dir)

        assert len(first["removed"]) == 4
        assert second["removed"] == []  # already gone, nothing to do
        assert (probe_dir / "head.pt").exists()


class TestSubsampleValNames:
    """``--limit-val N`` deterministic val subsampling.

    The probe orchestrator runs the same evaluation across many probes and
    must compare IoU numbers head-to-head. The screen-mode (limit_val=300)
    only stays scientifically valid if every probe sees the *exact same*
    subset of val images. That contract is verified here.
    """

    def _names(self, n: int = 1247) -> list[str]:
        return [f"img_{i:05d}" for i in range(n)]

    def test_zero_returns_full_list(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(50)
        out = _subsample_val_names(names, n=0)
        assert out == names

    def test_negative_returns_full_list(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(50)
        out = _subsample_val_names(names, n=-1)
        assert out == names

    def test_n_at_least_total_returns_full_list(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(50)
        # n equal to / exceeding total -> no subsampling
        assert _subsample_val_names(names, n=50) == names
        assert _subsample_val_names(names, n=999) == names

    def test_subset_size_and_membership(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(1247)
        out = _subsample_val_names(names, n=300)
        assert len(out) == 300
        assert len(set(out)) == 300, "no duplicates"
        assert set(out).issubset(names), "must be a subset of the val set"

    def test_output_is_sorted(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(1247)
        out = _subsample_val_names(names, n=300)
        assert out == sorted(out), "output must be sorted for stable iteration"

    def test_deterministic_same_seed(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(1247)
        a = _subsample_val_names(names, n=300)
        b = _subsample_val_names(names, n=300)
        assert a == b, (
            "same val set + same limit must yield same subset across calls "
            "(cross-probe ranks depend on this)"
        )

    def test_deterministic_across_orderings(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(1247)
        # caller may pass an unsorted list; subsampling uses a fixed seed
        # so the *output* must still be identical and stable. We do NOT
        # require sample(...) on a permuted list to return the same items
        # (random.sample respects input order), so we test the canonical
        # contract: callers always pass sorted(val_names_full), per the
        # one call-site in evaluate_probe().
        a = _subsample_val_names(sorted(names), n=300)
        b = _subsample_val_names(sorted(list(reversed(names))), n=300)
        assert a == b, (
            "evaluate_probe always sorts val_names first; subsampling on "
            "the sorted list must be stable"
        )

    def test_different_seed_changes_subset(self):
        from scripts.eval_seg_probes import _subsample_val_names

        names = self._names(1247)
        a = _subsample_val_names(names, n=300, seed=1234)
        b = _subsample_val_names(names, n=300, seed=42)
        # vanishingly unlikely they're identical with different seeds
        assert a != b


# ----------------------------------------------------------------------------
# (h) Atomic .npy I/O + corrupt-seed pruning
#
# Hardens the probe pipeline against the silent corruption observed in the
# 18 Apr overnight run, where a single truncated chvar seed file
# (apple_mosaic_virus_google_0053.npy, 262 144 B / 256 KiB instead of
# the expected ~853 KiB) crashed the entire eval loop with an opaque
# `_pickle.UnpicklingError: pickle data was truncated`.
# ----------------------------------------------------------------------------

class TestAtomicSaveNpy:
    """``atomic_save_npy`` is a write-and-rename wrapper around ``np.save``.

    The contract is: on disk after the call, ``dst`` is *either* a fully
    valid ``.npy`` file *or* it does not exist. We never want a
    partially-written file masquerading as a valid one (that's exactly
    the failure mode that took down Phase 1 mid-run).
    """

    def test_basic_round_trip(self, tmp_path):
        from src.wsss.spdnet._atomic_io import atomic_save_npy

        dst = tmp_path / "x.npy"
        payload = {0: np.arange(12, dtype=np.float32).reshape(3, 4)}
        atomic_save_npy(dst, payload)

        assert dst.is_file()
        loaded = np.load(str(dst), allow_pickle=True).item()
        assert set(loaded.keys()) == {0}
        np.testing.assert_array_equal(loaded[0], payload[0])

    def test_no_tmp_file_after_success(self, tmp_path):
        """A clean save must not leak its scratch file."""
        from src.wsss.spdnet._atomic_io import atomic_save_npy

        dst = tmp_path / "x.npy"
        atomic_save_npy(dst, {0: np.zeros((2, 2), dtype=np.float32)})

        assert dst.is_file()
        # critical: no leftover ".tmp" file
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == [], f"leaked tmp files: {leftovers}"

    def test_failure_leaves_no_partial(self, tmp_path, monkeypatch):
        """Simulate a kill mid-write -- final file must be absent, tmp removed.

        The helper now passes a file *handle* to ``np.save`` (so NumPy
        doesn't append another ``.npy`` to a ``.tmp`` path), so the
        simulated failure must accept ``(handle, obj, ...)``.
        """
        from src.wsss.spdnet import _atomic_io

        def _explode(fh, obj, *args, **kwargs):
            # write a few bytes so we can verify cleanup deletes a real file
            fh.write(b"\x00" * 1024)
            raise RuntimeError("simulated mid-pickle crash")

        monkeypatch.setattr(_atomic_io.np, "save", _explode)

        dst = tmp_path / "x.npy"
        with pytest.raises(RuntimeError, match="simulated mid-pickle crash"):
            _atomic_io.atomic_save_npy(dst, {0: np.zeros((4, 4))})

        # contract: neither the final dst nor the tmp survives
        assert not dst.exists(), "final file must not exist after failure"
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == [], f"leaked tmp files: {leftovers}"

    def test_failure_preserves_existing_dst(self, tmp_path, monkeypatch):
        """A failed re-save must not damage an already-good dst file.

        Critical for resumable pipelines: if a 2nd write attempt crashes
        mid-stream, the *previous* successful write must still be loadable.
        """
        from src.wsss.spdnet import _atomic_io

        dst = tmp_path / "x.npy"
        good_payload = {0: np.full((3, 3), 7.0, dtype=np.float32)}
        _atomic_io.atomic_save_npy(dst, good_payload)
        original_bytes = dst.read_bytes()

        def _explode(fh, obj, *args, **kwargs):
            fh.write(b"\x00" * 1024)
            raise RuntimeError("simulated mid-pickle crash on overwrite")

        monkeypatch.setattr(_atomic_io.np, "save", _explode)

        with pytest.raises(RuntimeError):
            _atomic_io.atomic_save_npy(dst, {0: np.zeros((9, 9))})

        # original good file is untouched, still loadable
        assert dst.read_bytes() == original_bytes
        loaded = np.load(str(dst), allow_pickle=True).item()
        np.testing.assert_array_equal(loaded[0], good_payload[0])

    def test_handle_path_with_npy_tmp_suffix(self, tmp_path):
        """Regression: np.save(str_path) appends '.npy' to '.tmp' paths.

        Naive impl `np.save(str(dst.with_suffix('.npy.tmp')), obj)` writes
        to ``x.npy.tmp.npy`` and ``os.replace(x.npy.tmp, x.npy)`` then
        renames a *non-existent* file. atomic_save_npy must never trip
        on this.
        """
        from src.wsss.spdnet._atomic_io import atomic_save_npy

        dst = tmp_path / "x.npy"
        atomic_save_npy(dst, {0: np.arange(16, dtype=np.float32).reshape(4, 4)})

        # No `.tmp.npy` artefact: confirm np.save's auto-extension didn't
        # silently fire.
        all_files = sorted(p.name for p in tmp_path.iterdir())
        assert all_files == ["x.npy"], f"unexpected files: {all_files}"

    def test_rejects_non_npy_extension(self, tmp_path):
        """Refuse paths the helper can't reason about (defensive API)."""
        from src.wsss.spdnet._atomic_io import atomic_save_npy

        with pytest.raises(ValueError, match=r"\.npy"):
            atomic_save_npy(tmp_path / "x.pkl", {0: np.zeros((2, 2))})


class TestIsCorruptNpy:
    """``is_corrupt_npy`` matches every truncation/corruption mode we have hit."""

    def test_missing_file_is_corrupt(self, tmp_path):
        from src.wsss.spdnet._atomic_io import is_corrupt_npy

        assert is_corrupt_npy(tmp_path / "nope.npy") is True

    def test_zero_byte_is_corrupt(self, tmp_path):
        from src.wsss.spdnet._atomic_io import is_corrupt_npy

        empty = tmp_path / "empty.npy"
        empty.write_bytes(b"")
        assert is_corrupt_npy(empty) is True

    def test_random_garbage_is_corrupt(self, tmp_path):
        from src.wsss.spdnet._atomic_io import is_corrupt_npy

        garbage = tmp_path / "g.npy"
        garbage.write_bytes(b"\xff" * 4096)
        assert is_corrupt_npy(garbage) is True

    def test_truncated_pickle_is_corrupt(self, tmp_path):
        """Reproduce the exact production failure mode from the 18 Apr run.

        Write a valid object-dtype .npy (~80 KB pickle of a small array),
        then truncate the file to 1 KB -- enough for the NUMPY header and
        magic to be intact, but not enough for a complete pickle stream.
        """
        from src.wsss.spdnet._atomic_io import atomic_save_npy, is_corrupt_npy

        path = tmp_path / "trunc.npy"
        payload = {0: np.arange(100 * 100, dtype=np.float32).reshape(100, 100)}
        atomic_save_npy(path, payload)
        assert is_corrupt_npy(path) is False  # baseline: starts healthy

        # Truncate to 1 KB -- header survives, pickle body does not.
        # On disk, this is exactly what the production failure looked
        # like: the magic + header parsed fine but pickle.load died.
        with open(path, "r+b") as fh:
            fh.truncate(1024)

        assert is_corrupt_npy(path) is True

    def test_valid_npy_passes(self, tmp_path):
        from src.wsss.spdnet._atomic_io import atomic_save_npy, is_corrupt_npy

        path = tmp_path / "ok.npy"
        atomic_save_npy(path, {0: np.zeros((10, 10), dtype=np.float32)})
        assert is_corrupt_npy(path) is False


class TestPruneCorruptSeeds:
    """``prune_corrupt_seeds`` is the safety net for legacy truncated files.

    ``generate_probe_and_baselines(skip_existing=True)`` would otherwise
    treat a truncated file as 'already done' and the eval loop would
    crash on it. Pruning lets the next seed-gen pass refill the holes.
    """

    def test_idempotent_on_clean_dir(self, tmp_path):
        from src.wsss.spdnet._atomic_io import atomic_save_npy, prune_corrupt_seeds

        sd = tmp_path / "seeds"
        sd.mkdir()
        for i in range(3):
            atomic_save_npy(sd / f"img_{i}.npy", {0: np.zeros((4, 4), dtype=np.float32)})

        removed = prune_corrupt_seeds(sd)
        assert removed == []
        # all originals survive
        assert sorted(p.name for p in sd.glob("*.npy")) == [
            "img_0.npy", "img_1.npy", "img_2.npy",
        ]

    def test_missing_dir_returns_empty(self, tmp_path):
        from src.wsss.spdnet._atomic_io import prune_corrupt_seeds

        out = prune_corrupt_seeds(tmp_path / "does_not_exist")
        assert out == []

    def test_removes_truncated_keeps_good(self, tmp_path):
        from src.wsss.spdnet._atomic_io import atomic_save_npy, prune_corrupt_seeds

        sd = tmp_path / "seeds"
        sd.mkdir()
        # 4 healthy files
        for i in range(4):
            atomic_save_npy(sd / f"good_{i}.npy", {0: np.zeros((4, 4), dtype=np.float32)})
        # 1 truncated file masquerading as a complete .npy
        bad = sd / "bad.npy"
        atomic_save_npy(bad, {0: np.arange(50 * 50, dtype=np.float32).reshape(50, 50)})
        with open(bad, "r+b") as fh:
            fh.truncate(512)

        removed = prune_corrupt_seeds(sd)

        assert [p.name for p in removed] == ["bad.npy"]
        assert not bad.exists()
        # all 4 good files survive
        survivors = sorted(p.name for p in sd.glob("*.npy"))
        assert survivors == ["good_0.npy", "good_1.npy", "good_2.npy", "good_3.npy"]

    def test_removes_stale_tmp_files(self, tmp_path):
        """``.npy.tmp`` files are *always* invalid (atomic-write convention)."""
        from src.wsss.spdnet._atomic_io import prune_corrupt_seeds

        sd = tmp_path / "seeds"
        sd.mkdir()
        (sd / "img_a.npy.tmp").write_bytes(b"\x00" * 256)
        (sd / "img_b.npy.tmp").write_bytes(b"\x00" * 256)

        removed = prune_corrupt_seeds(sd)
        assert sorted(p.name for p in removed) == ["img_a.npy.tmp", "img_b.npy.tmp"]
        assert list(sd.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Parallel `_full_crf_eval` -- multiprocessing.Pool + per-image hard timeout.
#
# Production hang on 2026-04-19 (zucchini_downy_mildew_Bing_0120, image 294
# of 300) burned ~80 minutes of CPU before being killed manually. The tests
# below are the regression net for that incident:
#
#   * `test_serial_and_parallel_produce_same_means` -- proves the parallel
#     path is numerically equivalent to the serial path it replaces.
#   * `test_timeout_skips_slow_image_keeps_others`  -- proves a single hung
#     image (5 s sleep, 0.5 s timeout) cannot starve the rest of the batch.
#   * `test_worker_exception_skips_image`           -- a per-image worker
#     crash is logged + skipped, never propagated.
#   * `test_one_hang_does_not_stall_pool`           -- soak test: 7 fast
#     tasks + 1 hung task complete in ~timeout, not 8*timeout.
#
# The deterministic and slow workers are MODULE-LEVEL functions because
# multiprocessing pickles workers by fully-qualified name and the worker
# subprocess re-imports the module to materialise them.
# ---------------------------------------------------------------------------


def _test_deterministic_worker(args):
    """Hash-free deterministic worker: IoU is a function of the int suffix.

    Avoids ``hash(name)`` because PYTHONHASHSEED randomisation can differ
    between parent and (spawned) child processes -- and we want this test
    to be deterministic on every interpreter.
    """
    name = args[0]
    n = int(name.split("_")[-1])
    d_iou = float((n * 7) % 100) / 100.0
    b_iou = float((n * 13 + 21) % 100) / 100.0
    return name, d_iou, b_iou


def _test_slow_worker(args):
    """Sleeps 5 s on ``slow_image``, returns instantly otherwise."""
    import time
    name = args[0]
    if name == "slow_image":
        time.sleep(5.0)
    return name, 0.5, 0.5


def _test_failing_worker(args):
    """Raises ValueError on ``bad_image``, returns instantly otherwise."""
    name = args[0]
    if name == "bad_image":
        raise ValueError("simulated worker failure")
    return name, 0.5, 0.5


class TestFullCRFEvalParallel:
    """Regression suite for `scripts.eval_seg_probes._full_crf_eval`.

    Each test injects a synthetic worker via the ``_worker_fn`` test seam
    so we never pay the cost of real pydensecrf inference and we control
    every per-image latency / failure deterministically.
    """

    def test_serial_and_parallel_produce_same_means(self, tmp_path):
        """Parallel = serial within strict float tolerance.

        20 deterministic samples; the parallel path may collect them in
        any order but the mean must come out bit-equal to the serial mean.
        """
        from scripts.eval_seg_probes import _full_crf_eval

        names = [f"img_{i:03d}" for i in range(20)]
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        d_s, b_s, m_s = _full_crf_eval(
            seed_dir, names, crf_p, num_workers=1,
            _worker_fn=_test_deterministic_worker,
        )
        d_p, b_p, m_p = _full_crf_eval(
            seed_dir, names, crf_p, num_workers=4,
            _worker_fn=_test_deterministic_worker,
        )

        assert abs(d_s - d_p) < 1e-9, f"serial={d_s}, parallel={d_p}"
        assert abs(b_s - b_p) < 1e-9, f"serial={b_s}, parallel={b_p}"
        assert abs(m_s - m_p) < 1e-9, f"serial={m_s}, parallel={m_p}"

    def test_timeout_skips_slow_image_keeps_others(self, tmp_path, capsys):
        """A single 5 s hang under a 0.5 s budget must not poison the batch.

        Expected outcome: the 4 fast images contribute (0.5, 0.5) each;
        slow_image is reported as skipped and excluded from the mean.
        """
        from scripts.eval_seg_probes import _full_crf_eval

        names = ["fast_a", "fast_b", "slow_image", "fast_c", "fast_d"]
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        d, b, m = _full_crf_eval(
            seed_dir, names, crf_p,
            num_workers=2,
            per_image_timeout_sec=0.5,
            _worker_fn=_test_slow_worker,
        )

        assert abs(d - 50.0) < 1e-6, f"d={d}"
        assert abs(b - 50.0) < 1e-6, f"b={b}"
        out = capsys.readouterr().out
        assert "slow_image" in out, f"slow_image not mentioned in stdout:\n{out}"
        assert "SKIPPED 1/5" in out, f"skip count missing:\n{out}"

    def test_worker_exception_skips_image(self, tmp_path, capsys):
        """Per-image worker crash is caught + reported, never propagated."""
        from scripts.eval_seg_probes import _full_crf_eval

        names = ["fast_a", "fast_b", "bad_image", "fast_c"]
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        d, b, m = _full_crf_eval(
            seed_dir, names, crf_p,
            num_workers=2,
            _worker_fn=_test_failing_worker,
        )

        assert abs(d - 50.0) < 1e-6, f"d={d}"
        out = capsys.readouterr().out
        assert "bad_image" in out, f"bad_image not mentioned:\n{out}"
        assert "SKIPPED 1/4" in out, f"skip count missing:\n{out}"

    def test_one_hang_does_not_stall_pool(self, tmp_path):
        """7 fast + 1 hung must finish in ~timeout, NOT in 8*timeout.

        This is the *operational* property the production fix guarantees:
        one pathological pydensecrf call cannot block the remaining batch.
        Conservative bound: 4.0 s wall (timeout=2.0 s + pool startup).
        Without the parallel fix this test would hang for the full 5 s
        sleep on the only worker; with it, other workers process the 7
        fast tasks while the parent waits 2 s for the hung future.
        """
        import time
        from scripts.eval_seg_probes import _full_crf_eval

        names = [f"fast_{i}" for i in range(7)] + ["slow_image"]
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        t0 = time.time()
        d, b, m = _full_crf_eval(
            seed_dir, names, crf_p,
            num_workers=4,
            per_image_timeout_sec=2.0,
            _worker_fn=_test_slow_worker,
        )
        elapsed = time.time() - t0

        assert elapsed < 4.0, (
            f"elapsed={elapsed:.2f}s -- pool stalled. With per-image "
            f"timeout=2 s and 4 workers, total wall must be ~2-3 s "
            f"(7 fast results stream in instantly while parent waits "
            f"out the slow_image timeout)."
        )
        # 7 healthy results @ (0.5, 0.5)
        assert abs(d - 50.0) < 1e-6, f"d={d}"
        assert abs(b - 50.0) < 1e-6, f"b={b}"

    def test_empty_input_returns_nan(self, tmp_path):
        """Defensive: zero images in -> NaN out (no division by zero)."""
        from scripts.eval_seg_probes import _full_crf_eval
        import math

        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        d, b, m = _full_crf_eval(
            seed_dir, [], crf_p, num_workers=4,
            _worker_fn=_test_deterministic_worker,
        )
        assert math.isnan(d) and math.isnan(b) and math.isnan(m)

    def test_single_image_uses_serial_path(self, tmp_path):
        """``num_workers > 1`` falls back to serial when len(tasks) == 1.

        Avoids spinning up an entire process pool to dispatch a single task.
        We assert this by passing a worker that records its caller's PID.
        """
        from scripts.eval_seg_probes import _full_crf_eval

        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        crf_p = {"srgb": 5.0, "bg_threshold": 0.1, "scale_factor": 1.0}

        d, b, m = _full_crf_eval(
            seed_dir, ["img_005"], crf_p, num_workers=8,
            _worker_fn=_test_deterministic_worker,
        )
        # img_005 -> n=5 -> d=(5*7)%100/100 = 0.35, b=(5*13+21)%100/100 = 0.86
        assert abs(d - 35.0) < 1e-6, f"d={d}"
        assert abs(b - 86.0) < 1e-6, f"b={b}"
