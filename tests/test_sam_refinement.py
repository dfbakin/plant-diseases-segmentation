"""Tests for SAM1 pseudomask refinement helpers and pipeline.

Covers:
  - filter_small_components: noise removal
  - mask_to_logits: shape, dtype, value range
  - mask_to_bbox: padding, empty mask, full mask
  - sample_points_from_cam: point count, label correctness, spatial diversity
  - _farthest_point_sample: greedy selection
  - Integration: end-to-end refinement on 5 synthetic images with SAM-base
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# Unit tests: filter_small_components
# ═══════════════════════════════════════════════════════════════
class TestFilterSmallComponents:
    def test_removes_single_pixel(self):
        from src.refine_masks_sam import filter_small_components

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # 100 px
        mask[50, 50] = 1  # 1 px
        result = filter_small_components(mask, min_size=10)
        assert result[50, 50] == 0
        assert result[15, 15] == 1

    def test_preserves_large_components(self):
        from src.refine_masks_sam import filter_small_components

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 10:50] = 1  # 1600 px
        mask[60:65, 60:65] = 1  # 25 px
        result = filter_small_components(mask, min_size=30)
        assert result[30, 30] == 1
        assert result[62, 62] == 0

    def test_no_filtering_when_zero(self):
        from src.refine_masks_sam import filter_small_components

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0, 0] = 1
        result = filter_small_components(mask, min_size=0)
        assert result[0, 0] == 1

    def test_empty_mask(self):
        from src.refine_masks_sam import filter_small_components

        mask = np.zeros((32, 32), dtype=np.uint8)
        result = filter_small_components(mask, min_size=10)
        assert (result == 0).all()

    def test_multiple_small_components_removed(self):
        from src.refine_masks_sam import filter_small_components

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10, 10] = 1
        mask[30, 30] = 1
        mask[50, 50] = 1
        mask[70:80, 70:80] = 1  # 100 px
        result = filter_small_components(mask, min_size=5)
        assert result[10, 10] == 0
        assert result[30, 30] == 0
        assert result[50, 50] == 0
        assert result[75, 75] == 1


# ═══════════════════════════════════════════════════════════════
# Unit tests: mask_to_logits
# ═══════════════════════════════════════════════════════════════
class TestMaskToLogits:
    def test_shape(self):
        from src.refine_masks_sam import mask_to_logits

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 1
        logits = mask_to_logits(mask)
        assert logits.shape == (1, 256, 256)

    def test_dtype_float32(self):
        from src.refine_masks_sam import mask_to_logits
        import torch

        mask = np.zeros((64, 64), dtype=np.uint8)
        logits = mask_to_logits(mask)
        assert logits.dtype == torch.float32

    def test_value_range(self):
        from src.refine_masks_sam import mask_to_logits

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        logits = mask_to_logits(mask)
        assert logits.min().item() == pytest.approx(-6.0)
        assert logits.max().item() == pytest.approx(6.0)

    def test_all_zeros(self):
        from src.refine_masks_sam import mask_to_logits

        mask = np.zeros((64, 64), dtype=np.uint8)
        logits = mask_to_logits(mask)
        assert logits.max().item() == pytest.approx(-6.0)

    def test_all_ones(self):
        from src.refine_masks_sam import mask_to_logits

        mask = np.ones((64, 64), dtype=np.uint8)
        logits = mask_to_logits(mask)
        assert logits.min().item() == pytest.approx(6.0)

    def test_custom_target_size(self):
        from src.refine_masks_sam import mask_to_logits

        mask = np.ones((64, 64), dtype=np.uint8)
        logits = mask_to_logits(mask, target_size=128)
        assert logits.shape == (1, 128, 128)


# ═══════════════════════════════════════════════════════════════
# Unit tests: mask_to_bbox
# ═══════════════════════════════════════════════════════════════
class TestMaskToBbox:
    def test_basic_bbox(self):
        from src.refine_masks_sam import mask_to_bbox

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:70] = 1
        bbox = mask_to_bbox(mask, padding_frac=0.0)
        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert x_min == 30
        assert y_min == 20
        assert x_max == 70
        assert y_max == 40

    def test_padding(self):
        from src.refine_masks_sam import mask_to_bbox

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 1
        bbox_no_pad = mask_to_bbox(mask, padding_frac=0.0)
        bbox_pad = mask_to_bbox(mask, padding_frac=0.05)
        assert bbox_pad[0] < bbox_no_pad[0]  # x_min should be smaller
        assert bbox_pad[1] < bbox_no_pad[1]  # y_min should be smaller
        assert bbox_pad[2] > bbox_no_pad[2]  # x_max should be larger
        assert bbox_pad[3] > bbox_no_pad[3]  # y_max should be larger

    def test_empty_mask_returns_none(self):
        from src.refine_masks_sam import mask_to_bbox

        mask = np.zeros((64, 64), dtype=np.uint8)
        assert mask_to_bbox(mask) is None

    def test_full_mask(self):
        from src.refine_masks_sam import mask_to_bbox

        mask = np.ones((64, 64), dtype=np.uint8)
        bbox = mask_to_bbox(mask, padding_frac=0.0)
        assert bbox == [0, 0, 64, 64]

    def test_clamped_to_image_bounds(self):
        from src.refine_masks_sam import mask_to_bbox

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:5, 0:5] = 1
        bbox = mask_to_bbox(mask, padding_frac=0.1)
        assert bbox[0] >= 0
        assert bbox[1] >= 0


# ═══════════════════════════════════════════════════════════════
# Unit tests: sample_points_from_cam
# ═══════════════════════════════════════════════════════════════
class TestSamplePointsFromCam:
    @staticmethod
    def _make_cam_and_mask():
        cam = np.random.RandomState(42).rand(100, 100).astype(np.float32)
        cam[20:40, 20:40] = 0.98  # high activation
        cam[80:90, 80:90] = 0.02  # low activation
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[15:45, 15:45] = 1
        return cam, mask

    def test_returns_correct_types(self):
        from src.refine_masks_sam import sample_points_from_cam

        cam, mask = self._make_cam_and_mask()
        pts, lbls = sample_points_from_cam(cam, mask, num_pos=3, num_neg=3)
        assert isinstance(pts, list)
        assert isinstance(lbls, list)
        assert all(isinstance(p, list) and len(p) == 2 for p in pts)

    def test_positive_labels_are_one(self):
        from src.refine_masks_sam import sample_points_from_cam

        cam, mask = self._make_cam_and_mask()
        pts, lbls = sample_points_from_cam(cam, mask, num_pos=3, num_neg=0)
        assert all(l == 1 for l in lbls)

    def test_negative_labels_are_zero(self):
        from src.refine_masks_sam import sample_points_from_cam

        cam, mask = self._make_cam_and_mask()
        pts, lbls = sample_points_from_cam(cam, mask, num_pos=0, num_neg=3)
        assert all(l == 0 for l in lbls)

    def test_respects_num_pos_neg(self):
        from src.refine_masks_sam import sample_points_from_cam

        cam, mask = self._make_cam_and_mask()
        pts, lbls = sample_points_from_cam(cam, mask, num_pos=2, num_neg=2)
        pos_count = sum(1 for l in lbls if l == 1)
        neg_count = sum(1 for l in lbls if l == 0)
        assert pos_count <= 2
        assert neg_count <= 2

    def test_xy_format(self):
        """Points should be [x, y] (col, row) not [row, col]."""
        from src.refine_masks_sam import sample_points_from_cam

        cam = np.zeros((100, 200), dtype=np.float32)
        cam[10:20, 150:180] = 0.99
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:20, 150:180] = 1

        pts, _ = sample_points_from_cam(cam, mask, num_pos=1, num_neg=0)
        if pts:
            x, y = pts[0]
            assert 150 <= x < 180, f"x={x} should be in column range [150, 180)"
            assert 10 <= y < 20, f"y={y} should be in row range [10, 20)"


# ═══════════════════════════════════════════════════════════════
# Unit tests: _select_mask
# ═══════════════════════════════════════════════════════════════
class TestSelectMask:
    def test_best_iou_picks_highest_score(self):
        import torch
        from src.refine_masks_sam import _select_mask

        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        masks[0, 10:20, 10:20] = 1  # 100 px
        masks[1, 10:40, 10:40] = 1  # 900 px
        masks[2, 10:30, 10:30] = 1  # 400 px
        scores = torch.tensor([0.7, 0.95, 0.8])
        idx, score = _select_mask(masks, scores, "best_iou")
        assert idx == 1
        assert score == pytest.approx(0.95)

    def test_smallest_area_picks_smallest(self):
        import torch
        from src.refine_masks_sam import _select_mask

        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        masks[0, 10:30, 10:30] = 1  # 400 px
        masks[1, 10:15, 10:15] = 1  # 25 px  (smallest)
        masks[2, 10:40, 10:40] = 1  # 900 px
        scores = torch.tensor([0.9, 0.6, 0.95])
        idx, score = _select_mask(masks, scores, "smallest_area")
        assert idx == 1
        assert score == pytest.approx(0.6)

    def test_smallest_area_skips_empty(self):
        import torch
        from src.refine_masks_sam import _select_mask

        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        masks[0] = 0  # empty
        masks[1, 5:10, 5:10] = 1  # 25 px
        masks[2, 5:30, 5:30] = 1  # 625 px
        scores = torch.tensor([0.99, 0.5, 0.7])
        idx, score = _select_mask(masks, scores, "smallest_area")
        assert idx == 1, "Should skip empty mask[0] and pick mask[1]"

    def test_smallest_area_all_empty_falls_back(self):
        import torch
        from src.refine_masks_sam import _select_mask

        masks = np.zeros((3, 64, 64), dtype=np.uint8)
        scores = torch.tensor([0.3, 0.9, 0.5])
        idx, _ = _select_mask(masks, scores, "smallest_area")
        assert idx == 1, "All empty: should fall back to best_iou"


# ═══════════════════════════════════════════════════════════════
# Unit tests: _farthest_point_sample
# ═══════════════════════════════════════════════════════════════
class TestFarthestPointSample:
    def test_returns_k_points(self):
        from src.refine_masks_sam import _farthest_point_sample

        candidates = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]])
        result = _farthest_point_sample(candidates, k=3, min_distance=1)
        assert len(result) == 3

    def test_returns_all_when_fewer_than_k(self):
        from src.refine_masks_sam import _farthest_point_sample

        candidates = np.array([[0, 0], [10, 10]])
        result = _farthest_point_sample(candidates, k=5, min_distance=1)
        assert len(result) == 2

    def test_min_distance_respected(self):
        from src.refine_masks_sam import _farthest_point_sample

        candidates = np.array([[0, 0], [1, 0], [2, 0], [100, 0]])
        result = _farthest_point_sample(candidates, k=3, min_distance=50)
        assert len(result) <= 2  # only (0,0) and (100,0) are far enough apart


# ═══════════════════════════════════════════════════════════════
# Integration test: end-to-end binary refinement with SAM-base
# ═══════════════════════════════════════════════════════════════
HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

requires_gpu = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


@requires_gpu
class TestSAMRefinementIntegration:
    """End-to-end test with SAM-base on 5 synthetic images."""

    @pytest.fixture(scope="class")
    def test_data(self, tmp_path_factory):
        """Create 5 synthetic images, masks, and CAMs."""
        tmpdir = tmp_path_factory.mktemp("sam_test")
        img_dir = tmpdir / "images"
        mask_dir = tmpdir / "masks"
        cam_dir = tmpdir / "cams"
        out_dir = tmpdir / "output"
        for d in (img_dir, mask_dir, cam_dir):
            d.mkdir()

        rng = np.random.RandomState(42)
        names = []
        for i in range(5):
            name = f"test_{i:03d}"
            names.append(name)
            h, w = 480, 640

            img = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(img).save(str(img_dir / f"{name}.jpg"))

            mask = np.zeros((h, w), dtype=np.uint8)
            r0, c0 = 100 + i * 20, 150 + i * 30
            mask[r0 : r0 + 200, c0 : c0 + 200] = 1
            mask[10, 10] = 1  # tiny noise
            Image.fromarray(mask).save(str(mask_dir / f"{name}.png"))

            cam = rng.rand(h, w).astype(np.float32) * 0.5
            cam[r0 : r0 + 200, c0 : c0 + 200] = 0.9
            cam_dict = {0: cam}
            np.save(str(cam_dir / f"{name}.npy"), cam_dict)

        return {
            "img_dir": img_dir,
            "mask_dir": mask_dir,
            "cam_dir": cam_dir,
            "out_dir": out_dir,
            "names": names,
        }

    def test_mask_only_mode(self, test_data):
        from src.refine_masks_sam import SAMRefineConfig, refine_masks_sam

        out = test_data["out_dir"] / "mask_only"
        cfg = SAMRefineConfig(
            image_dir=str(test_data["img_dir"]),
            mask_dir=str(test_data["mask_dir"]),
            output_dir=str(out),
            model_name="facebook/sam-vit-base",
            prompt_mode="mask_only",
            num_classes=2,
            batch_size=2,
            min_component_size=10,
        )
        refine_masks_sam(cfg)

        for name in test_data["names"]:
            p = out / f"{name}.png"
            assert p.exists(), f"Missing output: {p}"
            arr = np.array(Image.open(p))
            assert arr.shape == (480, 640)
            assert set(np.unique(arr)).issubset({0, 1})

    def test_mask_and_points_mode(self, test_data):
        from src.refine_masks_sam import SAMRefineConfig, refine_masks_sam

        out = test_data["out_dir"] / "mask_and_points"
        cfg = SAMRefineConfig(
            image_dir=str(test_data["img_dir"]),
            mask_dir=str(test_data["mask_dir"]),
            cam_dir=str(test_data["cam_dir"]),
            output_dir=str(out),
            model_name="facebook/sam-vit-base",
            prompt_mode="mask_and_points",
            num_classes=2,
            batch_size=2,
            min_component_size=10,
            num_pos_points=2,
            num_neg_points=2,
        )
        refine_masks_sam(cfg)

        for name in test_data["names"]:
            p = out / f"{name}.png"
            assert p.exists()
            arr = np.array(Image.open(p))
            assert set(np.unique(arr)).issubset({0, 1})
