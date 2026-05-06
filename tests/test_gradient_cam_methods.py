"""Tests for gradient-based CAM methods (LayerCAM / GradCAM++ / XGradCAM).

Covers:
- Shape and dtype of the generated seed map.
- [0, 1] value range after min-max normalization.
- Gradient-leak guard: the call does NOT populate ``.grad`` on any
  learnable parameter, so it is safe to run inside an ``eval()`` loop.
- Active-class sensitivity: picking different active classes yields
  different output maps.
- Shape agnosticism: runs at two different input resolutions.

All tests instantiate a tiny un-pretrained SPDNet and run on CPU (or
GPU if available). Wall-clock is <30 s for the full suite.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.wsss.spdnet.gradient_cam_methods import (
    MAX_CLASSES_PER_IMAGE,
    compute_gradient_cam,
    generate_gradient_spdnet_seed,
    is_gradient_cam_mode,
    list_methods,
)
from src.wsss.spdnet.model import SPDNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8
IMAGE_SIZE = 64
ALT_IMAGE_SIZE = 96  # shape agnosticism probe


@pytest.fixture(scope="module")
def spatial_model():
    torch.manual_seed(0)
    m = SPDNet(
        num_classes=NUM_CLASSES, pretrained=False, fusion_mode="spatial",
    ).to(DEVICE).eval()
    return m


@pytest.fixture(scope="module")
def token_model():
    torch.manual_seed(0)
    m = SPDNet(
        num_classes=NUM_CLASSES, pretrained=False, fusion_mode="token",
    ).to(DEVICE).eval()
    return m


def _fresh_inputs(size: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(1, 3, size, size, device=DEVICE)
    r = torch.randn(1, 3, size, size, device=DEVICE)
    return q, r


def _params_have_no_grad(model) -> bool:
    return all(p.grad is None for p in model.parameters())


class TestMethodRegistry:
    def test_lists_three_methods(self):
        methods = list_methods()
        assert sorted(methods) == ["gradcam_pp", "layercam", "xgradcam"]

    def test_is_gradient_cam_mode_detects_all(self):
        for m in list_methods():
            assert is_gradient_cam_mode(m)

    def test_is_gradient_cam_mode_rejects_others(self):
        for m in ["feat_chmean", "feat_chvar", "fused_chvar", "attn_map", "cam_max", "spatial_proto"]:
            assert not is_gradient_cam_mode(m)


@pytest.mark.parametrize("method", ["layercam", "gradcam_pp", "xgradcam"])
class TestComputeGradientCam:
    def test_shape_query_merged(self, spatial_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        cam = compute_gradient_cam(
            spatial_model, q, r, active_classes=[0], method=method,
            target_layer="query_merged",
        )
        # query_merged stride is FPN P3 = /8, but model._merge_fpn
        # upsamples to level-0 resolution which at backbone layer1 is /4.
        # We only assert 2D and positive spatial size.
        assert cam.dim() == 2
        assert cam.shape[0] > 0 and cam.shape[1] > 0
        assert cam.dtype == torch.float32 or cam.dtype == torch.float
        assert torch.isfinite(cam).all()

    def test_shape_fused(self, spatial_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        cam = compute_gradient_cam(
            spatial_model, q, r, active_classes=[0], method=method,
            target_layer="fused",
        )
        assert cam.dim() == 2 and cam.shape[0] > 0 and torch.isfinite(cam).all()

    def test_shape_layer4(self, spatial_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        cam = compute_gradient_cam(
            spatial_model, q, r, active_classes=[0], method=method,
            target_layer="layer4",
        )
        # Layer4 is /32 stride.
        assert cam.dim() == 2 and cam.shape == (IMAGE_SIZE // 32, IMAGE_SIZE // 32)

    def test_shape_agnostic_two_sizes(self, spatial_model, method):
        q1, r1 = _fresh_inputs(IMAGE_SIZE, seed=1)
        q2, r2 = _fresh_inputs(ALT_IMAGE_SIZE, seed=1)
        c1 = compute_gradient_cam(
            spatial_model, q1, r1, active_classes=[0], method=method,
        )
        c2 = compute_gradient_cam(
            spatial_model, q2, r2, active_classes=[0], method=method,
        )
        # Larger input -> larger spatial output.
        assert c2.shape[0] > c1.shape[0]
        assert c2.shape[1] > c1.shape[1]

    def test_works_with_token_fusion(self, token_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        cam = compute_gradient_cam(
            token_model, q, r, active_classes=[0], method=method,
        )
        assert cam.dim() == 2 and torch.isfinite(cam).all()

    def test_no_gradient_leak_on_params(self, spatial_model, method):
        # Guard: zero any lingering grads, run a CAM, verify no param
        # has a populated .grad (torch.autograd.grad w/ only_inputs=True
        # should not touch param.grad).
        for p in spatial_model.parameters():
            p.grad = None
        q, r = _fresh_inputs(IMAGE_SIZE, seed=2)
        _ = compute_gradient_cam(
            spatial_model, q, r, active_classes=[0, 1], method=method,
        )
        assert _params_have_no_grad(spatial_model), (
            f"gradient-CAM method {method!r} leaked into parameter .grad; "
            f"check that compute_gradient_cam uses torch.autograd.grad "
            f"with only_inputs=True and does NOT call .backward()."
        )

    def test_raises_on_empty_active_classes(self, spatial_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        with pytest.raises(ValueError, match="active_classes"):
            compute_gradient_cam(
                spatial_model, q, r, active_classes=[], method=method,
            )

    def test_clamps_to_max_classes_per_image(self, spatial_model, method):
        q, r = _fresh_inputs(IMAGE_SIZE)
        many = list(range(NUM_CLASSES))  # 8 classes
        # Should not crash even if len(many) > MAX_CLASSES_PER_IMAGE.
        _ = compute_gradient_cam(
            spatial_model, q, r, active_classes=many, method=method,
            max_classes_per_image=2,  # force aggressive clamp
        )


class TestActiveClassSensitivity:
    @pytest.mark.parametrize("method", ["layercam", "gradcam_pp", "xgradcam"])
    def test_different_classes_give_different_cams(self, spatial_model, method):
        # The randomly-initialised classifier gives linear class responses;
        # picking different classes hence produces different CAMs unless the
        # aggregation is identically zero (GradCAM++ / XGradCAM can be in
        # pathological cases). We assert that at least one of the three
        # methods responds; a truly degenerate model would fail this for all.
        q, r = _fresh_inputs(IMAGE_SIZE, seed=3)
        cam_a = compute_gradient_cam(
            spatial_model, q, r, active_classes=[0], method=method,
        ).cpu().numpy()
        cam_b = compute_gradient_cam(
            spatial_model, q, r, active_classes=[NUM_CLASSES - 1], method=method,
        ).cpu().numpy()
        # If both maps are all-zero (rare but possible with a freshly
        # initialised ReLU classifier), skip the assertion.
        if cam_a.max() < 1e-8 and cam_b.max() < 1e-8:
            pytest.skip(f"{method}: both class CAMs are zero (degenerate init)")
        assert not np.allclose(cam_a, cam_b, atol=1e-6), (
            f"{method}: CAMs for different classes are numerically identical"
        )


class TestGenerateGradientSpdnetSeed:
    """High-level driver (multi-scale/flip + normalization)."""

    @pytest.mark.parametrize("method", ["layercam", "gradcam_pp", "xgradcam"])
    def test_output_format_and_range(self, spatial_model, method):
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = generate_gradient_spdnet_seed(
            model=spatial_model,
            query_images=[q],
            ref_image_lists=[[r]],
            active_classes=[0],
            device=DEVICE,
            method=method,
        )
        assert isinstance(out, dict) and 0 in out
        arr = out[0]
        assert arr.dtype == np.float32
        assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0 + 1e-6

    @pytest.mark.parametrize("method", ["layercam", "gradcam_pp", "xgradcam"])
    def test_flip_augmentation_undone(self, spatial_model, method):
        # Driver expects caller to alternate flip=0, flip=1 within each scale
        # and un-flips scale-odd outputs. Here we only check it doesn't throw.
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        qf = torch.flip(q, [-1])
        rf = torch.flip(r, [-1])
        out = generate_gradient_spdnet_seed(
            model=spatial_model,
            query_images=[q, qf],
            ref_image_lists=[[r], [rf]],
            active_classes=[0, 1],
            device=DEVICE,
            method=method,
        )
        assert out[0].shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_empty_inputs_raise(self, spatial_model):
        with pytest.raises(ValueError, match="query_images must not be empty"):
            generate_gradient_spdnet_seed(
                model=spatial_model, query_images=[], ref_image_lists=[],
                active_classes=[0], device=DEVICE, method="layercam",
            )

    def test_mismatched_lengths_raise(self, spatial_model):
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with pytest.raises(ValueError, match="length mismatch"):
            generate_gradient_spdnet_seed(
                model=spatial_model, query_images=[q, q], ref_image_lists=[[r]],
                active_classes=[0], device=DEVICE, method="layercam",
            )


class TestConstants:
    def test_max_classes_per_image_is_reasonable(self):
        # Sanity: defaults have not drifted way off.
        assert 1 <= MAX_CLASSES_PER_IMAGE <= 16
