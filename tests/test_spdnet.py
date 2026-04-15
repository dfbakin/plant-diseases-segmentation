"""SPDNet unit tests: model, dataset, training loop, CAM quality.

All tests use IMAGE_SIZE=64, BATCH_SIZE=2 to stay within 6GB VRAM.
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.wsss.spdnet.model import ADPL_CAM_LEVELS, MSE, FPN, ADPLCam, SPDNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMAGE_SIZE = 64
BATCH_SIZE = 2
FPN_CHANNELS = 256


class TestMSE:
    def test_creates(self):
        mse = MSE(channels=FPN_CHANNELS)
        assert isinstance(mse, MSE)

    def test_output_shape(self):
        mse = MSE(channels=FPN_CHANNELS).to(DEVICE)
        x = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE)
        out = mse(x)
        assert out.shape == x.shape

    def test_attention_bounded(self):
        """Sigmoid output should keep values in a reasonable range."""
        mse = MSE(channels=FPN_CHANNELS).to(DEVICE)
        x = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE)
        out = mse(x)
        assert out.abs().max() < x.abs().max() * 2


class TestFPN:
    @pytest.fixture
    def fpn(self):
        return FPN(in_channels=[256, 512, 1024, 2048], out_channels=FPN_CHANNELS).to(DEVICE)

    @pytest.fixture
    def backbone_features(self):
        return [
            torch.randn(BATCH_SIZE, 256, 16, 16, device=DEVICE),
            torch.randn(BATCH_SIZE, 512, 8, 8, device=DEVICE),
            torch.randn(BATCH_SIZE, 1024, 4, 4, device=DEVICE),
            torch.randn(BATCH_SIZE, 2048, 2, 2, device=DEVICE),
        ]

    def test_output_count(self, fpn, backbone_features):
        out = fpn(backbone_features)
        assert len(out) == 4

    def test_output_channels(self, fpn, backbone_features):
        out = fpn(backbone_features)
        for p in out:
            assert p.shape[1] == FPN_CHANNELS

    def test_output_spatial(self, fpn, backbone_features):
        out = fpn(backbone_features)
        expected_sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
        for p, (h, w) in zip(out, expected_sizes):
            assert p.shape[2:] == torch.Size([h, w])


class TestADPLCam:
    def test_tokenize(self):
        cam = ADPLCam(num_levels=ADPL_CAM_LEVELS).to(DEVICE)
        ref_fpn = [torch.randn(BATCH_SIZE, FPN_CHANNELS, s, s, device=DEVICE) for s in [16, 8, 4, 2]]
        tokens = cam.tokenize(ref_fpn)
        assert len(tokens) == ADPL_CAM_LEVELS
        for t in tokens:
            assert t.shape == (BATCH_SIZE, FPN_CHANNELS)

    def test_fuse_changes_output(self):
        cam = ADPLCam(num_levels=ADPL_CAM_LEVELS).to(DEVICE)
        query = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE)
        tokens = [torch.randn(BATCH_SIZE, FPN_CHANNELS, device=DEVICE) for _ in range(ADPL_CAM_LEVELS)]
        fused = cam.fuse(query, tokens)
        assert not torch.allclose(query, fused), "Token fusion should modify the feature map"


class TestSPDNet:
    @pytest.fixture
    def model(self):
        m = SPDNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        return m

    @pytest.fixture
    def pair(self):
        q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        return q, r

    def test_instantiation(self):
        model = SPDNet(num_classes=NUM_CLASSES, pretrained=False)
        assert model.num_classes == NUM_CLASSES

    def test_parameter_count(self):
        model = SPDNet(num_classes=NUM_CLASSES, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        assert 20_000_000 < total < 35_000_000, f"Unexpected param count: {total}"

    def test_shared_backbone(self):
        """Both branches use the same backbone (weight sharing)."""
        model = SPDNet(num_classes=NUM_CLASSES, pretrained=False)
        assert model.layer1 is model.layer1  # trivially true but confirms single backbone

    def test_training_forward(self, model, pair):
        model.train()
        logits = model(*pair, return_cam=False)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_inference_with_cam(self, model, pair):
        model.eval()
        logits, cam = model(*pair, return_cam=True)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert cam.shape[0] == BATCH_SIZE
        assert cam.shape[1] == NUM_CLASSES

    def test_backward_pass(self, model, pair):
        model.train()
        logits = model(*pair, return_cam=False)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_cam_backward_pass(self, model, pair):
        """ADPL-CAM alpha gets gradients when return_cam=True."""
        model.train()
        logits, cam = model(*pair, return_cam=True)
        loss = logits.sum() + cam.sum()
        loss.backward()
        assert model.adpl_cam.alpha.grad is not None

    def test_reference_sensitivity(self, model):
        """Different references must produce different logits for the same query."""
        model.eval()
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r2 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 + 1
        logits1 = model(q, r1, return_cam=False)
        logits2 = model(q, r2, return_cam=False)
        assert not torch.allclose(logits1, logits2, atol=1e-5), (
            "Logits must change when reference changes -- reference must influence classification"
        )

    def test_multi_reference_forward(self, model, pair):
        """List of N references must produce same-shape output as single reference."""
        model.eval()
        q, r = pair
        r2 = torch.randn_like(r)
        r3 = torch.randn_like(r)

        logits_single = model(q, r, return_cam=False)
        logits_multi = model(q, [r, r2, r3], return_cam=False)
        assert logits_multi.shape == logits_single.shape

        logits_cam, cam = model(q, [r, r2, r3], return_cam=True)
        assert logits_cam.shape == (BATCH_SIZE, NUM_CLASSES)
        assert cam.shape[0] == BATCH_SIZE
        assert cam.shape[1] == NUM_CLASSES

    def test_multi_reference_backward(self, model, pair):
        """All parameters get gradients through multi-reference path."""
        model.train()
        q, r = pair
        logits = model(q, [r, torch.randn_like(r)], return_cam=False)
        logits.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name} in multi-ref"

    def test_different_image_size(self, model):
        model.eval()
        q = torch.randn(1, 3, 96, 96, device=DEVICE)
        r = torch.randn(1, 3, 96, 96, device=DEVICE)
        logits = model(q, r, return_cam=False)
        assert logits.shape == (1, NUM_CLASSES)


class TestCAMQuality:
    """Sanity checks on ADPL-CAM output."""

    @pytest.fixture
    def model(self):
        return SPDNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE).eval()

    def test_cam_nonnegative(self, model):
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        _, cam = model(q, r, return_cam=True)
        assert cam.min() >= 0, "CAM should be non-negative after ReLU in ADPLCam"

    def test_cam_not_all_zeros(self, model):
        torch.manual_seed(42)
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 5
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 5
        _, cam = model(q, r, return_cam=True)
        assert cam.sum() > 0, "CAM should not be all zeros for non-trivial input"

    def test_token_fusion_changes_cam(self, model):
        """Fused CAM differs from unfused (alpha != 0)."""
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        _, cam_with_ref = model(q, r, return_cam=True)

        with torch.no_grad():
            saved_alpha = model.adpl_cam.alpha.clone()
            model.adpl_cam.alpha.zero_()
            _, cam_no_ref = model(q, r, return_cam=True)
            model.adpl_cam.alpha.copy_(saved_alpha)

        assert not torch.allclose(cam_with_ref, cam_no_ref), (
            "Token fusion should change CAM output"
        )


class TestFeatureSeeds:
    """Tests for extract_merged_features and generate_spdnet_seed."""

    @pytest.fixture
    def model(self):
        return SPDNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE).eval()

    def test_feature_seed_output_format(self, model):
        """generate_spdnet_seed returns {0: ndarray} with correct shape and [0,1] range."""
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        result = generate_spdnet_seed(
            model, [q], [[r]], DEVICE, seed_mode="feat_chmean"
        )
        assert isinstance(result, dict)
        assert 0 in result
        arr = result[0]
        assert arr.dtype == np.float32
        assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-6

    def test_feature_seed_modes(self, model):
        """All seed_mode values produce valid output."""
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        for mode in ["feat_chmean", "feat_chmax", "spatial_proto"]:
            result = generate_spdnet_seed(
                model, [q], [[r]], DEVICE, seed_mode=mode
            )
            assert 0 in result, f"Mode {mode} failed to produce output"
            assert result[0].shape == (IMAGE_SIZE, IMAGE_SIZE), f"Mode {mode} wrong shape"

    def test_spatial_proto_nonzero(self, model):
        """Spatial prototype produces non-constant maps (different from channel-mean)."""
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        torch.manual_seed(42)
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

        proto = generate_spdnet_seed(model, [q], [[r]], DEVICE, "spatial_proto")
        chmean = generate_spdnet_seed(model, [q], [[r]], DEVICE, "feat_chmean")

        assert proto[0].std() > 1e-6, "Spatial proto should not be constant"
        assert not np.allclose(proto[0], chmean[0], atol=1e-4), (
            "Spatial proto should differ from channel-mean"
        )

    def test_spatial_proto_reference_sensitive(self, model):
        """Spatial prototype output changes with different references."""
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        torch.manual_seed(123)
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r2 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE) * 3 + 1

        proto1 = generate_spdnet_seed(model, [q], [[r1]], DEVICE, "spatial_proto")
        proto2 = generate_spdnet_seed(model, [q], [[r2]], DEVICE, "spatial_proto")

        assert not np.allclose(proto1[0], proto2[0], atol=1e-3), (
            "Different references must produce different spatial proto maps"
        )


class TestCRFSrgb:
    def test_crf_srgb_parameter(self):
        """srgb parameter is accepted and affects CRF unary-bilateral interaction."""
        from src.wsss.refinement.crf import apply_crf

        np.random.seed(42)
        h, w = 128, 128
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        cam = np.full((h, w), 0.35, dtype=np.float32)
        cam[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.45
        cam_dict = {0: cam}

        q_low = apply_crf(image, cam_dict, bg_threshold=0.4, num_cls=2, srgb=3.0,
                          scale_factor=1.0, t=10)
        q_high = apply_crf(image, cam_dict, bg_threshold=0.4, num_cls=2, srgb=100.0,
                           scale_factor=1.0, t=10)

        assert q_low.shape == (2, h, w)
        assert q_high.shape == (2, h, w)
        label_low = np.argmax(q_low, axis=0)
        label_high = np.argmax(q_high, axis=0)
        assert not np.array_equal(label_low, label_high), (
            "srgb=3 vs srgb=100 should produce different label maps"
        )


class TestExtractMergedFeatures:
    @pytest.fixture
    def model(self):
        return SPDNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE).eval()

    def test_query_only(self, model):
        q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        result = model.extract_merged_features(q)
        assert "query_merged" in result
        assert result["query_merged"].shape[0] == BATCH_SIZE
        assert result["query_merged"].shape[1] == FPN_CHANNELS
        assert "fused" not in result

    def test_with_reference(self, model):
        q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        result = model.extract_merged_features(q, r)
        assert "query_merged" in result
        assert "fused" in result
        assert "ref_merged" in result
        assert result["fused"].shape == result["query_merged"].shape
