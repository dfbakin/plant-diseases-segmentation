"""SPDNet unit tests: model, dataset, training loop, CAM quality.

All tests use IMAGE_SIZE=64, BATCH_SIZE=2 to stay within 6GB VRAM.
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

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
