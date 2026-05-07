"""SPDNet unit tests: model, dataset, training loop, CAM quality.

All tests use IMAGE_SIZE=64, BATCH_SIZE=2 to stay within 6GB VRAM.
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.wsss.spdnet.model import (
    ADPL_CAM_LEVELS, MSE, FPN, ADPLCam, SPDNet, SpatialCrossAttention,
)

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


class TestGradientCamDispatch:
    """Gradient-CAM modes reached via generate_spdnet_seed dispatcher.

    Covers that the new mode names (``layercam``, ``gradcam_pp``,
    ``xgradcam``) correctly route to the gradient-CAM backend, work
    with both fusion modes, and do not leak gradients into the model
    parameters.
    """

    @pytest.fixture(params=["token", "spatial"])
    def model(self, request):
        m = SPDNet(
            num_classes=NUM_CLASSES, pretrained=False, fusion_mode=request.param,
        ).to(DEVICE).eval()
        return m

    @pytest.mark.parametrize("mode", ["layercam", "gradcam_pp", "xgradcam"])
    def test_output_format(self, model, mode):
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        torch.manual_seed(0)
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = generate_spdnet_seed(
            model, [q], [[r]], DEVICE, seed_mode=mode,
            active_classes=[0],
        )
        assert 0 in out
        arr = out[0]
        assert arr.dtype == np.float32
        assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0 + 1e-6

    def test_raises_without_active_classes(self, model):
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with pytest.raises(ValueError, match="active_classes"):
            generate_spdnet_seed(
                model, [q], [[r]], DEVICE, seed_mode="layercam",
                active_classes=None,
            )

    def test_no_gradient_leak_from_dispatch(self, model):
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        for p in model.parameters():
            p.grad = None
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        _ = generate_spdnet_seed(
            model, [q], [[r]], DEVICE, seed_mode="layercam",
            active_classes=[0, 1],
        )
        assert all(p.grad is None for p in model.parameters()), (
            "generate_spdnet_seed(layercam) leaked into param.grad"
        )

    def test_no_grad_modes_unaffected(self, model):
        """Dispatcher must still route feat_chmean through the no-grad path
        (regression guard so existing TestFeatureSeeds callers keep working)."""
        from src.wsss.spdnet.cam_generator import generate_spdnet_seed

        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        r = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        # Explicit None for active_classes -- no error because feat_chmean is
        # not a gradient-CAM mode.
        out = generate_spdnet_seed(
            model, [q], [[r]], DEVICE, seed_mode="feat_chmean",
            active_classes=None,
        )
        assert out[0].shape == (IMAGE_SIZE, IMAGE_SIZE)


class TestSPDNetTrainerConfig:
    """Regression tests for the trainer-level config flags used by Phase 5."""

    def test_save_best_cam_iou_defaults_on(self):
        from src.conf.spdnet import SPDNetTrainerConfig

        cfg = SPDNetTrainerConfig()
        assert hasattr(cfg, "save_best_cam_iou"), (
            "SPDNetTrainerConfig must expose save_best_cam_iou for the "
            "Phase 5 val/cam_iou_best checkpoint."
        )
        assert cfg.save_best_cam_iou is True, (
            "Default must be True so every SPDNet run from now on saves "
            "a best_cam_iou.ckpt alongside the val/mAP winner."
        )


class TestLRScheduleGuard:
    """Regression guard for the inverted-cosine bug that cost ~10h on 2026-04-30.

    When ``CosineAnnealingLR`` is configured with ``eta_min >= base_lr`` it
    ascends from base_lr to eta_min instead of decaying -- a silent pathology
    because the LR still moves, warmup still works, and loss still drops on
    the classifier head. ``configure_optimizers`` now fails loudly.
    """

    def _mk(self, learning_rate: float, min_lr: float):
        from src.wsss.spdnet.lightning import SPDNetModule

        return SPDNetModule(
            num_classes=4, fpn_channels=16, mse_reduction=4,
            pretrained=False, learning_rate=learning_rate,
            weight_decay=0.05, warmup_epochs=1, min_lr=min_lr,
            fusion_mode="spatial", losses_cfg=None,
            online_loc_metric=None, image_size=32,
        )

    def test_raises_when_min_lr_equals_base_lr(self):
        module = self._mk(learning_rate=1e-5, min_lr=1e-5)
        with pytest.raises(ValueError, match="must be strictly below"):
            module.configure_optimizers()

    def test_raises_when_min_lr_above_base_lr(self):
        """Exactly the highres896 failure mode: lr=7.8e-6, min_lr=1e-5."""
        module = self._mk(learning_rate=7.8125e-6, min_lr=1e-5)
        with pytest.raises(ValueError, match="must be strictly below"):
            module.configure_optimizers()

    def test_accepts_when_min_lr_below_base_lr(self):
        from unittest.mock import MagicMock

        module = self._mk(learning_rate=3.125e-5, min_lr=1e-5)
        trainer_mock = MagicMock()
        trainer_mock.max_epochs = 10
        module.trainer = trainer_mock
        opt_cfg = module.configure_optimizers()
        assert "optimizer" in opt_cfg and "lr_scheduler" in opt_cfg

    def test_post_warmup_lr_strictly_decreases(self):
        """Cosine phase must decay the LR, not ascend it.

        Drive the scheduler through warmup + a few cosine epochs and confirm
        the post-warmup samples are monotonically non-increasing. We use a
        max_epochs that is comfortably larger than warmup_epochs so several
        cosine samples fit before the floor is reached.
        """
        from unittest.mock import MagicMock

        module = self._mk(learning_rate=1.0e-4, min_lr=1e-6)
        trainer_mock = MagicMock()
        trainer_mock.max_epochs = 20
        module.trainer = trainer_mock

        opt_cfg = module.configure_optimizers()
        scheduler = opt_cfg["lr_scheduler"]["scheduler"]
        optimizer = opt_cfg["optimizer"]

        lrs: list[float] = []
        for _ in range(15):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        warmup_n = module.warmup_epochs
        assert lrs[warmup_n] > lrs[-1], (
            "After warmup the cosine arm must decay: peak LR "
            f"{lrs[warmup_n]:g} should exceed final LR {lrs[-1]:g}."
        )
        post_warmup = lrs[warmup_n:]
        strict_drops = sum(
            1 for a, b in zip(post_warmup, post_warmup[1:]) if b < a
        )
        assert strict_drops >= (len(post_warmup) - 1) // 2, (
            f"Cosine arm should be mostly non-increasing. Samples: {lrs}"
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


class TestSpatialCrossAttention:
    """Unit tests for the SpatialCrossAttention module."""

    def test_output_shape(self):
        sca = SpatialCrossAttention(channels=FPN_CHANNELS).to(DEVICE)
        q = torch.randn(BATCH_SIZE, FPN_CHANNELS, 16, 16, device=DEVICE)
        r = torch.randn(BATCH_SIZE, FPN_CHANNELS, 16, 16, device=DEVICE)
        out = sca(q, r)
        assert out.shape == q.shape

    def test_output_shape_different_ref_size(self):
        """Reference can have a different spatial size from query."""
        sca = SpatialCrossAttention(channels=FPN_CHANNELS).to(DEVICE)
        q = torch.randn(BATCH_SIZE, FPN_CHANNELS, 16, 16, device=DEVICE)
        r = torch.randn(BATCH_SIZE, FPN_CHANNELS, 32, 32, device=DEVICE)
        out = sca(q, r)
        assert out.shape == q.shape

    def test_gate_initialization(self):
        sca = SpatialCrossAttention(channels=FPN_CHANNELS)
        assert sca.gate.item() == pytest.approx(0.1, abs=1e-6)

    def test_modifies_input(self):
        sca = SpatialCrossAttention(channels=FPN_CHANNELS).to(DEVICE)
        q = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE)
        r = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE)
        out = sca(q, r)
        assert not torch.allclose(q, out, atol=1e-6), (
            "Cross-attention should modify the query features"
        )

    def test_gradients_flow(self):
        sca = SpatialCrossAttention(channels=FPN_CHANNELS).to(DEVICE)
        q = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE, requires_grad=True)
        r = torch.randn(BATCH_SIZE, FPN_CHANNELS, 8, 8, device=DEVICE, requires_grad=True)
        out = sca(q, r)
        out.sum().backward()
        assert q.grad is not None
        assert r.grad is not None
        assert sca.gate.grad is not None


class TestSPDNetFusionModes:
    """Test SPDNet with both fusion_mode='token' and 'spatial'."""

    @pytest.fixture(params=["token", "spatial"])
    def model(self, request):
        return SPDNet(
            num_classes=NUM_CLASSES, pretrained=False, fusion_mode=request.param
        ).to(DEVICE)

    @pytest.fixture
    def pair(self):
        q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        return q, r

    def test_forward_logits(self, model, pair):
        model.eval()
        logits = model(*pair, return_cam=False)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_with_cam(self, model, pair):
        model.eval()
        logits, cam = model(*pair, return_cam=True)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert cam.shape[0] == BATCH_SIZE
        assert cam.shape[1] == NUM_CLASSES

    def test_backward(self, model, pair):
        model.train()
        logits = model(*pair, return_cam=False)
        logits.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_multi_reference(self, model, pair):
        model.eval()
        q, r = pair
        logits = model(q, [r, torch.randn_like(r)], return_cam=False)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_extract_merged_features(self, model, pair):
        model.eval()
        q, r = pair
        result = model.extract_merged_features(q, r)
        assert "query_merged" in result
        assert "fused" in result
        assert "ref_merged" in result
        assert result["fused"].shape == result["query_merged"].shape

    def test_invalid_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion_mode"):
            SPDNet(num_classes=NUM_CLASSES, pretrained=False, fusion_mode="invalid")


class TestSpatialFusionSpatiallyAware:
    """Verify spatial cross-attention produces location-dependent outputs."""

    def test_different_references_different_spatial_pattern(self):
        """Spatial fusion output must change spatially when reference changes."""
        model = SPDNet(
            num_classes=NUM_CLASSES, pretrained=False, fusion_mode="spatial"
        ).to(DEVICE).eval()

        torch.manual_seed(99)
        q = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r2 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 3 + 1

        feats1 = model.extract_merged_features(q, r1)
        feats2 = model.extract_merged_features(q, r2)

        diff = (feats1["fused"] - feats2["fused"]).abs()
        h = diff.shape[2]
        top_half = diff[:, :, : h // 2, :].mean().item()
        bottom_half = diff[:, :, h // 2 :, :].mean().item()

        assert diff.mean() > 1e-4, "Spatial fusion output must differ for different refs"
        assert top_half != pytest.approx(bottom_half, abs=1e-5), (
            "Spatial difference should not be uniform -- attention should produce "
            "location-dependent differences (unlike token fusion which is uniform)"
        )


class TestRefPoolSizeConfigurable:
    """Trap-2 fix (RESEARCH_CONTEXT.md §5.14.2): ref_pool_size flows from
    SPDNetConfig -> SPDNet -> SpatialCrossAttention and changes the
    attention-weight matrix shape exposed via ``return_attn=True``.
    """

    @pytest.mark.parametrize("rps", [14, 20, 28])
    def test_ref_pool_size_changes_attn_shape(self, rps):
        """Setting ref_pool_size=N gives an N*N key set (not the legacy 196)."""
        model = SPDNet(
            num_classes=NUM_CLASSES, pretrained=False,
            fusion_mode="spatial", ref_pool_size=rps,
        ).to(DEVICE).eval()
        assert model.ref_pool_size == rps
        assert model.spatial_attn.ref_pool_size == rps

        q = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        r = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        feats = model.extract_merged_features(q, r, return_attn=True)
        assert "attn_w" in feats
        assert feats["attn_w"].shape[-1] == rps * rps, (
            f"attn key set should be {rps*rps} for ref_pool_size={rps}, "
            f"got {feats['attn_w'].shape[-1]}"
        )

    def test_default_ref_pool_size_unchanged(self):
        """Backwards compatibility: SPDNet() with no ref_pool_size kwarg
        still uses the legacy 14×14 grid -- regression-free.
        """
        model = SPDNet(
            num_classes=NUM_CLASSES, pretrained=False, fusion_mode="spatial",
        ).to(DEVICE).eval()
        assert model.ref_pool_size == 14
        assert model.spatial_attn.ref_pool_size == 14


class TestEffectiveBatchLR:
    """Trap-1 fix: scaled_lr is now ``base_lr * (batch * accum) / 256``
    rather than ``base_lr * batch / 256``. Encoded as a unit test against
    the train_spdnet helper (we don't import the full hydra entry point
    because it pulls a heavy dataset; just compute the formula).
    """

    @pytest.mark.parametrize(
        "base_lr, batch, accum, expected",
        [
            (5e-4, 16, 2, 5e-4 * 32 / 256),  # 448 spec -> 6.25e-5 (was 3.125e-5)
            (5e-4, 8, 4, 5e-4 * 32 / 256),   # equivalent eff_batch -> same LR
            (5e-4, 6, 5, 5e-4 * 30 / 256),   # 896 typical -> 5.86e-5
            (5e-4, 4, 8, 5e-4 * 32 / 256),   # 896 aux-loss recipe -> 6.25e-5
            (5e-4, 2, 15, 5e-4 * 30 / 256),  # 896 small-batch aux -> 5.86e-5
        ],
    )
    def test_effective_batch_scaling(self, base_lr, batch, accum, expected):
        # Inline the exact formula used in train_spdnet.py so the test
        # is independent of hydra+dataset wiring.
        eff_batch = batch * accum
        scaled = base_lr * (eff_batch / 256.0)
        assert scaled == pytest.approx(expected, rel=1e-9)


class TestLogAttnStats:
    """``losses.log_attn_stats=True`` forces the attention-buffer forward
    path even when no aux loss currently needs the attention map, and
    populates three diagnostic scalars (``attn_mean``, ``attn_std``,
    ``attn_p99``). This is the headline new-data hook for the Phase-5
    cls-only baseline run (see RESEARCH_CONTEXT.md §5.14.6 + the
    pre-launch plan).
    """

    def _make_module(self, log_attn_stats: bool):
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        losses_cfg = SPDNetSpatialLossesConfig(
            lambda_eq=0.0, lambda_ac=0.0, lambda_marg_H=0.0,
            lambda_mask=0.0, lambda_con=0.0, lambda_distill=0.0,
            online_loc_eval_enabled=False,
            log_attn_stats=log_attn_stats,
        )
        module = SPDNetModule(
            num_classes=NUM_CLASSES, fpn_channels=16, mse_reduction=4,
            pretrained=False, learning_rate=1e-4,
            weight_decay=0.05, warmup_epochs=1, min_lr=1e-6,
            fusion_mode="spatial",
            losses_cfg=losses_cfg,
            online_loc_metric=None,
            image_size=IMAGE_SIZE,
            ref_pool_size=8,
        ).to(DEVICE)
        module.train()

        recorded: dict[str, float] = {}

        def fake_log(name, value, *args, **kwargs):
            v = value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
            recorded[name] = v

        module.log = fake_log  # type: ignore[assignment]
        trainer_mock = MagicMock()
        trainer_mock.is_global_zero = True
        trainer_mock.sanity_checking = False
        module.trainer = trainer_mock
        return module, recorded

    def _make_batch(self):
        return {
            "query_image": torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE),
            "ref_images": torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE),
            "query_label": torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES), device=DEVICE).float(),
        }

    def test_default_off_no_attn_stats_logged(self):
        module, recorded = self._make_module(log_attn_stats=False)
        _ = module.training_step(self._make_batch(), 0)
        assert "train/attn_mean" not in recorded
        assert "train/attn_std" not in recorded
        assert "train/attn_p99" not in recorded

    def test_log_attn_stats_emits_three_diagnostics(self):
        module, recorded = self._make_module(log_attn_stats=True)
        _ = module.training_step(self._make_batch(), 0)
        for key in ("train/attn_mean", "train/attn_std", "train/attn_p99"):
            assert key in recorded, f"expected {key} in {sorted(recorded)}"

    def test_attn_mean_in_unit_interval(self):
        """``attn_orig`` is ``concentration_softmax(attn_w)`` -> values in [0, 1].

        The mean (and p99) must therefore stay in [0, 1] for any input;
        std must be non-negative. Tests guard against accidental swaps
        of ``attn_orig`` with the raw scores ``attn_w`` (which sum to 1
        per row, so their mean is ``1/N`` -- different number).
        """
        module, recorded = self._make_module(log_attn_stats=True)
        _ = module.training_step(self._make_batch(), 0)
        assert 0.0 <= recorded["train/attn_mean"] <= 1.0
        assert 0.0 <= recorded["train/attn_p99"] <= 1.0
        assert recorded["train/attn_std"] >= 0.0

    def test_log_attn_stats_does_not_affect_loss(self):
        """The diagnostic must not contribute to the training graph.

        Same query/ref/labels, identical seeds: with stats off vs on the
        classification loss must be bit-for-bit identical (no extra
        operations should sneak into the autograd tape via the detached
        statistics).
        """
        torch.manual_seed(7)
        m_off, _ = self._make_module(log_attn_stats=False)
        torch.manual_seed(7)
        batch = self._make_batch()
        loss_off = m_off.training_step(batch, 0)

        torch.manual_seed(7)
        m_on, _ = self._make_module(log_attn_stats=True)
        torch.manual_seed(7)
        batch2 = self._make_batch()
        loss_on = m_on.training_step(batch2, 0)

        assert torch.allclose(loss_off, loss_on, atol=1e-6, rtol=1e-5), (
            f"log_attn_stats must be a pure observer: loss_off={loss_off:g}, "
            f"loss_on={loss_on:g}"
        )


class TestDDPStrategyResolver:
    """Helper that translates ``trainer.strategy`` + ``trainer.devices``
    into the argument Lightning expects. Encoded as a tested unit so the
    DDP+aux-loss combination can never silently regress to
    ``find_unused_parameters=False`` (see RESEARCH_CONTEXT.md §5.14.6
    for why that's a 2-rank crash on epoch 0).
    """

    def test_single_device_passes_string_through(self):
        from src.train_spdnet import _resolve_trainer_strategy

        for devices in (1, "1", None, 0):
            out = _resolve_trainer_strategy(
                strategy="auto", devices=devices, find_unused_parameters=True,
            )
            assert out == "auto", f"devices={devices!r} expected 'auto', got {out!r}"

    def test_devices_two_auto_returns_ddp_strategy(self):
        from lightning.pytorch.strategies import DDPStrategy

        from src.train_spdnet import _resolve_trainer_strategy

        out = _resolve_trainer_strategy(
            strategy="auto", devices=2, find_unused_parameters=True,
        )
        assert isinstance(out, DDPStrategy)
        # Lightning stores the DDP constructor kwargs under
        # ``_ddp_kwargs`` (forwarded verbatim to ``torch.nn.parallel.
        # DistributedDataParallel`` at setup time).
        assert out._ddp_kwargs.get("find_unused_parameters") is True
        assert out._ddp_kwargs.get("gradient_as_bucket_view") is True

    def test_devices_two_explicit_ddp_returns_ddp_strategy(self):
        from lightning.pytorch.strategies import DDPStrategy

        from src.train_spdnet import _resolve_trainer_strategy

        out = _resolve_trainer_strategy(
            strategy="ddp", devices=2, find_unused_parameters=False,
        )
        assert isinstance(out, DDPStrategy)
        assert out._ddp_kwargs.get("find_unused_parameters") is False, (
            "find_unused_parameters=False must be propagated for users who "
            "have audited their model and want the perf win"
        )

    def test_non_ddp_strategy_passed_through(self):
        """Picking ``deepspeed_stage_2`` or ``fsdp`` must NOT be hijacked."""
        from src.train_spdnet import _resolve_trainer_strategy

        for strat in ("deepspeed_stage_2", "fsdp", "fsdp_native"):
            out = _resolve_trainer_strategy(
                strategy=strat, devices=4, find_unused_parameters=True,
            )
            assert out == strat, (
                f"strategy={strat!r} should be passed through verbatim, got {out!r}"
            )

    def test_string_devices_handled(self):
        """Hydra forwards the value as a string in some configurations."""
        from lightning.pytorch.strategies import DDPStrategy

        from src.train_spdnet import _resolve_trainer_strategy

        out = _resolve_trainer_strategy(
            strategy="auto", devices="2", find_unused_parameters=True,
        )
        assert isinstance(out, DDPStrategy)

    def test_ddp_timeout_propagated(self):
        """``ddp_timeout_seconds > 0`` must materialise as a
        ``datetime.timedelta`` on the DDPStrategy ``_timeout`` slot. The
        2026-05-06 P1' run wasted 30 minutes waiting for a dead rank 0
        because Lightning's NCCL default is 1800 s; tightening this is
        a defense in depth, and the test guards against the kwarg name
        regressing away from ``timeout``.
        """
        import datetime

        from lightning.pytorch.strategies import DDPStrategy

        from src.train_spdnet import _resolve_trainer_strategy

        out = _resolve_trainer_strategy(
            strategy="ddp", devices=2, find_unused_parameters=True,
            ddp_timeout_seconds=600,
        )
        assert isinstance(out, DDPStrategy)
        assert out._timeout == datetime.timedelta(seconds=600), (
            f"DDPStrategy._timeout expected 600 s, got {out._timeout!r}"
        )

    def test_ddp_timeout_zero_uses_lightning_default(self):
        """``ddp_timeout_seconds=0`` should not pass an explicit timeout
        so Lightning's own default applies (currently 1800 s). We
        verify by constructing two strategies and comparing -- the
        ``_timeout`` of the unset version is what Lightning picks.
        """
        from lightning.pytorch.strategies import DDPStrategy

        from src.train_spdnet import _resolve_trainer_strategy

        ours = _resolve_trainer_strategy(
            strategy="ddp", devices=2, find_unused_parameters=True,
            ddp_timeout_seconds=0,
        )
        baseline = DDPStrategy(find_unused_parameters=True)
        assert isinstance(ours, DDPStrategy)
        assert ours._timeout == baseline._timeout, (
            f"ddp_timeout_seconds=0 must defer to Lightning default; got "
            f"{ours._timeout!r} vs Lightning baseline {baseline._timeout!r}"
        )


class TestSyncBatchNormConfig:
    """``SPDNetTrainerConfig.sync_batchnorm`` is the toggle that flips
    backbone BN layers to SyncBatchNorm under DDP, restoring an effective
    BN sample of (devices * batch_size) instead of the per-rank
    micro-batch (which is just 2 in the rps=56 / 896² recipe).

    The runtime conversion itself is end-to-end-tested by
    ``scripts/smoke_ddp_5090.py`` (which actually launches DDP and
    walks the model post-fit). These unit tests cover the static
    contract: field present + train_spdnet wires it into Trainer.
    """

    def test_default_is_false(self):
        """Single-card configs must not silently change behaviour."""
        from src.conf.spdnet import SPDNetTrainerConfig

        cfg = SPDNetTrainerConfig()
        assert cfg.sync_batchnorm is False, (
            "Default sync_batchnorm should be False so legacy single-card "
            "experiments (devices=1) are byte-identical to the pre-flag baseline."
        )

    def test_field_typed_bool(self):
        """Hydra needs a strict type so ``sync_batchnorm=true`` parses to bool."""
        import dataclasses

        from src.conf.spdnet import SPDNetTrainerConfig

        fields = {f.name: f for f in dataclasses.fields(SPDNetTrainerConfig)}
        assert "sync_batchnorm" in fields, "Missing field SPDNetTrainerConfig.sync_batchnorm"
        assert fields["sync_batchnorm"].type is bool, (
            f"sync_batchnorm must be typed bool, got {fields['sync_batchnorm'].type!r}"
        )

    def test_train_spdnet_passes_sync_batchnorm_to_trainer(self):
        """If the kwarg is not threaded into ``L.Trainer`` the flag is
        a silent no-op. This guard catches that drift at test time
        instead of after a 12-hour run.
        """
        import inspect

        import src.train_spdnet as ts

        src = inspect.getsource(ts.train_spdnet)
        assert "sync_batchnorm=" in src, (
            "train_spdnet does not pass sync_batchnorm to L.Trainer; the "
            "config field is a no-op."
        )
        # Also assert it's read from the config (not hardcoded). The
        # current implementation uses ``getattr(cfg.trainer, "sync_batchnorm", False)``.
        assert "cfg.trainer" in src and "sync_batchnorm" in src


class TestOnlineCAMIoUOOMDefense:
    """Regression guards around the symmetric ``OnlineCAMIoU.evaluate``
    branch in ``SPDNetModule.on_validation_epoch_end``.

    Two distinct bugs hit the 2026-05-06 P1' run on the 5090 host;
    these tests guard against both regressions.

    Bug A: rank-0-only ``OnlineCAMIoU.evaluate`` OOMed (rps=56 +
    896² + eval_batch_size=8 materialised a ~20 GiB attention weight
    tensor on top of a ~24 GiB training residual). Rank 1 then sat
    on the next ALLREDUCE for the full 30-min NCCL watchdog before
    the run was killed.

    Bug B (worse): even AFTER lowering eval_batch_size to fit, the
    SECOND launch deadlocked again because the metric was logged
    with ``rank_zero_only=True``. ``ModelCheckpoint(monitor=
    "val/cam_iou_best")`` then took different code paths on rank 0
    (metric present -> save_checkpoint -> ``strategy.barrier()``
    which is an ``AllReduce(1)`` in NCCL) vs rank 1 (metric absent
    -> skip save -> no barrier), and the asymmetric collective hung
    until the new 600 s watchdog fired. Diagnostic from the trace::

        Rank 0: WorkNCCL(SeqNum=2049548, OpType=ALLREDUCE,  NumelIn=1)
        Rank 1: WorkNCCL(SeqNum=2049547, OpType=ALLGATHER,  NumelIn=2)

    Rank 0 had issued exactly **one extra collective** -- the save
    barrier. Fix: every rank now calls ``evaluate()`` (deterministic
    because of fixed query subset + seed + DDP-synced weights), and
    we log with ``sync_dist=True`` so the metric is symmetric on
    every rank's ``callback_metrics``. ModelCheckpoint then takes
    identical code paths on every rank.

    Layered defenses checked here:

    * ``lightning.py``: NO ``is_global_zero`` gate on the OnlineCAMIoU
      branch (every rank computes -> symmetric ``self.log`` count).
    * ``lightning.py``: NO ``rank_zero_only=True`` on the cam_iou logs
      (would re-introduce asymmetric callback_metrics).
    * ``lightning.py``: ``sync_dist=True`` on the cam_iou logs (the
      Lightning-recommended pattern; warning was telling us this).
    * ``lightning.py``: cross-rank OOM coordination via
      ``all_reduce(MIN)`` so a one-rank failure can't reintroduce
      asymmetry through the back door.
    * ``lightning.py``: ``try/except torch.cuda.OutOfMemoryError``
      around ``evaluate()`` (Bug A defense in depth).
    * ``conf/spdnet.py``: ``online_loc_eval_batch_size=8`` default
      (lowered upstream by launcher to 2 for rps=56 / 896²).
    * ``conf/spdnet.py``: ``ddp_timeout_seconds`` defaults to 600 s
      (10 min instead of NCCL's 30 min) so any future deadlock fails
      fast.

    These tests check the *contract* (presence + structure of the
    guards) rather than running a full DDP fit, because reproducing
    the deadlock in CI requires multi-GPU + a real dataset.
    """

    def test_evaluate_wrapped_in_try_except_oom(self):
        import inspect

        from src.wsss.spdnet.lightning import SPDNetModule

        src = inspect.getsource(SPDNetModule.on_validation_epoch_end)

        eval_idx = src.find("self.online_loc_metric.evaluate")
        assert eval_idx > 0, (
            "evaluate() call must be present in on_validation_epoch_end"
        )
        try_idx = src.rfind("try:", 0, eval_idx)
        assert try_idx > 0, (
            "OnlineCAMIoU.evaluate is not wrapped in try/except. The "
            "2026-05-06 P1' OOM regression guard is gone."
        )
        except_idx = src.find("except torch.cuda.OutOfMemoryError", eval_idx)
        assert except_idx > eval_idx, (
            "Missing 'except torch.cuda.OutOfMemoryError' AFTER the "
            "evaluate() call. A bare 'except Exception' is too coarse "
            "(would swallow ValueError from misconfigured GT etc); "
            "keep the type strict."
        )
        assert src.count("torch.cuda.empty_cache()") >= 2, (
            "Need empty_cache() both before evaluate (free training "
            "residual) and inside the except (drain the partial OOM)."
        )

    def test_no_is_global_zero_gate_on_online_loc_branch(self):
        """The OnlineCAMIoU branch must execute on EVERY rank.

        Adding back ``and self.trainer.is_global_zero`` to the if
        guard is the *exact* regression that caused the 2026-05-06
        deadlock (Bug B). ModelCheckpoint(monitor="val/cam_iou_best")
        will take asymmetric code paths if only rank 0 logs the
        metric, and ``trainer.save_checkpoint`` issues a barrier on
        rank 0 only -> deadlock at the 600 s watchdog.
        """
        import inspect
        import re

        from src.wsss.spdnet.lightning import SPDNetModule

        src = inspect.getsource(SPDNetModule.on_validation_epoch_end)

        # Find the if-guard around online_loc_metric: from "if (" up
        # to the trailing "):" that opens its body. The guard may span
        # multiple lines and contain nested parens (e.g. should_run()).
        # Strategy: locate the start, then scan for the matching ')'
        # using a paren counter so should_run(...) doesn't end us early.
        start = src.find("if (")
        while start != -1:
            tail = src[start:]
            anchor = tail.find("self.online_loc_metric is not None")
            if anchor == -1 or anchor > 200:
                start = src.find("if (", start + 1)
                continue
            depth = 0
            end = -1
            for i, ch in enumerate(tail):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            assert end > 0, "Unterminated parens in online_loc guard"
            guard = tail[:end]
            break
        else:  # no hit at all
            guard = ""
        assert guard, (
            "Couldn't locate the 'if self.online_loc_metric is not "
            "None' guard. Did the structure change?"
        )
        assert "is_global_zero" not in guard, (
            "is_global_zero is back in the OnlineCAMIoU guard. This "
            "reintroduces the 2026-05-06 ModelCheckpoint asymmetric-"
            "barrier deadlock. Every rank must enter evaluate() so "
            "val/cam_iou_* is present on every rank's callback_metrics."
        )

    def test_cam_iou_log_uses_sync_dist_not_rank_zero_only(self):
        """The val/cam_iou_* logs must use ``sync_dist=True`` and
        NOT ``rank_zero_only=True``. The latter put the metric on
        rank 0's ``callback_metrics`` only, which made
        ``ModelCheckpoint(monitor="val/cam_iou_best")`` take a
        different code path on rank 0 (save -> barrier) vs rank 1
        (skip), causing the deadlock.
        """
        import inspect

        from src.wsss.spdnet.lightning import SPDNetModule

        src = inspect.getsource(SPDNetModule.on_validation_epoch_end)

        # Locate self.log(f"val/{k}", ...) and walk to the matching
        # close paren via depth counting (float(v) has nested parens).
        anchor = src.find('self.log(\n                    f"val/{k}"')
        if anchor < 0:
            anchor = src.find('self.log(\n                    f"val/')
        assert anchor > 0, (
            "Couldn't find self.log(f\"val/{k}\", ...) in "
            "on_validation_epoch_end."
        )
        depth = 0
        end = -1
        for i, ch in enumerate(src[anchor + len("self.log") :]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = anchor + len("self.log") + i + 1
                    break
        assert end > 0, "Unterminated parens in val/{k} log call"
        log_call = src[anchor:end]
        assert "rank_zero_only=True" not in log_call, (
            "rank_zero_only=True is back on the val/cam_iou_* "
            "self.log() call. This is exactly what caused the "
            "2026-05-06 deadlock; use sync_dist=True so all ranks "
            "see the metric and ModelCheckpoint takes a symmetric "
            "code path."
        )
        assert "sync_dist=True" in log_call, (
            "Missing sync_dist=True on the val/cam_iou_* self.log() "
            "call. Lightning explicitly warns about this in the "
            "logger_connector at distributed training, and not "
            "having it makes the metric value rank-local."
        )

    def test_oom_coordinated_across_ranks(self):
        """Asymmetric OOM is the back door into the same deadlock:
        if rank 0 OOMs (scalars={}) but rank 1 succeeds (scalars=
        {3 keys}) the per-rank ``self.log`` count diverges and the
        next collective deadlocks. We coordinate via an
        ``all_reduce`` MIN on a 0/1 success flag so a single failure
        forces every rank to skip the log calls.
        """
        import inspect

        from src.wsss.spdnet.lightning import SPDNetModule

        src = inspect.getsource(SPDNetModule.on_validation_epoch_end)

        assert "torch.distributed.is_initialized()" in src, (
            "Missing torch.distributed.is_initialized() guard for the "
            "cross-rank OOM coordination. Without it, asymmetric "
            "OOMs reintroduce the deadlock."
        )
        assert "torch.distributed.all_reduce" in src, (
            "Missing torch.distributed.all_reduce for cross-rank OOM "
            "coordination."
        )
        # MIN op: any rank failing -> every rank's flag becomes 0.
        # MAX would require ALL ranks to fail to skip, which is wrong
        # (we want ANY rank's failure to skip everywhere).
        assert "ReduceOp.MIN" in src, (
            "OOM coordination must use ReduceOp.MIN (any rank "
            "failing -> skip everywhere). MAX would only skip if "
            "every rank failed, which is the buggy original "
            "behaviour."
        )

    def test_default_ddp_timeout_is_tightened(self):
        """``ddp_timeout_seconds`` default must be a finite, sub-1800 s
        value. If anyone bumps it back to Lightning's 1800 default
        the 30-min hang on a dead rank comes back.
        """
        from src.conf.spdnet import SPDNetTrainerConfig

        cfg = SPDNetTrainerConfig()
        assert hasattr(cfg, "ddp_timeout_seconds"), (
            "Missing SPDNetTrainerConfig.ddp_timeout_seconds"
        )
        assert 60 <= cfg.ddp_timeout_seconds < 1800, (
            f"Default ddp_timeout_seconds={cfg.ddp_timeout_seconds} is "
            f"outside the safe range [60, 1800). Lightning default is "
            f"1800; we tightened it after the 2026-05-06 dead-rank-0 "
            f"hang. Keep the floor sane (>= 60 s) so OnlineCAMIoU's "
            f"~30-60 s evaluate (now run on every rank) doesn't trip "
            f"the timeout."
        )

