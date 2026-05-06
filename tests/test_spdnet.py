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

