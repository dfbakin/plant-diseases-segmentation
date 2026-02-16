"""Test gate 1.1: MCTformer model instantiation, forward pass, and checkpoint loading.

All tests use image_size=64, batch_size=2 to stay within 6GB VRAM.
"""

from pathlib import Path

import pytest
import torch

from src.wsss.mctformer.model import MCTformerPlus, create_mctformer_v2

PRETRAINED_PATH = Path("pretrained/MCTformerV2.pth")
HAS_CHECKPOINT = PRETRAINED_PATH.exists()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 20
IMAGE_SIZE = 64
BATCH_SIZE = 2


class TestMCTformerInstantiation:
    """Test that models can be created without errors."""

    def test_mctformer_plus_default(self):
        """MCTformerPlus with default VOC params."""
        model = MCTformerPlus(
            input_size=IMAGE_SIZE,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=NUM_CLASSES,
        )
        assert isinstance(model, MCTformerPlus)
        assert model.num_classes == NUM_CLASSES
        assert model.embed_dim == 384

    def test_create_mctformer_v2_factory(self):
        """Factory function creates a valid model."""
        model = create_mctformer_v2(
            num_classes=NUM_CLASSES,
            input_size=IMAGE_SIZE,
        )
        assert isinstance(model, MCTformerPlus)
        assert model.num_classes == NUM_CLASSES

    def test_parameter_count(self):
        """Check that model has reasonable parameter count (~22M for DeiT-small)."""
        model = create_mctformer_v2(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE)
        total_params = sum(p.numel() for p in model.parameters())
        # DeiT-small ~22M params, MCTformer adds class tokens + conv head
        assert 20_000_000 < total_params < 30_000_000, f"Unexpected param count: {total_params}"


class TestMCTformerForward:
    """Test forward pass with synthetic data."""

    @pytest.fixture
    def model(self):
        m = create_mctformer_v2(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE)
        m = m.to(DEVICE)
        return m

    @pytest.fixture
    def dummy_input(self):
        return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    def test_training_forward(self, model, dummy_input):
        """Training mode returns list of [cls_logits, cls_embeddings, patch_logits]."""
        model.train()
        output = model(dummy_input)
        assert isinstance(output, list)
        assert len(output) == 3

        cls_logits, all_cls_embeddings, patch_logits = output
        assert cls_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert patch_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        # all_cls_embeddings: (depth, B, num_classes, embed_dim)
        assert all_cls_embeddings.shape[0] == 12  # depth
        assert all_cls_embeddings.shape[1] == BATCH_SIZE
        assert all_cls_embeddings.shape[2] == NUM_CLASSES

    def test_inference_forward_no_att(self, model, dummy_input):
        """Inference mode without attention returns same as training."""
        model.eval()
        output = model(dummy_input, return_att=False)
        assert isinstance(output, list)
        assert len(output) == 3

    def test_inference_forward_with_att(self, model, dummy_input):
        """Inference with return_att=True returns (logits, cams, patch_attn)."""
        model.eval()
        logits, cams, patch_attn = model(dummy_input, return_att=True)

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

        # CAMs: (B, num_classes, feat_h, feat_w)
        feat_size = IMAGE_SIZE // 16  # patch_size=16
        assert cams.shape == (BATCH_SIZE, NUM_CLASSES, feat_size, feat_size)

        # patch_attn: (depth, B, num_patches, num_patches)
        num_patches = feat_size * feat_size
        assert patch_attn.shape[2] == num_patches
        assert patch_attn.shape[3] == num_patches

    @pytest.mark.parametrize("attention_type", ["fused", "patchcam", "mct"])
    def test_attention_types(self, model, dummy_input, attention_type):
        """All attention types produce valid CAMs."""
        model.eval()
        logits, cams, _ = model(dummy_input, return_att=True, attention_type=attention_type)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert cams.shape[0] == BATCH_SIZE
        assert cams.shape[1] == NUM_CLASSES

    def test_different_image_size(self, model):
        """Model handles image sizes different from input_size (pos embed interpolation)."""
        model.eval()
        # 96x96 instead of 64x64
        x = torch.randn(1, 3, 96, 96, device=DEVICE)
        output = model(x, return_att=False)
        assert isinstance(output, list)
        cls_logits = output[0]
        assert cls_logits.shape == (1, NUM_CLASSES)

    def test_non_square_image(self, model):
        """Model handles non-square images (pos embed interpolation)."""
        model.eval()
        x = torch.randn(1, 3, 64, 96, device=DEVICE)
        output = model(x, return_att=False)
        cls_logits = output[0]
        assert cls_logits.shape == (1, NUM_CLASSES)

    def test_backward_pass(self, model, dummy_input):
        """Gradients flow correctly in training mode."""
        model.train()
        output = model(dummy_input)
        cls_logits = output[0]
        loss = cls_logits.sum()
        loss.backward()

        # Check gradients exist on key parameters
        assert model.cls_token.grad is not None
        assert model.pos_embed_cls.grad is not None
        assert model.pos_embed_pat.grad is not None


class TestMCTformerCheckpoint:
    """Test checkpoint loading (skipped if weights not available)."""

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="MCTformerV2.pth not found in pretrained/")
    def test_load_pretrained_checkpoint(self):
        """Load the pre-trained MCTformer-V2 checkpoint."""
        model = create_mctformer_v2(
            num_classes=NUM_CLASSES,
            checkpoint_path=str(PRETRAINED_PATH),
            input_size=224,
        )
        assert isinstance(model, MCTformerPlus)

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="MCTformerV2.pth not found in pretrained/")
    def test_checkpoint_forward(self):
        """Pre-trained model produces valid output."""
        model = create_mctformer_v2(
            num_classes=NUM_CLASSES,
            checkpoint_path=str(PRETRAINED_PATH),
            input_size=IMAGE_SIZE,
        )
        model = model.to(DEVICE).eval()
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        logits, cams, _ = model(x, return_att=True)
        assert logits.shape == (1, NUM_CLASSES)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(cams).any()

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="MCTformerV2.pth not found in pretrained/")
    def test_checkpoint_state_dict_keys(self):
        """Verify the checkpoint has expected key structure."""
        state = torch.load(str(PRETRAINED_PATH), map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        # Key MCTformer-specific parameters must exist
        assert "cls_token" in state
        assert "pos_embed_cls" in state
        assert "pos_embed_pat" in state
        assert "head.weight" in state
        assert "blocks.0.attn.qkv.weight" in state
