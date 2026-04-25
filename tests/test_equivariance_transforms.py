"""Unit tests for ``src.wsss.spdnet.equivariance_transforms``.

Test names map to the invariants documented in the SPDNet aux-losses plan
(Phase C / "TestTransforms"). Each test is < 1 s on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.wsss.spdnet import equivariance_transforms as ET

torch.manual_seed(0)
BATCH_SIZE = 2
H = W = 16


class TestTransforms:
    def test_constants_consistent(self) -> None:
        assert ET.NUM_TRANSFORMS == 5
        assert len(ET.TRANSFORM_NAMES) == ET.NUM_TRANSFORMS
        ids = {
            ET.T_ID_IDENTITY, ET.T_ID_HFLIP, ET.T_ID_ROT90,
            ET.T_ID_ROT180, ET.T_ID_ROT270,
        }
        assert ids == set(range(ET.NUM_TRANSFORMS))

    def test_identity_roundtrip_image(self) -> None:
        x = torch.randn(BATCH_SIZE, 3, H, W)
        for t in range(ET.NUM_TRANSFORMS):
            y = ET.apply(x, t)
            x_back = ET.inverse(y, t)
            assert torch.equal(x_back, x), (
                f"identity roundtrip broken for t={t} ({ET.TRANSFORM_NAMES[t]})"
            )

    def test_identity_roundtrip_attention(self) -> None:
        x = torch.randn(BATCH_SIZE, H, W)
        for t in range(ET.NUM_TRANSFORMS):
            y = ET.apply(x, t)
            x_back = ET.inverse(y, t)
            assert torch.equal(x_back, x), (
                f"identity roundtrip broken for t={t} ({ET.TRANSFORM_NAMES[t]})"
            )

    def test_zero_input(self) -> None:
        x = torch.zeros(BATCH_SIZE, 3, H, W)
        for t in range(ET.NUM_TRANSFORMS):
            y = ET.apply(x, t)
            assert torch.equal(y, x), f"zero input not preserved for t={t}"

    def test_shape_preservation_square(self) -> None:
        x = torch.randn(BATCH_SIZE, 3, H, W)
        for t in range(ET.NUM_TRANSFORMS):
            assert ET.apply(x, t).shape == x.shape, f"shape changed for t={t}"

    def test_image_and_attention_share_t(self) -> None:
        """Same ``t_id`` applied to image and attention must produce the same
        geometric transform: a bright pixel ends up at the same ``(i, j)``."""
        img = torch.zeros(1, 1, H, W)
        attn = torch.zeros(1, H, W)
        img[0, 0, 2, 3] = 1.0
        attn[0, 2, 3] = 1.0
        for t in range(ET.NUM_TRANSFORMS):
            img_t = ET.apply(img, t)
            attn_t = ET.apply(attn, t)
            assert img_t[0, 0].sum().item() == 1.0
            assert attn_t[0].sum().item() == 1.0
            iy, ix = torch.nonzero(img_t[0, 0])[0].tolist()
            ay, ax = torch.nonzero(attn_t[0])[0].tolist()
            assert (iy, ix) == (ay, ax), (
                f"t={t} ({ET.TRANSFORM_NAMES[t]}): image bright at "
                f"({iy},{ix}) but attention at ({ay},{ax})"
            )

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transform id"):
            ET.apply(torch.zeros(1, H, W), 99)
        with pytest.raises(ValueError, match="Unknown transform id"):
            ET.inverse(torch.zeros(1, H, W), 99)

    def test_sample_transform_id_in_range(self) -> None:
        for _ in range(50):
            t = ET.sample_transform_id()
            assert 0 <= t < ET.NUM_TRANSFORMS

    def test_sample_transform_id_deterministic_with_generator(self) -> None:
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        s1 = [ET.sample_transform_id(g1) for _ in range(20)]
        s2 = [ET.sample_transform_id(g2) for _ in range(20)]
        assert s1 == s2, "sampler should be deterministic with same generator seed"
        assert all(0 <= t < ET.NUM_TRANSFORMS for t in s1)

    def test_hflip_is_involution(self) -> None:
        x = torch.randn(BATCH_SIZE, H, W)
        twice = ET.apply(ET.apply(x, ET.T_ID_HFLIP), ET.T_ID_HFLIP)
        assert torch.equal(twice, x)

    def test_rot180_is_involution(self) -> None:
        x = torch.randn(BATCH_SIZE, H, W)
        twice = ET.apply(ET.apply(x, ET.T_ID_ROT180), ET.T_ID_ROT180)
        assert torch.equal(twice, x)
