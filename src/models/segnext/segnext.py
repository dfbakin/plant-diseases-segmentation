"""SegNeXt: MSCAN encoder + LightHamHead decoder.

Pure PyTorch implementation. Reference: https://arxiv.org/abs/2209.08575
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segnext.hamburger import LightHamHead
from src.models.segnext.mscan import MSCAN, MSCAN_CONFIGS, MSCANVariant


DECODER_CONFIGS: dict[MSCANVariant, dict] = {
    "tiny": {"in_index": [1, 2, 3], "ham_channels": 256, "channels": 256, "md_r": 16},
    "small": {"in_index": [1, 2, 3], "ham_channels": 256, "channels": 256, "md_r": 16},
    "base": {"in_index": [1, 2, 3], "ham_channels": 512, "channels": 512, "md_r": 64},
    "large": {"in_index": [1, 2, 3], "ham_channels": 1024, "channels": 1024, "md_r": 64},
}


class SegNeXt(nn.Module):
    """MSCAN encoder + LightHamHead (Hamburger) decoder for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 2,
        variant: MSCANVariant = "base",
        in_channels: int = 3,
        in_index: list[int] | None = None,
        ham_channels: int | None = None,
        decoder_channels: int | None = None,
        md_r: int | None = None,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_stage_0: bool = False,
    ) -> None:
        super().__init__()

        if variant not in MSCAN_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Use one of {list(MSCAN_CONFIGS.keys())}")

        self.num_classes = num_classes
        self.variant = variant
        self.align_corners = align_corners

        encoder_config = MSCAN_CONFIGS[variant]
        embed_dims = encoder_config["embed_dims"]

        self.encoder = MSCAN.from_variant(variant, in_chans=in_channels)

        decoder_default = DECODER_CONFIGS[variant]

        if in_index is not None:
            self.in_index = in_index
        elif use_stage_0:
            self.in_index = [0, 1, 2, 3]
        else:
            self.in_index = decoder_default["in_index"]

        encoder_out_channels = [embed_dims[i] for i in self.in_index]

        self.decoder = LightHamHead(
            in_channels=encoder_out_channels,
            in_index=list(range(len(self.in_index))),
            ham_channels=ham_channels or decoder_default["ham_channels"],
            channels=decoder_channels or decoder_default["channels"],
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            md_r=md_r or decoder_default["md_r"],
            align_corners=align_corners,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, num_classes, H, W) at input resolution."""
        input_size = x.shape[2:]
        features = self.encoder(x)
        selected = [features[i] for i in self.in_index]
        logits = self.decoder(selected)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=self.align_corners)

    @property
    def encoder_output_channels(self) -> list[int]:
        return self.encoder.embed_dims

    @classmethod
    def from_config(cls, variant: MSCANVariant = "base", num_classes: int = 2, **kwargs) -> "SegNeXt":
        return cls(num_classes=num_classes, variant=variant, **kwargs)


SegNeXtVariant = Literal["tiny", "small", "base", "large"]
SEGNEXT_VARIANTS = list(MSCAN_CONFIGS.keys())


