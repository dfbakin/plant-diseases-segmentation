"""SegNeXt semantic segmentation model.

Pure PyTorch implementation combining MSCAN encoder and LightHamHead decoder.
Reference: https://arxiv.org/abs/2209.08575
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segnext.hamburger import LightHamHead
from src.models.segnext.mscan import MSCAN, MSCAN_CONFIGS, MSCANVariant


# Default decoder configurations per variant
DECODER_CONFIGS: dict[MSCANVariant, dict] = {
    "tiny": {
        "in_index": [1, 2, 3],
        "ham_channels": 256,
        "channels": 256,
        "md_r": 16,
    },
    "small": {
        "in_index": [1, 2, 3],
        "ham_channels": 256,
        "channels": 256,
        "md_r": 16,
    },
    "base": {
        "in_index": [1, 2, 3],
        "ham_channels": 512,
        "channels": 512,
        "md_r": 64,
    },
    "large": {
        "in_index": [1, 2, 3],
        "ham_channels": 1024,
        "channels": 1024,
        "md_r": 64,
    },
}


class SegNeXt(nn.Module):
    """SegNeXt semantic segmentation model.

    Combines MSCAN (Multi-Scale Convolutional Attention Network) encoder
    with LightHamHead (Hamburger-style) decoder for efficient semantic
    segmentation.

    Attributes:
        encoder: MSCAN backbone.
        decoder: LightHamHead decoder.
        num_classes: Number of output classes.
        variant: Model size variant.
    """

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
        """Initialize SegNeXt model.

        Args:
            num_classes: Number of output segmentation classes.
            variant: Model variant ('tiny', 'small', 'base', 'large').
            in_channels: Number of input image channels.
            in_index: Encoder stage indices to use in decoder.
                Default uses stages [1, 2, 3] (skips stage 0).
                Set use_stage_0=True to include stage 0.
            ham_channels: Hamburger module channels. If None, uses variant default.
            decoder_channels: Decoder output channels. If None, uses variant default.
            md_r: NMF decomposition rank. If None, uses variant default.
            dropout_ratio: Dropout ratio before classification.
            align_corners: align_corners for F.interpolate.
            use_stage_0: Whether to include stage 0 features in decoder.
                Only used if in_index is None.
        """
        super().__init__()

        if variant not in MSCAN_CONFIGS:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Use one of {list(MSCAN_CONFIGS.keys())}"
            )

        self.num_classes = num_classes
        self.variant = variant
        self.align_corners = align_corners

        # Get encoder config
        encoder_config = MSCAN_CONFIGS[variant]
        embed_dims = encoder_config["embed_dims"]

        # Create encoder
        self.encoder = MSCAN.from_variant(variant, in_chans=in_channels)

        # Get decoder config defaults
        decoder_default = DECODER_CONFIGS[variant]

        # Determine which stages to use
        if in_index is not None:
            self.in_index = in_index
        elif use_stage_0:
            self.in_index = [0, 1, 2, 3]
        else:
            self.in_index = decoder_default["in_index"]

        # Get input channels for selected stages
        encoder_out_channels = [embed_dims[i] for i in self.in_index]

        # Use provided values or defaults
        _ham_channels = ham_channels or decoder_default["ham_channels"]
        _decoder_channels = decoder_channels or decoder_default["channels"]
        _md_r = md_r or decoder_default["md_r"]

        # Create decoder
        self.decoder = LightHamHead(
            in_channels=encoder_out_channels,
            in_index=list(range(len(self.in_index))),  # Re-index for selected features
            ham_channels=_ham_channels,
            channels=_decoder_channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            md_r=_md_r,
            align_corners=align_corners,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        input_size = x.shape[2:]

        # Encoder: get multi-scale features
        features = self.encoder(x)

        # Select features for decoder
        selected_features = [features[i] for i in self.in_index]

        # Decoder: produce segmentation map
        logits = self.decoder(selected_features)

        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        return logits

    @property
    def encoder_output_channels(self) -> list[int]:
        """Return encoder output channels for each stage."""
        return self.encoder.embed_dims

    @classmethod
    def from_config(
        cls,
        variant: MSCANVariant = "base",
        num_classes: int = 2,
        **kwargs,
    ) -> "SegNeXt":
        """Create SegNeXt from variant configuration.

        Args:
            variant: Model variant.
            num_classes: Number of output classes.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Configured SegNeXt model.
        """
        return cls(num_classes=num_classes, variant=variant, **kwargs)


# Type alias for variant names (for external use)
SegNeXtVariant = Literal["tiny", "small", "base", "large"]

# Export variant configs for reference
SEGNEXT_VARIANTS = list(MSCAN_CONFIGS.keys())


