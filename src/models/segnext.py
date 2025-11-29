"""SegNeXt model implementation using timm encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SegNeXtWrapper(nn.Module):
    """SegNeXt wrapper using timm MSCAN encoder.

    SegNeXt uses Multi-Scale Convolutional Attention Network (MSCAN) as encoder
    with a simple Hamburger decoder.

    Reference: https://arxiv.org/abs/2209.08575

    Attributes:
        encoder: MSCAN encoder from timm.
        decoder: Lightweight decoder head.
        num_classes: Number of output classes.
    """

    # MSCAN variants in timm
    VARIANT_MAP = {
        "tiny": "mscan_t",
        "small": "mscan_s",
        "base": "mscan_b",
        "large": "mscan_l",
    }

    # Output channels for each variant
    CHANNEL_MAP = {
        "tiny": [32, 64, 160, 256],
        "small": [64, 128, 320, 512],
        "base": [64, 128, 320, 512],
        "large": [64, 128, 320, 512],
    }

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "base",
        pretrained: bool = True,
        decoder_channels: int = 256,
    ) -> None:
        """Initialize SegNeXt wrapper.

        Args:
            num_classes: Number of output classes.
            variant: Model variant (tiny, small, base, large).
            pretrained: Whether to load pretrained weights.
            decoder_channels: Decoder hidden channels.
        """
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant.lower()

        if self.variant not in self.VARIANT_MAP:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Use one of {list(self.VARIANT_MAP.keys())}"
            )

        encoder_name = self.VARIANT_MAP[self.variant]
        encoder_channels = self.CHANNEL_MAP[self.variant]

        # Create MSCAN encoder from timm
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Lightweight decoder (simplified Hamburger-style)
        self.decoder = SegNeXtDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Logits of shape (N, num_classes, H, W).
        """
        # Get multi-scale features from encoder
        features = self.encoder(x)

        # Decode to segmentation map
        logits = self.decoder(features)

        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class SegNeXtDecoder(nn.Module):
    """Simplified decoder for SegNeXt.

    Uses feature pyramid fusion with depthwise separable convolutions.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: int = 256,
        num_classes: int = 2,
    ) -> None:
        """Initialize decoder.

        Args:
            encoder_channels: List of encoder output channels per stage.
            decoder_channels: Decoder hidden dimension.
            num_classes: Number of output classes.
        """
        super().__init__()

        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, decoder_channels, kernel_size=1)
            for ch in encoder_channels
        ])

        # Fusion convolutions (depthwise separable)
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    decoder_channels, decoder_channels,
                    kernel_size=3, padding=1, groups=decoder_channels,
                ),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
            )
            for _ in encoder_channels
        ])

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features.

        Args:
            features: List of feature maps from encoder stages.

        Returns:
            Segmentation logits (before upsampling).
        """
        # Apply lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down fusion
        target_size = laterals[0].shape[2:]

        fused = None
        for i in range(len(laterals) - 1, -1, -1):
            lateral = laterals[i]

            # Upsample to target resolution
            if lateral.shape[2:] != target_size:
                lateral = F.interpolate(
                    lateral, size=target_size,
                    mode="bilinear", align_corners=False,
                )

            # Add to fused features
            if fused is None:
                fused = lateral
            else:
                fused = fused + lateral

            # Apply fusion conv
            fused = self.fusion_convs[i](fused)

        # Segmentation head
        return self.seg_head(fused)




