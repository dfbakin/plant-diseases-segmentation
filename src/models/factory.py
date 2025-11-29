"""Model factory for segmentation architectures."""

from typing import Any, Literal

import segmentation_models_pytorch as smp
import torch.nn as nn


# Type alias for supported models
ModelName = Literal[
    "deeplabv3plus",
    "unet",
    "segformer",
    "segnext",
]


def create_model(
    name: ModelName,
    num_classes: int = 2,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
    """Create a segmentation model.

    Supports:
    - DeepLabv3+ (via segmentation_models_pytorch)
    - U-Net (via segmentation_models_pytorch)
    - SegFormer (via transformers)
    - SegNeXt (via mmsegmentation/custom)

    Args:
        name: Model architecture name.
        num_classes: Number of output classes.
        encoder_name: Encoder backbone (for smp models).
        encoder_weights: Pretrained weights for encoder.
        **kwargs: Additional model-specific arguments.

    Returns:
        Initialized segmentation model.

    Raises:
        ValueError: If model name is not supported.
    """
    name = name.lower()

    if name == "deeplabv3plus":
        return _create_deeplabv3plus(
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            **kwargs,
        )

    elif name == "unet":
        return _create_unet(
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            **kwargs,
        )

    elif name == "segformer":
        return _create_segformer(
            num_classes=num_classes,
            **kwargs,
        )

    elif name == "segnext":
        return _create_segnext(
            num_classes=num_classes,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Supported: deeplabv3plus, unet, segformer, segnext"
        )


def _create_deeplabv3plus(
    num_classes: int,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
    """Create DeepLabv3+ model.

    Uses segmentation_models_pytorch for robust implementation.

    Args:
        num_classes: Number of output classes.
        encoder_name: Encoder backbone.
        encoder_weights: Pretrained weights.

    Returns:
        DeepLabv3+ model.
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,  # Raw logits for loss computation
    )


def _create_unet(
    num_classes: int,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
    """Create U-Net model.

    Args:
        num_classes: Number of output classes.
        encoder_name: Encoder backbone.
        encoder_weights: Pretrained weights.

    Returns:
        U-Net model.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )


def _create_segformer(
    num_classes: int,
    variant: str = "b3",
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Create SegFormer model.

    Uses HuggingFace transformers for SegFormer implementation.

    Args:
        num_classes: Number of output classes.
        variant: Model variant (b0, b1, b2, b3, b4, b5).
        pretrained: Whether to use pretrained weights.

    Returns:
        SegFormer wrapper model.
    """
    from src.models.segformer import SegFormerWrapper

    return SegFormerWrapper(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
    )


def _create_segnext(
    num_classes: int,
    variant: str = "base",
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Create SegNeXt model.

    Uses timm/mmseg for SegNeXt implementation.

    Args:
        num_classes: Number of output classes.
        variant: Model variant (tiny, small, base, large).
        pretrained: Whether to use pretrained weights.

    Returns:
        SegNeXt wrapper model.
    """
    from src.models.segnext import SegNeXtWrapper

    return SegNeXtWrapper(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        **kwargs,
    )


# Model configurations for reference
MODEL_CONFIGS = {
    "deeplabv3plus": {
        "encoders": ["resnet50", "resnet101", "efficientnet-b4", "mobilenet_v2"],
        "default_encoder": "resnet50",
    },
    "unet": {
        "encoders": ["resnet50", "resnet101", "efficientnet-b4", "mobilenet_v2"],
        "default_encoder": "resnet50",
    },
    "segformer": {
        "variants": ["b0", "b1", "b2", "b3", "b4", "b5"],
        "default_variant": "b3",
    },
    "segnext": {
        "variants": ["tiny", "small", "base", "large"],
        "default_variant": "base",
    },
}

