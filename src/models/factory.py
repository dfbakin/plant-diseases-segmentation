"""Model factory for segmentation architectures.

Supports: DeepLabv3+, U-Net (via smp), SegFormer (via HuggingFace), SegNeXt (custom).
"""

from typing import Any, Literal

import segmentation_models_pytorch as smp
import torch.nn as nn


ModelName = Literal["deeplabv3plus", "unet", "segformer", "segnext"]


def create_model(
    name: ModelName,
    num_classes: int = 2,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
    """Create a segmentation model by name.

    Raises ValueError for unknown model names.
    """
    name = name.lower()

    if name == "deeplabv3plus":
        return _create_deeplabv3plus(num_classes, encoder_name, encoder_weights, **kwargs)
    elif name == "unet":
        return _create_unet(num_classes, encoder_name, encoder_weights, **kwargs)
    elif name == "segformer":
        return _create_segformer(num_classes, **kwargs)
    elif name == "segnext":
        return _create_segnext(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: deeplabv3plus, unet, segformer, segnext")


def _create_deeplabv3plus(
    num_classes: int,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )


def _create_unet(
    num_classes: int,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> nn.Module:
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
    """SegFormer via HuggingFace transformers. Variants: b0-b5."""
    from src.models.segformer import SegFormerWrapper

    return SegFormerWrapper(num_classes=num_classes, variant=variant, pretrained=pretrained)


def _create_segnext(
    num_classes: int,
    variant: str = "base",
    use_stage_0: bool = False,
    pretrained: bool = True,
    encoder_checkpoint: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """SegNeXt with MSCAN encoder + LightHamHead decoder.

    Variants: tiny, small, base, large.
    If pretrained=True, loads ImageNet-1K weights from 'pretrained/mscan/'.
    """
    from src.models.segnext import SegNeXt, load_pretrained_mscan

    model = SegNeXt(num_classes=num_classes, variant=variant, use_stage_0=use_stage_0, **kwargs)

    if pretrained:
        load_pretrained_mscan(
            model=model,
            variant=variant,  # type: ignore[arg-type]
            checkpoint_path=encoder_checkpoint,
            strict=False,
        )

    return model


# Reference configs for available options
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
        "pretrained_encoder": True,
    },
}

