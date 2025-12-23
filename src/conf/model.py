from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    name: str
    num_classes: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"  # Options: cross_entropy, dice


@dataclass
class DeepLabV3PlusConfig(ModelConfig):
    name: str = "deeplabv3plus"

    encoder_name: str = "resnet50"  # Options: resnet50, resnet101, efficientnet-b4, mobilenet_v2
    encoder_weights: str = "imagenet"

    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"


@dataclass
class UNetConfig(ModelConfig):
    name: str = "unet"

    encoder_name: str = "resnet50"  # Options: resnet50, resnet101, efficientnet-b4, mobilenet_v2
    encoder_weights: str = "imagenet"

    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"


@dataclass
class SegFormerConfig(ModelConfig):
    name: str = "segformer"

    # Model variant: b0, b1, b2, b3, b4, b5
    variant: str = "b3"
    pretrained: bool = True

    learning_rate: float = 6e-5
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"

    # Not used but needed for compatibility with factory
    encoder_name: Optional[str] = None
    encoder_weights: Optional[str] = None


@dataclass
class SegNeXtConfig(ModelConfig):
    name: str = "segnext"

    # Model variant: tiny, small, base, large
    variant: str = "base"
    pretrained: bool = True
    decoder_channels: Optional[int] = None

    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"

    # Not used but needed for compatibility with factory
    encoder_name: Optional[str] = None
    encoder_weights: Optional[str] = None
