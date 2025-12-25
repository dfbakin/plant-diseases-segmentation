"""Factory for creating classification models.

Supports ResNet and EfficientNet from torchvision with pretrained weights.
"""

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

ClassifierName = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
]

RESNET_WEIGHTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": ResNet101_Weights.IMAGENET1K_V2,
}

EFFICIENTNET_WEIGHTS = {
    "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
    "efficientnet_b1": EfficientNet_B1_Weights.IMAGENET1K_V1,
    "efficientnet_b2": EfficientNet_B2_Weights.IMAGENET1K_V1,
    "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
    "efficientnet_b4": EfficientNet_B4_Weights.IMAGENET1K_V1,
    "efficientnet_b5": EfficientNet_B5_Weights.IMAGENET1K_V1,
    "efficientnet_b6": EfficientNet_B6_Weights.IMAGENET1K_V1,
    "efficientnet_b7": EfficientNet_B7_Weights.IMAGENET1K_V1,
}


def create_classifier(
    name: ClassifierName,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create a classification model.

    Args:
        name: Model architecture name
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights

    Returns:
        Model with replaced classification head
    """
    name = name.lower()

    if name.startswith("resnet"):
        return _create_resnet(name, num_classes, pretrained)
    elif name.startswith("efficientnet"):
        return _create_efficientnet(name, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def _create_resnet(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = RESNET_WEIGHTS.get(name) if pretrained else None
    model_fn = getattr(models, name)
    model = model_fn(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def _create_efficientnet(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = EFFICIENTNET_WEIGHTS.get(name) if pretrained else None
    model_fn = getattr(models, name)
    model = model_fn(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

