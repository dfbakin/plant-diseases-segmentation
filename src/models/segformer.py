"""SegFormer wrapper using HuggingFace transformers.

Reference: https://arxiv.org/abs/2105.15203
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerWrapper(nn.Module):
    """Wraps HuggingFace SegFormer for custom num_classes
    and upsamples output to input resolution."""

    VARIANT_MAP = {
        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "b3",
        pretrained: bool = True,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        model_name = self.VARIANT_MAP.get(variant.lower())
        if model_name is None:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Use one of {list(self.VARIANT_MAP.keys())}"
            )

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits upsampled to input resolution."""
        logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )

    @property
    def encoder_output_channels(self) -> list[int]:
        return list(self.model.config.hidden_sizes)
