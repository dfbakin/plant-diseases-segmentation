"""SegFormer model wrapper using HuggingFace transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerWrapper(nn.Module):
    """SegFormer wrapper for custom number of classes.

    SegFormer is a transformer-based architecture for semantic segmentation
    that outputs multi-scale features and uses a simple MLP decoder.

    Reference: https://arxiv.org/abs/2105.15203

    Attributes:
        model: HuggingFace SegFormer model.
        num_classes: Number of output classes.
    """

    # Mapping from variant to HuggingFace model name
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
        """Initialize SegFormer wrapper.

        Args:
            num_classes: Number of output classes.
            variant: Model variant (b0-b5).
            pretrained: Whether to load pretrained weights.
            image_size: Expected input image size.
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        model_name = self.VARIANT_MAP.get(variant.lower())
        if model_name is None:
            raise ValueError(f"Unknown variant: {variant}. Use one of {list(self.VARIANT_MAP.keys())}")

        if pretrained:
            # Load pretrained and modify classifier head
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Create from config
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Logits of shape (N, num_classes, H, W).
        """
        # HuggingFace returns dict with 'logits' key
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample to input resolution (SegFormer outputs 1/4 resolution)
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits

    @property
    def encoder_output_channels(self) -> list[int]:
        """Return encoder output channels for each stage."""
        return list(self.model.config.hidden_sizes)

