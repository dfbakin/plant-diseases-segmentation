"""Segmentation metrics: mIoU, Dice, Boundary IoU."""

import torch
from torchmetrics import Metric


class SegmentationMetrics(Metric):
    """Combined segmentation metrics.

    Computes:
    - Mean Intersection over Union (mIoU)
    - Dice Coefficient
    - Per-class IoU

    Attributes:
        num_classes: Number of segmentation classes.
        ignore_index: Class index to ignore (e.g., unlabeled regions).
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int | None = None,
    ) -> None:
        """Initialize metrics.

        Args:
            num_classes: Number of segmentation classes.
            ignore_index: Optional class index to ignore in computation.
        """
        super().__init__()
        assert num_classes > 0, "Number of classes must be greater than 0"
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Accumulate intersection and union for each class
        self.add_state(
            "intersection",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "union",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "target_sum",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states.

        Args:
            preds: Predictions of shape (N, C, H, W) or (N, H, W).
            target: Ground truth of shape (N, H, W).
        """
        # Handle logits (N, C, H, W) -> class predictions (N, H, W)
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        preds = preds.flatten()
        target = target.flatten()

        # Mask for valid pixels
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            preds = preds[valid]
            target = target[valid]

        # Compute per-class stats
        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().float()
            pred_sum = pred_mask.sum().float()
            target_sum = target_mask.sum().float()
            union = pred_sum + target_sum - intersection

            self.intersection[cls] += intersection
            self.union[cls] += union
            self.pred_sum[cls] += pred_sum
            self.target_sum[cls] += target_sum

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute final metrics.

        Returns:
            Dict with 'miou', 'dice', 'iou_per_class'.
        """
        # IoU per class
        iou_per_class = self.intersection / (self.union + 1e-8)

        # Mean IoU (excluding classes with no samples)
        valid_classes = self.union > 0
        miou = iou_per_class[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)

        # Dice coefficient (F1 score per class, then averaged)
        dice_per_class = (2 * self.intersection) / (self.pred_sum + self.target_sum + 1e-8)
        dice = dice_per_class[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)

        return {
            "miou": miou,
            "dice": dice,
            "iou_per_class": iou_per_class,
            "iou_background": iou_per_class[0],
            "iou_disease": iou_per_class[1],
        }


def compute_boundary_iou(
    preds: torch.Tensor,
    target: torch.Tensor,
    dilation: int = 3,
) -> torch.Tensor:
    """Compute Boundary IoU.

    Measures segmentation quality at object boundaries.

    Based on: https://arxiv.org/abs/2103.16562

    Args:
        preds: Binary predictions (N, H, W).
        target: Binary targets (N, H, W).
        dilation: Boundary thickness in pixels.

    Returns:
        Boundary IoU score.
    """
    import torch.nn.functional as F

    # Create morphological kernel
    kernel_size = 2 * dilation + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=preds.device)

    # Extract boundaries using morphological gradient
    def get_boundary(mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.float().unsqueeze(1)  # (N, 1, H, W)
        dilated = F.conv2d(mask_float, kernel, padding=dilation) > 0
        eroded = F.conv2d(mask_float, kernel, padding=dilation) == kernel.numel()
        boundary = dilated.float() - eroded.float()
        return boundary.squeeze(1)  # (N, H, W) 

    pred_boundary = get_boundary(preds)
    target_boundary = get_boundary(target)

    # Compute IoU on boundaries
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection

    return intersection / (union + 1e-8)

