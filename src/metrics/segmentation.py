"""Segmentation metrics: mIoU, mAcc, Dice, Boundary IoU."""

import torch
from torchmetrics import Metric
import torch.nn.functional as F


class SegmentationMetrics(Metric):
    """Accumulates mIoU, Dice, and Boundary IoU across batches."""

    def __init__(self, num_classes: int = 2, ignore_index: int | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        assert num_classes > 0
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("boundary_intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("boundary_union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("pred_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update with preds (N, C, H, W) or (N, H, W) and target (N, H, W)."""
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        for cls in range(self.num_classes):
            pred_mask_spatial = preds == cls  # (N, H, W)
            target_mask_spatial = target == cls
            self.update_boundary_metrics(pred_mask_spatial, target_mask_spatial, cls)

        # Flatten for standard IoU/Dice computation
        preds_flat, target_flat = preds.flatten(), target.flatten()

        if self.ignore_index is not None:
            valid = target_flat != self.ignore_index
            preds_flat, target_flat = preds_flat[valid], target_flat[valid]

        for cls in range(self.num_classes):
            pred_mask = preds_flat == cls
            target_mask = target_flat == cls

            intersection = (pred_mask & target_mask).sum().float()
            pred_sum = pred_mask.sum().float()
            target_sum = target_mask.sum().float()

            self.intersection[cls] += intersection
            self.union[cls] += pred_sum + target_sum - intersection
            self.pred_sum[cls] += pred_sum
            self.target_sum[cls] += target_sum

    def compute(self) -> dict[str, torch.Tensor]:
        iou_per_class = self.intersection / (self.union + self.eps)
        acc_per_class = self.intersection / (self.target_sum + self.eps)  # TP / (TP + FN)
        boundary_iou_per_class = self.boundary_intersection / (self.boundary_union + self.eps)

        valid = self.union > 0
        miou = iou_per_class[valid].mean() if valid.any() else torch.tensor(0.0)
        macc = acc_per_class[valid].mean() if valid.any() else torch.tensor(0.0)
        boundary_iou = boundary_iou_per_class[valid].mean() if valid.any() else torch.tensor(0.0)

        dice_per_class = (2 * self.intersection) / (self.pred_sum + self.target_sum + self.eps)
        dice = dice_per_class[valid].mean() if valid.any() else torch.tensor(0.0)

        result = {
            "miou": miou,
            "macc": macc,
            "dice": dice,
            "iou_per_class": iou_per_class,
            "acc_per_class": acc_per_class,
            "boundary_iou": boundary_iou,
            "boundary_iou_per_class": boundary_iou_per_class,
        }

        # Add named per-class metrics for binary segmentation
        if self.num_classes == 2:
            result.update({
                "iou_background": iou_per_class[0],
                "iou_disease": iou_per_class[1],
                "acc_background": acc_per_class[0],
                "acc_disease": acc_per_class[1],
                "boundary_iou_background": boundary_iou_per_class[0],
                "boundary_iou_disease": boundary_iou_per_class[1],
            })

        return result


    def update_boundary_metrics(
        self, preds: torch.Tensor, target: torch.Tensor, cls: int, dilation: int = 3
    ) -> None:
        """Update Boundary IoU stats. Based on: https://arxiv.org/abs/2103.16562"""
        kernel_size = 2 * dilation + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=preds.device)

        def get_boundary(mask: torch.Tensor) -> torch.Tensor:
            mask_float = mask.float().unsqueeze(1)
            dilated = F.conv2d(mask_float, kernel, padding=dilation) > 0
            eroded = F.conv2d(mask_float, kernel, padding=dilation) == kernel.numel()
            return (dilated.float() - eroded.float()).squeeze(1)

        pred_boundary = get_boundary(preds)
        target_boundary = get_boundary(target)

        intersection = (pred_boundary * target_boundary).sum()
        self.boundary_intersection[cls] += intersection
        self.boundary_union[cls] += pred_boundary.sum() + target_boundary.sum() - intersection
