"""CAM (Class Activation Map) evaluation metrics.

Evaluates CAM quality by comparing generated heatmaps to ground truth
segmentation masks from PlantSeg dataset.
"""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


CAMMethod = Literal["gradcam", "layercam"]


class CAMEvaluator:
    """Evaluates CAM quality against ground truth segmentation masks."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        cam_method: CAMMethod = "gradcam",
        threshold: float = 0.5,
    ):
        """
        Args:
            model: Classification model
            target_layer: Layer to extract CAM from (typically last conv layer)
            cam_method: CAM generation method
            threshold: Threshold for binarizing CAM heatmap
        """
        self.model = model
        self.threshold = threshold

        cam_cls = {"gradcam": GradCAM, "layercam": LayerCAM}[cam_method]
        self.cam = cam_cls(model=model, target_layers=[target_layer])

    def generate_cam(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Generate CAM heatmaps for a batch of images.

        Args:
            images: (N, C, H, W) tensor
            target_classes: (N,) class indices. If None, uses predicted class.

        Returns:
            (N, H, W) CAM heatmaps normalized to [0, 1]
        """
        if target_classes is not None:
            targets = [ClassifierOutputTarget(c.item()) for c in target_classes]
        else:
            targets = None

        # pytorch-grad-cam returns (N, H, W) numpy array
        cams = self.cam(input_tensor=images, targets=targets)
        return cams

    def compute_metrics(
        self,
        cam: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, float]:
        """Compute CAM quality metrics for a single sample.

        Args:
            cam: (H, W) CAM heatmap in [0, 1]
            gt_mask: (H, W) binary ground truth mask

        Returns:
            Dictionary of metrics
        """
        # Resize CAM to match GT mask size if needed
        if cam.shape != gt_mask.shape:
            cam = self._resize_cam(cam, gt_mask.shape)

        gt_binary = (gt_mask > 0).astype(np.float32)
        cam_binary = (cam > self.threshold).astype(np.float32)

        # Pointing accuracy: does max CAM point fall in GT?
        max_idx = np.unravel_index(cam.argmax(), cam.shape)
        pointing_acc = float(gt_binary[max_idx] > 0)

        # Energy inside GT: fraction of CAM energy within GT region
        cam_sum = cam.sum() + 1e-8
        energy_inside = (cam * gt_binary).sum() / cam_sum

        # IoU between thresholded CAM and GT
        intersection = (cam_binary * gt_binary).sum()
        union = cam_binary.sum() + gt_binary.sum() - intersection + 1e-8
        iou = intersection / union

        # Precision and recall
        cam_area = cam_binary.sum() + 1e-8
        gt_area = gt_binary.sum() + 1e-8
        precision = intersection / cam_area
        recall = intersection / gt_area

        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "pointing_acc": pointing_acc,
            "energy_inside": float(energy_inside),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _resize_cam(self, cam: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resize CAM to target shape using bilinear interpolation."""
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            cam_tensor, size=target_shape, mode="bilinear", align_corners=False
        )
        return resized.squeeze().numpy()


class CAMMetricsAccumulator:
    """Accumulates CAM metrics across batches."""

    METRIC_NAMES = ["pointing_acc", "energy_inside", "iou", "precision", "recall", "f1"]

    def __init__(self):
        self.reset()

    def reset(self):
        self._metrics = {name: [] for name in self.METRIC_NAMES}
        self._count = 0

    def update(self, metrics: dict[str, float]):
        for name in self.METRIC_NAMES:
            if name in metrics:
                self._metrics[name].append(metrics[name])
        self._count += 1

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {name: 0.0 for name in self.METRIC_NAMES}
        return {name: np.mean(values) for name, values in self._metrics.items()}

    @property
    def count(self) -> int:
        return self._count


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Get the appropriate target layer for CAM extraction.

    Args:
        model: The classifier model
        model_name: Name of the model architecture

    Returns:
        Target layer for CAM
    """
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return model.layer4[-1]
    elif model_name.startswith("efficientnet"):
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model for CAM: {model_name}")

