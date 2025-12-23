"""Base LightningModule for segmentation models."""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.segmentation import SegmentationMetrics


class SegmentationModule(L.LightningModule):
    """Lightning module wrapping any segmentation backbone.

    Handles train/val/test loops with mIoU, mAcc, Dice, and Boundary IoU metrics.
    Supports cross-entropy or Dice loss with optional class weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        loss_fn: str = "cross_entropy",
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn_name = loss_fn

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.train_metrics = SegmentationMetrics(num_classes=num_classes)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes)
        self.test_metrics = SegmentationMetrics(num_classes=num_classes)

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (N, num_classes, H, W)."""
        return self.model(x)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn_name == "dice":
            return self._dice_loss(logits, target)
        return F.cross_entropy(logits, target.long(), weight=self.class_weights)

    def _dice_loss(
        self, logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """Soft Dice loss averaged over classes."""
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.long(), self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        cardinality = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice_per_class.mean()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        self.train_metrics.update(logits, masks)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=images.size(0),
        )
        return loss

    def _log_metrics(self, metrics: dict, prefix: str, prog_bar: bool = False) -> None:
        self.log(f"{prefix}/miou", metrics["miou"], prog_bar=prog_bar)
        self.log(f"{prefix}/macc", metrics["macc"])
        self.log(f"{prefix}/dice", metrics["dice"])
        self.log(f"{prefix}/boundary_iou", metrics["boundary_iou"])

        if self.num_classes == 2:
            self.log(f"{prefix}/iou_background", metrics["iou_background"])
            self.log(f"{prefix}/iou_disease", metrics["iou_disease"])
            self.log(f"{prefix}/acc_background", metrics["acc_background"])
            self.log(f"{prefix}/acc_disease", metrics["acc_disease"])
            self.log(
                f"{prefix}/boundary_iou_background", metrics["boundary_iou_background"]
            )
            self.log(f"{prefix}/boundary_iou_disease", metrics["boundary_iou_disease"])

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        self._log_metrics(metrics, "train", prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        self.val_metrics.update(logits, masks)
        self.log(
            "val/loss", loss, prog_bar=True, on_epoch=True, batch_size=images.size(0)
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self._log_metrics(metrics, "val", prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        images, masks = batch["image"], batch["mask"]
        logits = self(images)
        loss = self.compute_loss(logits, masks)

        self.test_metrics.update(logits, masks)
        self.log("test/loss", loss, on_epoch=True, batch_size=images.size(0))

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self._log_metrics(metrics, "test")
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        images = batch["image"]
        logits = self(images)
        return {
            "predictions": logits.argmax(dim=1),
            "probabilities": F.softmax(logits, dim=1),
            "names": batch.get("name", []),
        }
