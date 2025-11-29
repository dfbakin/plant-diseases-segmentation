"""Base LightningModule for segmentation models."""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.segmentation import SegmentationMetrics


class SegmentationModule(L.LightningModule):
    """Base Lightning module for semantic segmentation.

    Handles training, validation, and test loops with proper metric logging.

    Attributes:
        model: The segmentation backbone (nn.Module).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        class_weights: Optional tensor for weighted loss.
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
        """Initialize segmentation module.

        Args:
            model: Segmentation backbone.
            num_classes: Number of segmentation classes.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            class_weights: Optional class weights for loss.
            loss_fn: Loss function ('cross_entropy' or 'dice').
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn_name = loss_fn

        # Register class weights if provided
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Metrics
        self.train_metrics = SegmentationMetrics(num_classes=num_classes)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes)
        self.test_metrics = SegmentationMetrics(num_classes=num_classes)

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Logits of shape (N, num_classes, H, W).
        """
        return self.model(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute segmentation loss.

        Args:
            logits: Model output (N, C, H, W).
            target: Ground truth (N, H, W).

        Returns:
            Scalar loss value.
        """
        if self.loss_fn_name == "dice":
            return self._dice_loss(logits, target)
        else:
            # Cross entropy with optional class weights
            return F.cross_entropy(
                logits,
                target.long(),
                weight=self.class_weights,
            )

    def _dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits: Model output (N, C, H, W).
            target: Ground truth (N, H, W).
            smooth: Smoothing factor to avoid division by zero.

        Returns:
            Dice loss (1 - Dice coefficient).
        """
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.long(), self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        # Compute per-class Dice
        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        cardinality = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice_per_class.mean()


    # TODO generalize step to call in training_step, validation_step, test_step

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dict with 'image' and 'mask'.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.compute_loss(logits, masks)

        # Update metrics
        self.train_metrics.update(logits, masks)

        # Log loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log("train/miou", metrics["miou"], prog_bar=True)
        self.log("train/dice", metrics["dice"])
        self.log("train/iou_background", metrics["iou_background"])
        self.log("train/iou_disease", metrics["iou_disease"])
        self.train_metrics.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dict with 'image' and 'mask'.
            batch_idx: Batch index.
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.compute_loss(logits, masks)

        # Update metrics
        self.val_metrics.update(logits, masks)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        self.log("val/miou", metrics["miou"], prog_bar=True)
        self.log("val/dice", metrics["dice"])
        self.log("val/iou_background", metrics["iou_background"])
        self.log("val/iou_disease", metrics["iou_disease"])
        self.val_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Dict with 'image' and 'mask'.
            batch_idx: Batch index.
        """
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.compute_loss(logits, masks)

        # Update metrics
        self.test_metrics.update(logits, masks)

        self.log("test/loss", loss, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        self.log("test/miou", metrics["miou"])
        self.log("test/dice", metrics["dice"])
        self.log("test/iou_background", metrics["iou_background"])
        self.log("test/iou_disease", metrics["iou_disease"])
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Dict with optimizer and LR scheduler config.
        """
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
        """Prediction step for inference.

        Args:
            batch: Dict with 'image'.
            batch_idx: Batch index.

        Returns:
            Dict with 'predictions', 'probabilities', and 'names'.
        """
        images = batch["image"]
        logits = self(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        return {
            "predictions": preds,
            "probabilities": probs,
            "names": batch.get("name", []),
        }

