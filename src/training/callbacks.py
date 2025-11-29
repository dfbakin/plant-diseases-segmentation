"""Custom Lightning callbacks."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np


class VisualizationCallback(Callback):
    """Callback for visualizing predictions during training.

    Saves sample predictions to disk at the end of each validation epoch.

    Attributes:
        output_dir: Directory to save visualizations.
        num_samples: Number of samples to visualize.
        denorm_mean: Normalization mean for denormalization.
        denorm_std: Normalization std for denormalization.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/visualizations",
        num_samples: int = 4,
        denorm_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        denorm_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize callback.

        Args:
            output_dir: Directory to save visualizations.
            num_samples: Number of samples to visualize per epoch.
            denorm_mean: Mean used for normalization.
            denorm_std: Std used for normalization.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.denorm_mean = torch.tensor(denorm_mean).view(3, 1, 1)
        self.denorm_std = torch.tensor(denorm_std).view(3, 1, 1)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save visualizations on first validation batch.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Batch outputs.
            batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        """
        if batch_idx != 0:
            return

        # Create output directory
        epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions
        images = batch["image"]
        masks = batch["mask"]

        with torch.no_grad():
            logits = pl_module(images)
            preds = logits.argmax(dim=1)

        # Denormalize images
        device = images.device
        mean = self.denorm_mean.to(device)
        std = self.denorm_std.to(device)
        images_denorm = images * std + mean
        images_denorm = images_denorm.clamp(0, 1)

        # Visualize first N samples
        n = min(self.num_samples, images.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(epoch_dir / "predictions.png", dpi=150)
        plt.close()


class MLflowModelCheckpoint(Callback):
    """Callback to log best model to MLflow.

    Logs the best model checkpoint as an MLflow artifact.

    Attributes:
        monitor: Metric to monitor for best model.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        monitor: str = "val/miou",
        mode: str = "max",
    ) -> None:
        """Initialize callback.

        Args:
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Check if current model is best and log to MLflow.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        import mlflow

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (
            (self.mode == "max" and current > self.best_value)
            or (self.mode == "min" and current < self.best_value)
        )

        if is_best:
            self.best_value = current.item()
            mlflow.log_metric(f"best_{self.monitor}", self.best_value)

            # Log model
            if mlflow.active_run():
                mlflow.pytorch.log_model(
                    pl_module.model,
                    artifact_path="best_model",
                )


class EarlyStoppingWithPatience(Callback):
    """Early stopping with warmup patience.

    Allows model to train for minimum epochs before early stopping kicks in.

    Attributes:
        monitor: Metric to monitor.
        patience: Number of epochs with no improvement to wait.
        min_epochs: Minimum epochs before early stopping.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        monitor: str = "val/miou",
        patience: int = 10,
        min_epochs: int = 20,
        mode: str = "max",
    ) -> None:
        """Initialize callback.

        Args:
            monitor: Metric to monitor.
            patience: Patience epochs.
            min_epochs: Minimum training epochs.
            mode: 'min' or 'max'.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.wait_count = 0

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Check for early stopping condition.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        current_epoch = trainer.current_epoch
        if current_epoch < self.min_epochs:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_improvement = (
            (self.mode == "max" and current > self.best_value)
            or (self.mode == "min" and current < self.best_value)
        )

        if is_improvement:
            self.best_value = current.item()
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True




