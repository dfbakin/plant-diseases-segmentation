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

    Saves sample predictions to disk and optionally to MLflow.
    Logs to MLflow at: first epoch, best epoch, and final epoch.

    Attributes:
        output_dir: Directory to save visualizations.
        num_samples: Number of samples to visualize.
        denorm_mean: Normalization mean for denormalization.
        denorm_std: Normalization std for denormalization.
        log_to_mlflow: Whether to log key visualizations to MLflow.
        monitor: Metric to monitor for "best" visualization.
        mode: 'min' or 'max' for best metric.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/visualizations",
        num_samples: int = 4,
        denorm_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        denorm_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        log_to_mlflow: bool = True,
        monitor: str = "val/miou",
        mode: str = "max",
    ) -> None:
        """Initialize callback.

        Args:
            output_dir: Directory to save visualizations.
            num_samples: Number of samples to visualize per epoch.
            denorm_mean: Mean used for normalization.
            denorm_std: Std used for normalization.
            log_to_mlflow: Whether to log to MLflow artifacts.
            monitor: Metric to monitor for best visualization.
            mode: 'min' or 'max'.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.denorm_mean = torch.tensor(denorm_mean).view(3, 1, 1)
        self.denorm_std = torch.tensor(denorm_std).view(3, 1, 1)
        self.log_to_mlflow = log_to_mlflow
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self._last_fig_path: Path | None = None

    def _get_mlflow_logger(self, trainer: L.Trainer):
        """Get MLFlowLogger from trainer if available."""
        from lightning.pytorch.loggers import MLFlowLogger

        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        if hasattr(trainer.logger, "_loggers"):
            for logger in trainer.logger._loggers:
                if isinstance(logger, MLFlowLogger):
                    return logger
        return None

    def _log_figure_to_mlflow(
        self,
        trainer: L.Trainer,
        fig_path: Path,
        artifact_name: str,
    ) -> None:
        """Log a figure to MLflow artifacts."""
        import mlflow

        if not self.log_to_mlflow:
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return

        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_artifact(str(fig_path), artifact_path=artifact_name)

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
        fig_path = epoch_dir / "predictions.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Store path for MLflow logging in on_validation_epoch_end
        self._last_fig_path = fig_path

        # Log first epoch to MLflow
        if trainer.current_epoch == 0:
            self._log_figure_to_mlflow(trainer, fig_path, "visualizations/epoch_000")

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log best visualization to MLflow when metric improves."""
        if self._last_fig_path is None:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (
            (self.mode == "max" and current > self.best_value)
            or (self.mode == "min" and current < self.best_value)
        )

        if is_best:
            self.best_value = current.item()
            # Log as "best" visualization
            self._log_figure_to_mlflow(
                trainer, self._last_fig_path, "visualizations/best"
            )

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log final visualization to MLflow."""
        if self._last_fig_path is not None and self._last_fig_path.exists():
            self._log_figure_to_mlflow(
                trainer, self._last_fig_path, "visualizations/final"
            )


class MLflowModelCheckpoint(Callback):
    """Callback to log best model to MLflow.

    Logs the best model checkpoint as an MLflow artifact.
    Uses the trainer's MLFlowLogger run context instead of global mlflow.

    Attributes:
        monitor: Metric to monitor for best model.
        mode: 'min' or 'max'.
        log_model: Whether to log full model (can be large).
    """

    def __init__(
        self,
        monitor: str = "val/miou",
        mode: str = "max",
        log_model: bool = True,
        input_size: tuple[int, int] = (512, 512),
    ) -> None:
        """Initialize callback.

        Args:
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
            log_model: Whether to log the model artifact.
            input_size: Input image size for signature inference.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.log_model = log_model
        self.input_size = input_size
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def _get_mlflow_logger(self, trainer: L.Trainer):
        """Get MLFlowLogger from trainer if available."""
        from lightning.pytorch.loggers import MLFlowLogger

        if trainer.logger is None:
            return None
        # Handle single logger
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        # Handle multiple loggers
        if hasattr(trainer.logger, "_loggers"):
            for logger in trainer.logger._loggers:
                if isinstance(logger, MLFlowLogger):
                    return logger
        return None

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
        from mlflow import MlflowClient

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (
            (self.mode == "max" and current > self.best_value)
            or (self.mode == "min" and current < self.best_value)
        )

        if not is_best:
            return

        self.best_value = current.item()

        # Get MLflow logger and run_id from trainer
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        run_id = mlflow_logger.run_id
        if run_id is None:
            return

        # Use MlflowClient to log to the correct run
        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)

        # Log best metrics
        client.log_metric(run_id, f"best_{self.monitor}", self.best_value)
        client.log_metric(run_id, "best_epoch", trainer.current_epoch)

        # Log model without input_example to avoid device mismatch issues
        # MLflow tries to run inference for signature which fails on GPU models
        if self.log_model:
            with mlflow.start_run(run_id=run_id):
                mlflow.pytorch.log_model(
                    pl_module.model,
                    name="best_model",
                )

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log final summary metrics, config, and checkpoint paths.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        import mlflow
        from mlflow import MlflowClient

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        run_id = mlflow_logger.run_id
        if run_id is None:
            return

        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)

        # Log final epoch
        client.log_metric(run_id, "final_epoch", trainer.current_epoch)

        # Log best checkpoint path as parameter
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is not None and ckpt_callback.best_model_path:
            client.log_param(run_id, "best_checkpoint_path", ckpt_callback.best_model_path)

        # Log output directory
        client.log_param(run_id, "output_dir", str(trainer.default_root_dir))

        with mlflow.start_run(run_id=run_id):
            # Log Hydra config as artifact
            hydra_dir = Path(trainer.default_root_dir) / ".hydra"
            if hydra_dir.exists():
                mlflow.log_artifacts(str(hydra_dir), artifact_path="hydra_config")

            # Log all visualizations directory
            vis_dir = Path(trainer.default_root_dir) / "visualizations"
            if vis_dir.exists():
                mlflow.log_artifacts(str(vis_dir), artifact_path="visualizations/all_epochs")


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




