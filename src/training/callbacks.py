"""Custom Lightning callbacks for visualization, MLflow logging, and early stopping."""

from pathlib import Path
from typing import Any

import lightning as L
import matplotlib
matplotlib.use("Agg")  # Headless backend - must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class VisualizationCallback(Callback):
    """Saves prediction visualizations to disk and MLflow.

    Logs at: first epoch, best epoch (by monitored metric), and final epoch.
    Only runs on rank 0 for multi-GPU compatibility.
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

    def _log_figure_to_mlflow(self, trainer: L.Trainer, fig_path: Path, artifact_name: str) -> None:
        import mlflow

        if not self.log_to_mlflow:
            return
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_artifact(str(fig_path), artifact_path=artifact_name)

    def on_validation_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any,
        batch: dict, batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero or batch_idx != 0:
            return

        epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        images, masks = batch["image"], batch["mask"]
        with torch.no_grad():
            preds = pl_module(images).argmax(dim=1)

        # Denormalize
        device = images.device
        images_denorm = (images * self.denorm_std.to(device) + self.denorm_mean.to(device)).clamp(0, 1)

        n = min(self.num_samples, images.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(preds[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        plt.tight_layout()
        fig_path = epoch_dir / "predictions.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        self._last_fig_path = fig_path
        if trainer.current_epoch == 0:
            self._log_figure_to_mlflow(trainer, fig_path, "visualizations/epoch_000")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero or self._last_fig_path is None:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (self.mode == "max" and current > self.best_value) or \
                  (self.mode == "min" and current < self.best_value)
        if is_best:
            self.best_value = current.item()
            self._log_figure_to_mlflow(trainer, self._last_fig_path, "visualizations/best")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.is_global_zero and self._last_fig_path and self._last_fig_path.exists():
            self._log_figure_to_mlflow(trainer, self._last_fig_path, "visualizations/final")


class MLflowModelCheckpoint(Callback):
    """Logs checkpoint paths and best metrics to MLflow.

    Only logs paths (not the heavy checkpoint files themselves).
    Uses trainer's MLFlowLogger context. Runs only on rank 0.
    """

    def __init__(self, monitor: str = "val/miou", mode: str = "max") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def _get_mlflow_logger(self, trainer: L.Trainer):
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

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        from mlflow import MlflowClient

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (self.mode == "max" and current > self.best_value) or \
                  (self.mode == "min" and current < self.best_value)
        if not is_best:
            return

        self.best_value = current.item()

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return

        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)
        client.log_metric(mlflow_logger.run_id, f"best_{self.monitor}", self.best_value)
        client.log_metric(mlflow_logger.run_id, "best_epoch", trainer.current_epoch)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        import mlflow
        from mlflow import MlflowClient

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return

        run_id = mlflow_logger.run_id
        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)

        client.log_metric(run_id, "final_epoch", trainer.current_epoch)

        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is not None:
            if ckpt_callback.best_model_path:
                client.log_param(run_id, "best_checkpoint_path", ckpt_callback.best_model_path)
            if ckpt_callback.last_model_path:
                client.log_param(run_id, "last_checkpoint_path", ckpt_callback.last_model_path)

        client.log_param(run_id, "output_dir", str(trainer.default_root_dir))

        # Log Hydra config for reproducibility
        with mlflow.start_run(run_id=run_id):
            hydra_dir = Path(trainer.default_root_dir) / ".hydra"
            if hydra_dir.exists():
                mlflow.log_artifacts(str(hydra_dir), artifact_path="hydra_config")


class EarlyStoppingWithPatience(Callback):
    """Early stopping that waits for min_epochs before activating."""

    def __init__(
        self,
        monitor: str = "val/miou",
        patience: int = 10,
        min_epochs: int = 20,
        mode: str = "max",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.wait_count = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_improvement = (self.mode == "max" and current > self.best_value) or \
                         (self.mode == "min" and current < self.best_value)

        if is_improvement:
            self.best_value = current.item()
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True




