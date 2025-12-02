"""Main training entry point.

Usage:
    # Train with default config (DeepLabv3+)
    python src/train.py

    # Train with SegFormer
    python src/train.py model=segformer

    # Train with SegNeXt and custom epochs
    python src/train.py model=segnext trainer.max_epochs=50

    # Override multiple settings
    python src/train.py model=segformer data.batch_size=16 model.learning_rate=1e-4

    # Multirun for benchmarking all models
    python src/train.py --multirun model=deeplabv3plus,segformer,segnext
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.data import PlantSegDataModule
from src.models import SegmentationModule, create_model
from src.training.callbacks import (
    EarlyStoppingWithPatience,
    MLflowModelCheckpoint,
    VisualizationCallback,
)

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> float:
    """Run training with given config.

    Args:
        cfg: Hydra config.

    Returns:
        Best validation mIoU.
    """
    # Set seed for reproducibility
    L.seed_everything(cfg.experiment.seed, workers=True)

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directories
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    datamodule = PlantSegDataModule(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        mean=cfg.data.normalization.mean,
        std=cfg.data.normalization.std,
    )

    # Create model
    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        encoder_name=cfg.model.get("encoder_name", "resnet50"),
        encoder_weights=cfg.model.get("encoder_weights", "imagenet"),
        variant=cfg.model.get("variant"),
        pretrained=cfg.model.get("pretrained", True),
        decoder_channels=cfg.model.get("decoder_channels"),
    )

    # Wrap in Lightning module
    module = SegmentationModule(
        model=model_backbone,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        loss_fn=cfg.model.loss_fn,
    )

    # Setup MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"{cfg.model.name}_{cfg.experiment.seed}",
        tags={
            "model": cfg.model.name,
            "encoder": cfg.model.get("encoder_name", cfg.model.get("variant", "default")),
        },
    )

    # Callbacks
    callbacks = [
        # Checkpointing
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename="{epoch:02d}-{val/miou:.4f}",
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            save_last=cfg.trainer.checkpoint.save_last,
        ),
        # Early stopping
        EarlyStoppingWithPatience(
            monitor=cfg.trainer.early_stopping.monitor,
            patience=cfg.trainer.early_stopping.patience,
            mode=cfg.trainer.early_stopping.mode,
            min_epochs=cfg.trainer.early_stopping.min_epochs,
        ),
        # LR monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # Progress bar
        RichProgressBar(),
        # Visualization
        VisualizationCallback(
            output_dir=output_dir / "visualizations",
            num_samples=4,
            denorm_mean=cfg.data.normalization.mean,
            denorm_std=cfg.data.normalization.std,
        ),
        # MLflow model logging
        MLflowModelCheckpoint(
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
        ),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=cfg.trainer.deterministic,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    # Train
    log.info("Starting training...")
    trainer.fit(module, datamodule=datamodule)

    # Test with best checkpoint
    log.info("Running test evaluation...")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    # Return best validation metric for hyperparameter optimization
    best_miou = trainer.callback_metrics.get("val/miou", torch.tensor(0.0))
    return float(best_miou)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main entry point.

    Args:
        cfg: Hydra config.

    Returns:
        Best validation mIoU.
    """
    return train(cfg)


if __name__ == "__main__":
    main()




