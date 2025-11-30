"""Evaluation script for trained models.

Usage:
    # Evaluate a checkpoint
    python src/evaluate.py checkpoint_path=outputs/experiment/checkpoints/best.ckpt

    # Evaluate with specific config
    python src/evaluate.py checkpoint_path=path/to/ckpt model=segformer
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import PlantSegDataModule
from src.models import SegmentationModule, create_model

log = logging.getLogger(__name__)


def evaluate(cfg: DictConfig) -> dict[str, float]:
    """Run evaluation on test set.

    Args:
        cfg: Hydra config with checkpoint_path.

    Returns:
        Dict of test metrics.
    """
    # Set seed
    L.seed_everything(cfg.experiment.seed, workers=True)

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

    # Load model from checkpoint
    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided")

    log.info(f"Loading checkpoint from: {checkpoint_path}")

    # Create model backbone
    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        encoder_name=cfg.model.get("encoder_name", "resnet50"),
        encoder_weights=None,  # Will be loaded from checkpoint
        variant=cfg.model.get("variant"),
        pretrained=False,
    )

    # Load checkpoint
    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path,
        model=model_backbone,
    )

    # Create trainer for evaluation
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )

    # Run test
    log.info("Running evaluation on test set...")
    results = trainer.test(module, datamodule=datamodule)

    return results[0] if results else {}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Args:
        cfg: Hydra config.
    """
    results = evaluate(cfg)
    log.info(f"Test Results:\n{OmegaConf.to_yaml(results)}")


if __name__ == "__main__":
    main()




