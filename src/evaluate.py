"""Evaluation script.

Examples:
    python src/evaluate.py checkpoint_path=outputs/.../best.ckpt
    python src/evaluate.py checkpoint_path=path/to/ckpt model=segformer
"""

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.data import PlantSegDataModule
from src.models import SegmentationModule, create_model

log = logging.getLogger(__name__)


def evaluate(cfg: DictConfig) -> dict[str, float]:
    """Evaluate checkpoint on test set. Returns test metrics dict."""
    L.seed_everything(cfg.experiment.seed, workers=True)

    datamodule = PlantSegDataModule(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        mean=cfg.data.normalization.mean,
        std=cfg.data.normalization.std,
    )

    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided")

    log.info(f"Loading checkpoint: {checkpoint_path}")

    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        encoder_name=cfg.model.get("encoder_name", "resnet50"),
        encoder_weights=None,
        variant=cfg.model.get("variant"),
        pretrained=False,
    )

    module = SegmentationModule.load_from_checkpoint(checkpoint_path, model=model_backbone)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )

    log.info("Running test evaluation...")
    results = trainer.test(module, datamodule=datamodule)
    return results[0] if results else {}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    results = evaluate(cfg)
    log.info(f"Test Results:\n{OmegaConf.to_yaml(results)}")


if __name__ == "__main__":
    main()




