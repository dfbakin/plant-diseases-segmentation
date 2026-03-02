"""Train DeepLab-v3+ on pseudo masks (WSSS student model).

Uses the existing SegmentationModule + model factory with WSSDataset.
Dataset-agnostic: configure via Hydra overrides.

Example:
    python src/train_deeplab_wsss.py \
        train_mask_dir=outputs/pseudo_masks \
        val_mask_dir=data/VOC2012/SegmentationClassAug
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.voc_wsss import WSSDataset
from src.models.base import SegmentationModule
from src.models.factory import create_model

log = logging.getLogger(__name__)


@dataclass
class DeepLabWSSConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    train_image_dir: str = "data/VOC2012/JPEGImages"
    train_mask_dir: str = "outputs/pseudo_masks"
    val_image_dir: str = "data/VOC2012/JPEGImages"
    val_mask_dir: str = "data/VOC2012/SegmentationClassAug"
    val_names_file: str = "data/VOC2012/ImageSets/Segmentation/val.txt"
    image_ext: str = ".jpg"

    encoder_name: str = "resnet101"
    encoder_weights: str = "imagenet"
    num_classes: int = 21
    ignore_index: int = 255
    image_size: int = 512

    batch_size: int = 16
    max_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    loss_fn: str = "cross_entropy"
    num_workers: int = 8
    precision: str = "16-mixed"
    seed: int = 0

    early_stopping_patience: int = 15

    experiment_name: str = "deeplab_wsss"
    output_dir: str = "outputs/deeplab_wsss"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "${experiment_name}"


cs = ConfigStore.instance()
cs.store(name="deeplab_wss_config", node=DeepLabWSSConfig)


def train_deeplab_wsss(cfg: DeepLabWSSConfig) -> None:
    L.seed_everything(cfg.seed)

    # --- Data ---
    train_ds = WSSDataset(
        image_dir=cfg.train_image_dir,
        mask_dir=cfg.train_mask_dir,
        image_ext=cfg.image_ext,
        transform=get_train_transforms(cfg.image_size),
        is_train=True,
    )
    log.info(f"Train set: {len(train_ds)} images from {cfg.train_mask_dir}")

    val_loader = None
    if cfg.val_mask_dir and Path(cfg.val_mask_dir).exists():
        val_ds = WSSDataset(
            image_dir=cfg.val_image_dir,
            mask_dir=cfg.val_mask_dir,
            image_ext=cfg.image_ext,
            transform=get_val_transforms(cfg.image_size),
            is_train=False,
        )
        if cfg.val_names_file and Path(cfg.val_names_file).exists():
            allowed = set(
                l.strip()
                for l in Path(cfg.val_names_file).read_text().splitlines()
                if l.strip()
            )
            val_ds.names = [n for n in val_ds.names if n in allowed]
            log.info(f"Filtered val to {len(val_ds.names)} images via {cfg.val_names_file}")
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        log.info(f"Val set: {len(val_ds)} images from {cfg.val_mask_dir}")
    else:
        log.warning("No val_mask_dir or directory missing, training without validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Model ---
    backbone = create_model(
        "deeplabv3plus",
        num_classes=cfg.num_classes,
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
    )
    module = SegmentationModule(
        model=backbone,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        loss_fn=cfg.loss_fn,
        ignore_index=cfg.ignore_index,
    )
    log.info(
        f"DeepLabV3+ encoder={cfg.encoder_name}, "
        f"num_classes={cfg.num_classes}, ignore_index={cfg.ignore_index}"
    )

    # --- Output / logging ---
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"deeplab_{cfg.encoder_name}_{cfg.image_size}_{cfg.seed}",
        tags={
            "model": "deeplabv3plus",
            "encoder": cfg.encoder_name,
            "image_size": str(cfg.image_size),
            "num_classes": str(cfg.num_classes),
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    # --- Callbacks ---
    monitor = "val/miou" if val_loader is not None else "train/loss"
    monitor_mode = "max" if monitor == "val/miou" else "min"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="deeplab-{epoch:02d}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if val_loader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/miou",
                mode="max",
                patience=cfg.early_stopping_patience,
                min_delta=0.001,
                verbose=True,
            )
        )

    # --- Train ---
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        precision=cfg.precision,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=50,
    )

    trainer.fit(module, train_loader, val_loader)
    log.info(f"Training complete. Checkpoints at {output_dir / 'checkpoints'}")


@hydra.main(version_base=None, config_name="deeplab_wss_config")
def main(cfg: DictConfig) -> None:
    train_deeplab_wsss(cfg)


if __name__ == "__main__":
    main()
