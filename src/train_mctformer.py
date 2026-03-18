"""MCTformer-V2 multi-label classification training.

Supports PASCAL VOC 2012 (default) and PlantSeg datasets via ``dataset``
config switch.

Examples:
    python src/train_mctformer.py
    python src/train_mctformer.py dataset=plantseg data.root=data/plantsegv3
    python src/train_mctformer.py model.checkpoint_path=pretrained/MCTformerV2.pth
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.conf.classifier import MCTformerModelConfig, VOCDataConfig
from src.data.voc_classification import (
    NUM_PLANTSEG_FG_CLASSES,
    BinaryPlantDataset,
    PlantSegMCTformerDataset,
    VOCClassificationDataset,
)
from src.models.classification import ClassificationModule
from src.models.classifier_factory import create_classifier

log = logging.getLogger(__name__)

NUM_VOC_CLASSES = 20


@dataclass
class PlantSegDataConfig:
    root: str = "data/plantsegv3"
    pv_root: str = "data/plant-village"
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 448
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class MCTformerTrainerConfig:
    max_epochs: int = 45
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0


@dataclass
class MCTformerConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    dataset: str = "voc"
    experiment_name: str = "mctformer_voc"
    seed: int = 0

    model: MCTformerModelConfig = field(default_factory=MCTformerModelConfig)
    data: VOCDataConfig = field(default_factory=VOCDataConfig)
    plantseg_data: PlantSegDataConfig = field(default_factory=PlantSegDataConfig)
    trainer: MCTformerTrainerConfig = field(default_factory=MCTformerTrainerConfig)

    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "${experiment_name}"
    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"


cs = ConfigStore.instance()
cs.store(name="mctformer_config", node=MCTformerConfig)


def build_train_transform(image_size: int) -> transforms.Compose:
    """Match original MCTformer: timm create_transform with RandAugment + RandomErasing."""
    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )


def build_val_transform(image_size: int) -> transforms.Compose:
    size = int((256 / 224) * image_size)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def _build_datasets(
    cfg: MCTformerConfig,
) -> tuple[Dataset, Dataset, int]:
    """Build train/val datasets and return (train_ds, val_ds, num_classes)."""
    if cfg.dataset == "plantseg_binary":
        dcfg = cfg.plantseg_data
        image_size = dcfg.image_size
        train_ds = BinaryPlantDataset(
            plantseg_root=dcfg.root,
            plantvillage_root=dcfg.pv_root,
            split=dcfg.train_split,
            image_size=image_size,
            transform=build_train_transform(image_size),
            include_plantvillage=True,
        )
        val_ds = BinaryPlantDataset(
            plantseg_root=dcfg.root,
            plantvillage_root=dcfg.pv_root,
            split=dcfg.val_split,
            image_size=image_size,
            transform=build_val_transform(image_size),
            include_plantvillage=True,
        )
        return train_ds, val_ds, 1
    elif cfg.dataset in ("plantseg", "plantseg_with_pv"):
        dcfg = cfg.plantseg_data
        image_size = dcfg.image_size
        include_pv = cfg.dataset == "plantseg_with_pv"
        train_ds = PlantSegMCTformerDataset(
            root=dcfg.root,
            split=dcfg.train_split,
            image_size=image_size,
            transform=build_train_transform(image_size),
            plantvillage_root=dcfg.pv_root,
            include_plantvillage=include_pv,
        )
        val_ds = PlantSegMCTformerDataset(
            root=dcfg.root,
            split=dcfg.val_split,
            image_size=image_size,
            transform=build_val_transform(image_size),
            plantvillage_root=dcfg.pv_root,
            include_plantvillage=include_pv,
        )
        return train_ds, val_ds, NUM_PLANTSEG_FG_CLASSES
    else:
        dcfg = cfg.data
        image_size = dcfg.image_size
        train_ds = VOCClassificationDataset(
            root=dcfg.root,
            split=dcfg.train_split,
            image_size=image_size,
            transform=build_train_transform(image_size),
        )
        val_ds = VOCClassificationDataset(
            root=dcfg.root,
            split=dcfg.val_split,
            image_size=image_size,
            transform=build_val_transform(image_size),
        )
        return train_ds, val_ds, NUM_VOC_CLASSES


def train_mctformer(cfg: MCTformerConfig) -> float:
    """Train MCTformer-V2. Returns best validation mAP."""
    L.seed_everything(cfg.seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes = _build_datasets(cfg)
    if cfg.model.num_classes != num_classes:
        log.info(f"Overriding model.num_classes: {cfg.model.num_classes} -> {num_classes}")
        cfg.model.num_classes = num_classes
    log.info(f"Dataset: {cfg.dataset}, num_classes: {num_classes}")
    log.info(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    dcfg = cfg.plantseg_data if cfg.dataset in ("plantseg", "plantseg_with_pv", "plantseg_binary") else cfg.data
    train_loader = DataLoader(
        train_ds,
        batch_size=dcfg.batch_size,
        shuffle=True,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(1.5 * dcfg.batch_size),
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
    )

    scaled_lr = cfg.model.learning_rate * dcfg.batch_size / 512.0
    log.info(f"LR scaling: {cfg.model.learning_rate} * {dcfg.batch_size}/512 = {scaled_lr}")

    backbone = create_classifier(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        checkpoint_path=cfg.model.checkpoint_path,
        input_size=cfg.model.input_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )

    module = ClassificationModule(
        model=backbone,
        num_classes=cfg.model.num_classes,
        learning_rate=scaled_lr,
        weight_decay=cfg.model.weight_decay,
        label_smoothing=cfg.model.label_smoothing,
        multi_label=True,
        warmup_epochs=5,
        min_lr=1e-5,
    )
    log.info(
        f"Model: {cfg.model.name}, pretrained={cfg.model.pretrained}, "
        f"checkpoint={cfg.model.checkpoint_path}"
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"mctformer_{dcfg.image_size}_{cfg.seed}",
        tags={
            "model": cfg.model.name,
            "image_size": str(dcfg.image_size),
            "num_classes": str(cfg.model.num_classes),
            "dataset": cfg.dataset,
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{val/mAP:.4f}",
            monitor="val/mAP",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val or None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    log.info("Starting MCTformer training...")
    trainer.fit(module, train_loader, val_loader)

    best_mAP = float(trainer.callback_metrics.get("val/mAP", torch.tensor(0.0)))
    log.info(f"Best validation mAP: {best_mAP:.4f}")
    return best_mAP


@hydra.main(version_base=None, config_name="mctformer_config")
def main(cfg: DictConfig) -> float:
    return train_mctformer(cfg)


if __name__ == "__main__":
    main()
