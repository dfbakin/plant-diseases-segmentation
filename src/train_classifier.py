"""Classification training entry point.

Examples:
    python src/train_classifier.py model.name=resnet50
    python src/train_classifier.py model.name=efficientnet_b4 data.image_size=380
    python src/train_classifier.py --multirun model.name=resnet18,resnet50,efficientnet_b4
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.conf.augmentation import get_augmentation_config
from src.conf.classifier import ClassifierConfig, register_classifier_configs
from src.data import (
    NUM_CLASSIFICATION_CLASSES,
    PlantSegClassificationDataset,
    PlantVillageDataset,
    get_combined_sample_weights,
)
from src.models.classification import ClassificationModule
from src.models.classifier_factory import create_classifier

log = logging.getLogger(__name__)

register_classifier_configs()


def train_classifier(cfg: ClassifierConfig) -> float:
    """Train a classifier. Returns best validation accuracy."""
    L.seed_everything(cfg.seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build augmentation
    aug_config_class = get_augmentation_config(cfg.augmentation.name)
    aug_config = aug_config_class(
        **{k: v for k, v in OmegaConf.to_container(cfg.augmentation).items() if k != "name"}
    )
    transform = aug_config.build(
        image_size=cfg.data.image_size,
        mean=list(cfg.data.mean),
        std=list(cfg.data.std),
    )
    log.info(f"Using augmentation: {cfg.augmentation.name}")

    # Create datasets
    pv_train = PlantVillageDataset(
        cfg.data.plantvillage_root,
        split="train",
        transform=transform,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
    )
    pv_val = PlantVillageDataset(
        cfg.data.plantvillage_root,
        split="val",
        transform=transform,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
    )
    ps_train = PlantSegClassificationDataset(
        cfg.data.plantseg_root,
        split="train",
        transform=transform,
        return_mask=False,
    )
    ps_val = PlantSegClassificationDataset(
        cfg.data.plantseg_root,
        split="val",
        transform=transform,
        return_mask=cfg.cam.enabled,
    )

    # Limit samples for quick testing
    if cfg.data.max_samples is not None:
        n = cfg.data.max_samples
        log.info(f"Limiting datasets to {n} samples each (testing mode)")
        pv_train = Subset(pv_train, range(min(n, len(pv_train))))
        pv_val = Subset(pv_val, range(min(n, len(pv_val))))
        ps_train = Subset(ps_train, range(min(n, len(ps_train))))
        ps_val = Subset(ps_val, range(min(n, len(ps_val))))

    combined_train = ConcatDataset([pv_train, ps_train])
    log.info(f"Train: PlantVillage={len(pv_train)}, PlantSeg={len(ps_train)}, Total={len(combined_train)}")
    log.info(f"Val: PlantVillage={len(pv_val)}, PlantSeg={len(ps_val)}")

    # Create dataloaders
    use_weighted = cfg.data.use_weighted_sampler and cfg.data.max_samples is None
    if use_weighted:
        # Weighted sampler only works with full datasets (not Subset)
        sample_weights = get_combined_sample_weights(pv_train, ps_train)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(combined_train),
            replacement=True,
        )
        train_loader = DataLoader(
            combined_train,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        train_loader = DataLoader(
            combined_train,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

    val_loader = DataLoader(
        ps_val,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Create model
    backbone = create_classifier(
        name=cfg.model.name,
        num_classes=NUM_CLASSIFICATION_CLASSES,
        pretrained=cfg.model.pretrained,
    )

    module = ClassificationModule(
        model=backbone,
        num_classes=NUM_CLASSIFICATION_CLASSES,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        label_smoothing=cfg.model.label_smoothing,
        model_name=cfg.model.name if cfg.cam.enabled else None,
        enable_cam_eval=cfg.cam.enabled,
        cam_method=cfg.cam.method,
        cam_threshold=cfg.cam.threshold,
    )
    log.info(f"Model: {cfg.model.name}, pretrained={cfg.model.pretrained}")
    log.info(f"CAM evaluation: {cfg.cam.enabled}")

    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"{cfg.model.name}_{cfg.augmentation.name}_{cfg.seed}",
        tags={
            "model": cfg.model.name,
            "pretrained": str(cfg.model.pretrained),
            "augmentation": cfg.augmentation.name,
            "image_size": str(cfg.data.image_size),
            "num_classes": str(NUM_CLASSIFICATION_CLASSES),
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/acc",
            patience=cfg.trainer.early_stopping_patience,
            mode="max",
            min_delta=0.001,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    log.info("Starting training...")
    trainer.fit(module, train_loader, val_loader)

    log.info("Running test evaluation...")
    ps_test = PlantSegClassificationDataset(
        cfg.data.plantseg_root,
        split="test",
        transform=transform,
        return_mask=cfg.cam.enabled,
    )
    if cfg.data.max_samples is not None:
        ps_test = Subset(ps_test, range(min(cfg.data.max_samples, len(ps_test))))
    test_loader = DataLoader(
        ps_test,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    trainer.test(module, test_loader, ckpt_path="best")

    best_acc = float(trainer.callback_metrics.get("val/acc", torch.tensor(0.0)))
    log.info(f"Best validation accuracy: {best_acc:.4f}")

    return best_acc


@hydra.main(version_base=None, config_name="classifier_config")
def main(cfg: DictConfig) -> float:
    return train_classifier(cfg)


if __name__ == "__main__":
    main()
