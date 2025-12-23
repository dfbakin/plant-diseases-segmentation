"""Training entry point.

Examples:
    python src/train.py                                # DeepLabv3+ default
    python src/train.py model=segformer                # SegFormer
    python src/train.py model=segnext trainer.max_epochs=50
    python src/train.py --multirun model=deeplabv3plus,segformer,segnext
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.conf.augmentation import get_augmentation_config
from src.conf.config import Config, register_configs
from src.data import PlantSegDataModule
from src.models import SegmentationModule, create_model
from src.training.callbacks import EarlyStoppingWithPatience, MLflowModelCheckpoint, VisualizationCallback

log = logging.getLogger(__name__)

# Register structured configs before hydra.main
register_configs()


def train(cfg: Config) -> float:
    """Run training. Returns best val mIoU."""
    L.seed_everything(cfg.experiment.seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation_name = cfg.augmentation.name
    aug_config_class = get_augmentation_config(augmentation_name)
    aug_config = aug_config_class(**{k: v for k, v in OmegaConf.to_container(cfg.augmentation).items() if k != "name"})
    train_transform = aug_config.build(
        image_size=cfg.data.image_size,
        mean=list(cfg.data.normalization.mean),
        std=list(cfg.data.normalization.std),
    )
    augmentation_preset = HydraConfig.get().runtime.choices.get("augmentation", augmentation_name)
    log.info(f"Using augmentation preset: {augmentation_preset}")

    multiclass = cfg.data.multiclass
    datamodule = PlantSegDataModule(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        mean=list(cfg.data.normalization.mean),
        std=list(cfg.data.normalization.std),
        train_transform=train_transform,
        multiclass=multiclass,
    )

    # Validate num_classes matches data mode
    expected_classes = 116 if multiclass else 2
    if cfg.model.num_classes != expected_classes:
        log.warning(
            f"Model num_classes={cfg.model.num_classes} doesn't match data mode "
            f"(multiclass={multiclass} expects {expected_classes}). Using {expected_classes}."
        )
    num_classes = expected_classes

    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=num_classes,
        encoder_name=getattr(cfg.model, "encoder_name", "resnet50"),
        encoder_weights=getattr(cfg.model, "encoder_weights", "imagenet"),
        variant=getattr(cfg.model, "variant", None),
        pretrained=getattr(cfg.model, "pretrained", True),
        decoder_channels=getattr(cfg.model, "decoder_channels", None),
    )

    module = SegmentationModule(
        model=model_backbone,
        num_classes=num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        loss_fn=cfg.model.loss_fn,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"{cfg.model.name}_{augmentation_preset}_{cfg.experiment.seed}",
        tags={
            "model": cfg.model.name,
            "encoder": getattr(cfg.model, "encoder_name", None) or getattr(cfg.model, "variant", "default"),
            "augmentation": augmentation_preset,
            "multiclass": str(multiclass),
            "num_classes": str(num_classes),
        },
    )

    # Log full Hydra config as MLflow artifact
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename="{epoch:02d}-{val/miou:.4f}",
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            save_last=cfg.trainer.checkpoint.save_last,
        ),
        EarlyStoppingWithPatience(
            monitor=cfg.trainer.early_stopping.monitor,
            patience=cfg.trainer.early_stopping.patience,
            mode=cfg.trainer.early_stopping.mode,
            min_epochs=cfg.trainer.early_stopping.min_epochs,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        VisualizationCallback(
            output_dir=output_dir / "visualizations",
            num_samples=4,
            denorm_mean=list(cfg.data.normalization.mean),
            denorm_std=list(cfg.data.normalization.std),
        ),
        MLflowModelCheckpoint(monitor=cfg.trainer.checkpoint.monitor, mode=cfg.trainer.checkpoint.mode),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
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

    log.info("Starting training...")
    trainer.fit(module, datamodule=datamodule)

    log.info("Running test evaluation...")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    return float(trainer.callback_metrics.get("val/miou", torch.tensor(0.0)))


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()
