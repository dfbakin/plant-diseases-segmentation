"""SPDNet Siamese classification training.

Example:
    python src/train_spdnet.py
    python src/train_spdnet.py trainer.max_epochs=5 data.batch_size=8
"""

import logging
from pathlib import Path

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
from torch.utils.data import DataLoader
from torchvision import transforms

from src.conf.spdnet import SPDNetConfig
from src.data.voc_classification import NUM_PLANTSEG_FG_CLASSES, PlantSegMCTformerDataset
from src.wsss.spdnet.dataset import SiamesePlantSegDataset, siamese_collate_fn
from src.wsss.spdnet.lightning import SPDNetModule
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="spdnet_config", node=SPDNetConfig)


def build_train_transform(image_size: int, augmentation: str = "heavy") -> transforms.Compose:
    if augmentation == "heavy":
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
    elif augmentation == "light":
        return create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=0.2,
            auto_augment="rand-m5-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.1,
            re_mode="pixel",
            re_count=1,
        )
    elif augmentation == "minimal":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    else:
        raise ValueError(f"Unknown augmentation variant: {augmentation!r}")


def build_val_transform(image_size: int) -> transforms.Compose:
    size = int((256 / 224) * image_size)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def train_spdnet(cfg: SPDNetConfig) -> float:
    """Train SPDNet. Returns best validation mAP."""
    L.seed_everything(cfg.seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.run_name:
        output_dir = Path("outputs") / cfg.experiment_name / cfg.run_name
    else:
        output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg.data
    image_size = dcfg.image_size

    aug_variant = getattr(dcfg, "augmentation", "heavy")
    train_base = PlantSegMCTformerDataset(
        root=dcfg.root,
        split=dcfg.train_split,
        image_size=image_size,
        transform=build_train_transform(image_size, augmentation=aug_variant),
        plantvillage_root=dcfg.pv_root,
        include_plantvillage=dcfg.include_plantvillage,
    )
    val_base = PlantSegMCTformerDataset(
        root=dcfg.root,
        split=dcfg.val_split,
        image_size=image_size,
        transform=build_val_transform(image_size),
        plantvillage_root=dcfg.pv_root,
        include_plantvillage=dcfg.include_plantvillage,
    )

    num_refs = getattr(dcfg, "num_references", 1)
    train_ds = SiamesePlantSegDataset(train_base, num_references=num_refs)
    val_ds = SiamesePlantSegDataset(val_base, num_references=num_refs)

    num_classes = NUM_PLANTSEG_FG_CLASSES
    log.info(f"Dataset: plantseg siamese, num_classes: {num_classes}")
    log.info(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(
        train_ds,
        batch_size=dcfg.batch_size,
        shuffle=True,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        drop_last=True,
        collate_fn=siamese_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dcfg.batch_size,
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        collate_fn=siamese_collate_fn,
    )

    scaled_lr = cfg.model.learning_rate * dcfg.batch_size / 256.0
    log.info(f"LR scaling: {cfg.model.learning_rate} * {dcfg.batch_size}/256 = {scaled_lr}")

    fusion_mode = getattr(cfg.model, "fusion_mode", "token")

    losses_cfg = getattr(cfg, "losses", None)
    online_metric: OnlineCAMIoU | None = None
    if losses_cfg is not None and bool(losses_cfg.online_loc_eval_enabled):
        try:
            online_metric = OnlineCAMIoU(
                plantseg_root=dcfg.root,
                gt_binary_dir=losses_cfg.online_loc_gt_binary_dir,
                num_classes=num_classes,
                subset_size=int(losses_cfg.online_loc_eval_subset_size),
                seed=int(losses_cfg.online_loc_eval_seed),
                every_n_epochs=int(losses_cfg.online_loc_eval_every_n_epochs),
                image_size=image_size,
                eval_batch_size=int(losses_cfg.online_loc_eval_batch_size),
                enabled=True,
            )
            log.info(
                f"OnlineCAMIoU enabled: {len(online_metric.query_names)} queries, "
                f"every {losses_cfg.online_loc_eval_every_n_epochs} epoch(s)"
            )
        except FileNotFoundError as e:
            log.warning(
                f"OnlineCAMIoU disabled (resource missing): {e}. "
                "DVC-pull the binary GT masks to enable."
            )
            online_metric = None

    module = SPDNetModule(
        num_classes=num_classes,
        fpn_channels=cfg.model.fpn_channels,
        mse_reduction=cfg.model.mse_reduction,
        pretrained=cfg.model.pretrained,
        learning_rate=scaled_lr,
        weight_decay=cfg.model.weight_decay,
        warmup_epochs=cfg.trainer.warmup_epochs,
        min_lr=cfg.trainer.min_lr,
        fusion_mode=fusion_mode,
        losses_cfg=losses_cfg,
        online_loc_metric=online_metric,
        image_size=image_size,
    )
    total_params = sum(p.numel() for p in module.parameters())
    log.info(f"Model parameters: {total_params:,}  fusion_mode={fusion_mode}")
    if losses_cfg is not None:
        log.info(
            f"Aux losses: lambda_eq={losses_cfg.lambda_eq}, "
            f"lambda_con={losses_cfg.lambda_con}, "
            f"lambda_distill={losses_cfg.lambda_distill}, "
            f"distill_warmup_epochs={losses_cfg.distill_warmup_epochs}, "
            f"con_warmup_start_epoch={losses_cfg.con_warmup_start_epoch}, "
            f"con_warmup_epochs={losses_cfg.con_warmup_epochs}"
        )

    # Optional warmstart: load *weights only* from an existing checkpoint.
    # Optimizer state, LR scheduler state, and epoch counter stay fresh --
    # so this is "start a new training session from these weights", not
    # "resume an interrupted run". Exposed via the Hydra override
    # ``+checkpoint=<path>``. Missing keys (e.g. proj_head when the source
    # run had lambda_con=0) are expected and logged, not fatal.
    # Use OmegaConf.select so the absent-field case is a clean ``None``
    # rather than a ConfigAttributeError from the structured SPDNetConfig.
    ckpt_path = OmegaConf.select(cfg, "checkpoint", default=None)
    if ckpt_path:
        ckpt_file = Path(str(ckpt_path))
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"Warmstart checkpoint not found: {ckpt_file}"
            )
        log.info(f"Warmstart: loading weights from {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(
                f"Warmstart: {len(missing)} missing keys (expected for fresh "
                f"heads such as proj_head / ema_teacher). Examples: "
                f"{missing[:3]}"
            )
        if unexpected:
            log.warning(
                f"Warmstart: {len(unexpected)} unexpected keys dropped. "
                f"Examples: {unexpected[:3]}"
            )
        log.info(
            "Warmstart done: model weights loaded; optimizer/scheduler/epoch "
            "counter remain fresh."
        )

    include_pv = getattr(dcfg, "include_plantvillage", False)
    run_name = cfg.run_name or f"spdnet_{image_size}_{cfg.seed}"
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=run_name,
        tags={
            "model": "spdnet_resnet50",
            "image_size": str(image_size),
            "num_classes": str(num_classes),
            "num_references": str(num_refs),
            "augmentation": aug_variant,
            "fusion_mode": fusion_mode,
            "include_plantvillage": str(include_pv),
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch={epoch:02d}-val_mAP={val/mAP:.4f}",
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

    log.info("Starting SPDNet training...")
    trainer.fit(module, train_loader, val_loader)

    best_mAP = float(trainer.callback_metrics.get("val/mAP", torch.tensor(0.0)))
    log.info(f"Best validation mAP: {best_mAP:.4f}")
    return best_mAP


@hydra.main(version_base=None, config_name="spdnet_config")
def main(cfg: DictConfig) -> float:
    return train_spdnet(cfg)


if __name__ == "__main__":
    main()
