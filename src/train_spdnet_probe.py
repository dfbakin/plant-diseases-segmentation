"""SPDNet seg-probe training entrypoint.

Trains a small ``ProbeHead`` at one of the six SPDNet probe positions,
either with the host SPDNet frozen (Phase 1) or jointly fine-tuned
(Phase 2). Loss is a multi-task combination of pixel-wise BCE+Dice on
the binary PlantSeg GT mask and (optionally) the original SPDNet
classification loss.

Skip-if-exists: if ``head.pt`` already exists in the output dir, the
script logs and returns 0 without doing any work. Pass
``resume_if_exists=false`` to force a re-train.

Example:
    python src/train_spdnet_probe.py \
        ckpt_tag=token_n1_heavy \
        checkpoint=outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
        model.position=P3_query_merged \
        model.freeze_backbone=true \
        trainer.max_epochs=5
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision

from src.conf.spdnet_probe import SPDNetProbeConfig
from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint
from src.wsss.spdnet.seg_dataset import (
    SiamesePlantSegSegDataset,
    siamese_seg_collate_fn,
)
from src.wsss.spdnet.seg_probe import (
    NEEDS_REFERENCE,
    SPDNetWithProbes,
    bce_dice_loss,
)

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="spdnet_probe_config", node=SPDNetProbeConfig)


def _binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean per-image foreground IoU on bool tensors."""
    p = pred.bool()
    t = target.bool()
    inter = (p & t).flatten(1).sum(dim=1).float()
    union = (p | t).flatten(1).sum(dim=1).float()
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class SegProbeModule(L.LightningModule):
    """Lightning module wrapping ``SPDNetWithProbes`` for multi-task training."""

    def __init__(
        self,
        checkpoint: str,
        position: str,
        num_classes: int = 115,
        fpn_channels: int = 256,
        head_hidden_dim: int = 64,
        target_size: tuple[int, int] = (448, 448),
        freeze_backbone: bool = True,
        seg_loss_weight: float = 1.0,
        cls_loss_weight: float = 0.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        head_lr: float = 1e-3,
        backbone_lr: float = 1e-5,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        spdnet = load_spdnet_from_checkpoint(
            checkpoint, num_classes=num_classes, fpn_channels=fpn_channels,
        )
        self.wrapper = SPDNetWithProbes(
            spdnet=spdnet,
            position=position,
            head_hidden_dim=head_hidden_dim,
            target_size=target_size,
            freeze_backbone=freeze_backbone,
        )
        self.cls_loss_fn = nn.MultiLabelSoftMarginLoss()
        self.val_mAP = MultilabelAveragePrecision(num_labels=num_classes)
        self._needs_ref = position in NEEDS_REFERENCE

    def forward(self, query, refs):
        return self.wrapper(query, refs, return_cls=True)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        query = batch["query_image"]
        refs = batch["ref_images"] if self._needs_ref else None
        mask = batch["query_mask"].to(query.dtype)
        labels = batch["query_label"]

        seg_logits, cls_logits = self.forward(query, refs)
        seg_loss = bce_dice_loss(
            seg_logits, mask,
            bce_weight=self.hparams.bce_weight,
            dice_weight=self.hparams.dice_weight,
        )

        if self.hparams.cls_loss_weight > 0 and not self.hparams.freeze_backbone:
            cls_loss = self.cls_loss_fn(cls_logits, labels)
        else:
            cls_loss = torch.tensor(0.0, device=seg_loss.device)

        total_loss = (
            self.hparams.seg_loss_weight * seg_loss
            + self.hparams.cls_loss_weight * cls_loss
        )

        with torch.no_grad():
            pred = (torch.sigmoid(seg_logits) > 0.5).float()
            iou = _binary_iou(pred, mask)

        bs = query.size(0)
        self.log(f"{stage}/seg_loss", seg_loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/iou_at_0.5", iou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

        if stage == "val":
            self.val_mAP.update(torch.sigmoid(cls_logits.detach()), labels.int())

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_validation_epoch_end(self):
        self.log("val/cls_mAP", self.val_mAP.compute(), prog_bar=True)
        self.val_mAP.reset()

    def configure_optimizers(self):
        if self.hparams.freeze_backbone:
            params = list(self.wrapper.head_parameters())
            optimizer = torch.optim.AdamW(
                params, lr=self.hparams.head_lr, weight_decay=self.hparams.weight_decay,
            )
        else:
            head_params = list(self.wrapper.head.parameters())
            backbone_params = [
                p for n, p in self.wrapper.spdnet.named_parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW([
                {"params": head_params, "lr": self.hparams.head_lr},
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
            ], weight_decay=self.hparams.weight_decay)
        return optimizer


def _resolve_target_size(cfg: SPDNetProbeConfig) -> tuple[int, int]:
    return (cfg.model.target_size_h, cfg.model.target_size_w)


def _output_dir(cfg: SPDNetProbeConfig) -> Path:
    """Resolve the experiment output dir.

    Phase 2 sweeps ``seg_loss_weight`` (= ``lambda`` in the plan) per
    (ckpt, position) so its outputs MUST be nested by lambda or runs
    will overwrite each other. Phase 1 keeps the flat layout for clean
    SUMMARY tables.
    """
    base = (
        cfg.output_dir
        .replace("${phase}", cfg.phase)
        .replace("${ckpt_tag}", cfg.ckpt_tag)
        .replace("${model.position}", cfg.model.position)
    )
    if cfg.phase == "phase2":
        base = f"{base}/seg{cfg.model.seg_loss_weight}_cls{cfg.model.cls_loss_weight}"
    return Path(base)


def train_probe(cfg: SPDNetProbeConfig) -> float:
    """Train one seg probe. Returns best val IoU at threshold=0.5."""
    L.seed_everything(cfg.seed, workers=True)
    log.info(f"Probe config:\n{OmegaConf.to_yaml(cfg)}")

    out_dir = _output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "head.pt"
    done_marker = out_dir / ".TRAIN_DONE"

    if cfg.resume_if_exists and done_marker.exists() and head_path.exists():
        log.info(f"Skip-if-exists: {done_marker} found, returning 0.0")
        return 0.0

    train_ds = SiamesePlantSegSegDataset(
        root=cfg.data.root,
        split=cfg.data.train_split,
        image_size=cfg.data.image_size,
        train_aug=cfg.data.train_aug,
        num_references=cfg.data.num_references,
        limit=cfg.data.limit_train,
    )
    val_ds = SiamesePlantSegSegDataset(
        root=cfg.data.root,
        split=cfg.data.val_split,
        image_size=cfg.data.image_size,
        train_aug=False,
        num_references=cfg.data.num_references,
        limit=cfg.data.limit_val,
    )
    log.info(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True, collate_fn=siamese_seg_collate_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=siamese_seg_collate_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )

    module = SegProbeModule(
        checkpoint=cfg.checkpoint,
        position=cfg.model.position,
        num_classes=cfg.model.num_classes,
        fpn_channels=cfg.model.fpn_channels,
        head_hidden_dim=cfg.model.head_hidden_dim,
        target_size=_resolve_target_size(cfg),
        freeze_backbone=cfg.model.freeze_backbone,
        seg_loss_weight=cfg.model.seg_loss_weight,
        cls_loss_weight=cfg.model.cls_loss_weight,
        bce_weight=cfg.model.bce_weight,
        dice_weight=cfg.model.dice_weight,
        head_lr=cfg.model.head_lr,
        backbone_lr=cfg.model.backbone_lr,
        weight_decay=cfg.model.weight_decay,
    )
    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    log.info(f"Trainable params: {n_trainable:,} / {n_total:,}  fusion={module.wrapper.spdnet.fusion_mode}")

    run_name = (
        cfg.run_name
        or (
            f"seg_probe_{cfg.ckpt_tag}_{cfg.model.position}_{cfg.phase}"
            f"_seg{cfg.model.seg_loss_weight}_cls{cfg.model.cls_loss_weight}"
        )
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=run_name,
        tags={
            "model": "spdnet_seg_probe",
            "ckpt_tag": cfg.ckpt_tag,
            "position": cfg.model.position,
            "phase": cfg.phase,
            "freeze_backbone": str(cfg.model.freeze_backbone),
            "seg_loss_weight": str(cfg.model.seg_loss_weight),
            "cls_loss_weight": str(cfg.model.cls_loss_weight),
            "lambda": str(cfg.model.seg_loss_weight),  # alias used by Phase 2 grid
        },
    )
    OmegaConf.save(cfg, out_dir / "config.yaml")

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="epoch={epoch:02d}-val_iou={val/iou_at_0.5:.4f}",
            monitor="val/iou_at_0.5",
            mode="max",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
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
        default_root_dir=str(out_dir),
        enable_progress_bar=True,
    )
    trainer.fit(module, train_loader, val_loader)

    best_iou = float(trainer.callback_metrics.get("val/iou_at_0.5", torch.tensor(0.0)))
    log.info(f"Best val IoU @ 0.5: {best_iou:.4f}")

    torch.save({
        "head_state_dict": module.wrapper.head.state_dict(),
        "position": cfg.model.position,
        "ckpt_tag": cfg.ckpt_tag,
        "freeze_backbone": cfg.model.freeze_backbone,
        "best_val_iou": best_iou,
        "head_hidden_dim": cfg.model.head_hidden_dim,
    }, head_path)
    log.info(f"Saved head to {head_path}")

    if not cfg.model.freeze_backbone:
        full_path = out_dir / "spdnet_finetuned.pt"
        torch.save({
            "spdnet_state_dict": module.wrapper.spdnet.state_dict(),
            "fusion_mode": module.wrapper.spdnet.fusion_mode,
        }, full_path)
        log.info(f"Saved fine-tuned SPDNet to {full_path}")

    done_marker.touch()
    return best_iou


@hydra.main(version_base=None, config_name="spdnet_probe_config")
def main(cfg: DictConfig) -> float:
    return train_probe(cfg)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
