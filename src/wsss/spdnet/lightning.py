"""Lightning module for SPDNet Siamese training.

Follows the same patterns as ``ClassificationModule`` but handles paired
(query, reference) batches produced by ``SiamesePlantSegDataset``.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from src.wsss.spdnet.model import SPDNet


class SPDNetModule(L.LightningModule):
    """Multi-label Siamese classification with AdamW + cosine annealing."""

    def __init__(
        self,
        num_classes: int = 115,
        fpn_channels: int = 256,
        mse_reduction: int = 4,
        pretrained: bool = True,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        min_lr: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SPDNet(
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            mse_reduction=mse_reduction,
            pretrained=pretrained,
        )
        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        mk = lambda: torchmetrics.classification.MultilabelAveragePrecision(
            num_labels=num_classes
        )
        self.train_mAP = mk()
        self.val_mAP = mk()

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor],
        return_cam: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(query, reference, return_cam=return_cam)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.model(batch["query_image"], batch["ref_images"], return_cam=False)
        loss = self.criterion(logits, batch["query_label"])

        preds = torch.sigmoid(logits.detach())
        self.train_mAP.update(preds, batch["query_label"].int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=logits.size(0))
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/mAP", self.train_mAP.compute(), prog_bar=True)
        self.train_mAP.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch["query_image"], batch["ref_images"], return_cam=False)
        loss = self.criterion(logits, batch["query_label"])

        preds = torch.sigmoid(logits.detach())
        self.val_mAP.update(preds, batch["query_label"].int())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=logits.size(0))

    def on_validation_epoch_end(self) -> None:
        self.log("val/mAP", self.val_mAP.compute(), prog_bar=True)
        self.val_mAP.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs if self.trainer else 45
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - self.warmup_epochs, 1), eta_min=self.min_lr
        )
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = cosine
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
