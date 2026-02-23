"""Lightning module for WeakCLIP training."""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex

from src.wsss.weakclip.losses import cues_from_pseudo_mask, seeding_loss, stable_softmax
from src.wsss.weakclip.model import WeakCLIP


class WeakCLIPModule(L.LightningModule):
    """Lightning wrapper: seeding + identity loss, frozen backbone/text encoder."""

    def __init__(
        self,
        model: WeakCLIP,
        num_classes: int = 21,
        learning_rate: float = 2e-4,
        weight_decay: float = 3e-5,
        warmup_iters: int = 1500,
        poly_power: float = 0.9,
        total_iters: int = 80_000,
        identity_loss_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.poly_power = poly_power
        self.total_iters = total_iters
        self.identity_loss_weight = identity_loss_weight
        self.save_hyperparameters(ignore=["model"])

        self.val_miou = JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=255
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _compute_losses(
        self,
        seg_logits: torch.Tensor,
        score_map: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feat_size = tuple(self.model.decode_head.feature_size)
        logits_resized = F.interpolate(
            seg_logits,
            size=feat_size,
            mode="bilinear",
            align_corners=False,
        )
        cues = cues_from_pseudo_mask(gt_mask, self.num_classes, feat_size)
        probs = stable_softmax(logits_resized)
        loss_seeding = seeding_loss(probs, cues)

        score_scaled = score_map / self.model.tau
        score_resized = F.interpolate(
            score_scaled,
            size=feat_size,
            mode="bilinear",
            align_corners=False,
        )
        score_probs = stable_softmax(score_resized)
        loss_identity = seeding_loss(score_probs, cues)

        return {
            "loss_seeding": loss_seeding,
            "loss_identity": loss_identity * self.identity_loss_weight,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        gt_mask = batch["mask"]

        seg_logits, score_map = self(images)
        losses = self._compute_losses(seg_logits, score_map, gt_mask)

        total_loss = sum(losses.values())
        bs = images.size(0)
        self.log("train/loss", total_loss.detach(), prog_bar=True, batch_size=bs)
        self.log("train/loss_seeding", losses["loss_seeding"].detach(), batch_size=bs)
        self.log("train/loss_identity", losses["loss_identity"].detach(), batch_size=bs)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        gt_mask = batch["mask"]

        seg_logits, score_map = self(images)
        losses = self._compute_losses(seg_logits, score_map, gt_mask)
        total_loss = sum(losses.values())

        bs = images.size(0)
        self.log("val/loss", total_loss.detach(), prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(
            "val/loss_seeding", losses["loss_seeding"].detach(), batch_size=bs, sync_dist=True
        )
        self.log(
            "val/loss_identity", losses["loss_identity"].detach(), batch_size=bs, sync_dist=True
        )

        mask_size = gt_mask.shape[2:]
        preds = F.interpolate(seg_logits, size=mask_size, mode="bilinear", align_corners=False)
        pred_labels = preds.argmax(dim=1)
        gt_labels = gt_mask.squeeze(1)
        self.val_miou.update(pred_labels, gt_labels)

    def on_validation_epoch_end(self) -> None:
        miou = self.val_miou.compute()
        self.log("val/mIoU", miou, prog_bar=True)
        self.val_miou.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        frozen = {"backbone", "text_encoder"}
        trainable_params = []
        for name, param in self.model.named_parameters():
            if any(name.startswith(f) for f in frozen):
                param.requires_grad = False
            else:
                trainable_params.append(param)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def poly_lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_iters:
                return current_step / max(1, self.warmup_iters)
            progress = (current_step - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            return max(0.0, (1.0 - progress) ** self.poly_power)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
