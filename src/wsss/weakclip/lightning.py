"""Lightning module for WeakCLIP training.

Losses match the original WeakCLIP repo:
- Decode head: seeding_loss + CRF boundary loss on seg_logits
- Identity head: cross-entropy on score_map/tau (not seeding_loss)
"""

import logging
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex

from src.wsss.weakclip.losses import (
    crf_boundary_loss,
    cues_from_pseudo_mask,
    seeding_loss,
    stable_softmax,
)
from src.wsss.weakclip.model import WeakCLIP

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize_batch(images: torch.Tensor) -> np.ndarray:
    """Convert normalized (B,3,H,W) tensor to uint8 (B,H,W,3) numpy for CRF."""
    imgs = images.detach().cpu().float().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = imgs * IMAGENET_STD[None, None, None, :] + IMAGENET_MEAN[None, None, None, :]
    imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(imgs)


def _run_crf_on_batch(images_np: np.ndarray, probs_np: np.ndarray, t: int = 10) -> np.ndarray:
    """Run DenseCRF on each image in the batch. Matches original dgcn_crf_operation."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    from scipy.ndimage import zoom

    B, C, h, w = probs_np.shape
    probs_np = probs_np.clip(1e-4, None)

    result = np.zeros_like(probs_np)
    for i in range(B):
        img = images_np[i]
        img_h, img_w = img.shape[:2]
        img_resized = zoom(img, (h / img_h, w / img_w, 1.0), order=1).astype(np.uint8)
        img_resized = np.ascontiguousarray(img_resized)

        d = dcrf.DenseCRF2D(w, h, C)
        unary = unary_from_softmax(probs_np[i])
        d.setUnaryEnergy(np.ascontiguousarray(unary))
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_resized, compat=10)
        q = d.inference(t)
        result[i] = np.array(q).reshape(C, h, w)

    result = result.clip(1e-4, None)
    result = result / result.sum(axis=1, keepdims=True)
    return np.log(result)


class WeakCLIPModule(L.LightningModule):
    """Lightning wrapper matching the original WeakCLIP training losses.

    Decode head: seeding_loss + crf_boundary_loss on FPN seg_logits.
    Identity head: cross-entropy on score_map/tau vs gt mask (like mmseg IdentityHead).
    """

    def __init__(
        self,
        model: WeakCLIP,
        num_classes: int = 21,
        learning_rate: float = 1e-4,
        weight_decay: float = 3e-5,
        warmup_iters: int = 1500,
        total_iters: int = 20_000,
        min_lr: float = 1e-6,
        identity_loss_weight: float = 0.4,
        use_crf_loss: bool = True,
        crf_iters: int = 10,
        norm_eval: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.identity_loss_weight = identity_loss_weight
        self.use_crf_loss = use_crf_loss
        self.crf_iters = crf_iters
        self.norm_eval = norm_eval
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
        images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat_size = tuple(self.model.decode_head.feature_size)

        logits_resized = F.interpolate(
            seg_logits, size=feat_size, mode="bilinear", align_corners=False,
        )
        cues = cues_from_pseudo_mask(gt_mask, self.num_classes, feat_size)
        probs = stable_softmax(logits_resized)
        loss_seeding = seeding_loss(probs, cues)

        losses = {"loss_seeding": loss_seeding}

        if self.use_crf_loss and images is not None and self.training:
            images_np = _denormalize_batch(images)
            probs_np = probs.detach().cpu().numpy()
            crf_result = _run_crf_on_batch(images_np, probs_np, t=self.crf_iters)
            losses["loss_boundary"] = crf_boundary_loss(probs, crf_result)

        # Identity loss: standard cross-entropy on score_map/tau vs gt mask
        # Matches original IdentityHead with CrossEntropyLoss(ignore_index=255)
        mask_size = gt_mask.shape[2:]
        score_scaled = score_map / self.model.tau
        score_resized = F.interpolate(
            score_scaled, size=mask_size, mode="bilinear", align_corners=False,
        )
        gt_labels = gt_mask.squeeze(1).long()
        loss_identity = F.cross_entropy(
            score_resized, gt_labels, ignore_index=255
        )
        losses["loss_identity"] = loss_identity * self.identity_loss_weight

        return losses

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        gt_mask = batch["mask"]

        seg_logits, score_map = self(images)
        losses = self._compute_losses(seg_logits, score_map, gt_mask, images=images)

        total_loss = sum(losses.values())
        bs = images.size(0)
        self.log("train/loss", total_loss.detach(), prog_bar=True, batch_size=bs)
        for k, v in losses.items():
            self.log(f"train/{k}", v.detach(), batch_size=bs)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        gt_mask = batch["mask"]

        seg_logits, score_map = self(images)
        losses = self._compute_losses(seg_logits, score_map, gt_mask, images=None)
        total_loss = sum(losses.values())

        bs = images.size(0)
        self.log("val/loss", total_loss.detach(), prog_bar=True, batch_size=bs, sync_dist=True)
        for k, v in losses.items():
            self.log(f"val/{k}", v.detach(), batch_size=bs, sync_dist=True)

        mask_size = gt_mask.shape[2:]
        preds = F.interpolate(seg_logits, size=mask_size, mode="bilinear", align_corners=False)
        pred_labels = preds.argmax(dim=1)
        gt_labels = gt_mask.squeeze(1)
        self.val_miou.update(pred_labels, gt_labels)

    def on_validation_epoch_end(self) -> None:
        miou = self.val_miou.compute()
        self.log("val/mIoU", miou, prog_bar=True)
        self.val_miou.reset()

    def _freeze_bn(self) -> None:
        """Freeze BN layers only in backbone and text_encoder.

        Matches the original WeakCLIP `backbone.norm_eval=True`: only the
        pretrained backbone/text_encoder BN stats stay fixed.  The neck, decode
        head, and context decoder BN layers remain in train mode so their
        running statistics and affine parameters can be learned.
        """
        frozen_modules = [self.model.backbone, self.model.text_encoder]
        for parent in frozen_modules:
            for m in parent.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def on_train_start(self) -> None:
        if self.norm_eval:
            self._freeze_bn()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.norm_eval:
            self._freeze_bn()

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

        # CosineAnnealing with linear warmup, matching the original schedule_20k.py
        def cosine_with_warmup(current_step: int) -> float:
            if current_step < self.warmup_iters:
                return max(1e-6, current_step / max(1, self.warmup_iters))
            progress = (current_step - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
            min_ratio = self.min_lr / self.learning_rate
            return max(min_ratio, cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=cosine_with_warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
