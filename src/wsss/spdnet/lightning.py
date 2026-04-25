"""Lightning module for SPDNet Siamese training.

Follows the same patterns as ``ClassificationModule`` but handles paired
(query, reference) batches produced by ``SiamesePlantSegDataset``.

Optionally adds the three auxiliary spatial losses (equivariance, patch
contrastive, self-distillation) plus an online localization metric when
``losses_cfg`` is supplied; passing ``losses_cfg=None`` (the default)
preserves the exact pre-aux-loss baseline behaviour.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.wsss.spdnet import equivariance_transforms as ET
from src.wsss.spdnet.model import SPDNet
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU
from src.wsss.spdnet.spatial_losses import (
    EMATeacher,
    ProjectionHead,
    equivariance_loss,
    patch_contrastive_loss,
    self_distillation_loss,
)


class SPDNetModule(L.LightningModule):
    """Multi-label Siamese classification with AdamW + cosine annealing.

    When ``losses_cfg`` is provided with non-zero loss weights:

    * ``lambda_eq > 0``: also computes equivariance loss
      :math:`L_{eq} = \\mathrm{MSE}(M(T(q), r), T(M(q, r)))` -- adds one
      extra forward pass for ``T(q)``.
    * ``lambda_con > 0``: also computes patch-level supervised contrastive
      loss on ``P3_query_merged``; instantiates a 1x1 ``ProjectionHead``.
    * ``lambda_distill > 0``: also computes DINO-style KL distillation
      against an EMA teacher of the student. EMA teacher is created at
      ``__init__`` and updated after every optimizer step. A
      ``distill_center`` buffer of shape ``(P,)`` for ``P =
      (image_size // 4) ** 2`` (the merged FPN spatial size for the
      default ResNet50+FPN) is registered for DINO centring.

    When ``online_loc_metric`` is supplied, the metric is run in
    :meth:`on_validation_epoch_end` and the three scalars
    ``val/cam_iou_best``, ``val/cam_iou_best_thr``, ``val/cam_iou_auc``
    are logged.
    """

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
        fusion_mode: str = "token",
        losses_cfg: SPDNetSpatialLossesConfig | None = None,
        online_loc_metric: OnlineCAMIoU | None = None,
        image_size: int = 448,
    ) -> None:
        super().__init__()
        # OnlineCAMIoU + losses_cfg are constructed externally and may not be
        # picklable into hyperparameters.
        self.save_hyperparameters(ignore=["online_loc_metric", "losses_cfg"])
        self.model = SPDNet(
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            mse_reduction=mse_reduction,
            pretrained=pretrained,
            fusion_mode=fusion_mode,
        )
        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.image_size = image_size

        mk = lambda: torchmetrics.classification.MultilabelAveragePrecision(
            num_labels=num_classes
        )
        self.train_mAP = mk()
        self.val_mAP = mk()

        # ------------------------ aux losses --------------------------
        self.losses_cfg = losses_cfg or SPDNetSpatialLossesConfig()
        self._spatial_fusion = (fusion_mode == "spatial")

        # Equivariance loss state: validate transform IDs at __init__.
        for t_id in self.losses_cfg.equivariance_transforms:
            if t_id not in ET.ALL_TRANSFORMS:
                raise ValueError(
                    f"Unknown transform id {t_id} in equivariance_transforms; "
                    f"valid IDs are {sorted(ET.ALL_TRANSFORMS)}"
                )

        # Patch-contrastive projector (only allocated when needed).
        if self.losses_cfg.lambda_con > 0:
            self.proj_head: ProjectionHead | None = ProjectionHead(
                in_channels=fpn_channels,
                out_channels=self.losses_cfg.con_projection_dim,
            )
        else:
            self.proj_head = None

        # Self-distillation EMA teacher + DINO centre buffer.
        if self.losses_cfg.lambda_distill > 0:
            self.ema_teacher: EMATeacher | None = EMATeacher(
                self.model, alpha=self.losses_cfg.ema_alpha,
            )
            # ResNet50 stem (/4) + layer1 keeps res, FPN merges to layer1's
            # resolution -> P3_query_merged is at /4. P = (image_size/4) ** 2.
            P_size = (self.image_size // 4) ** 2
            self.register_buffer("distill_center", torch.zeros(P_size))
        else:
            self.ema_teacher = None
            self.distill_center = None  # type: ignore[assignment]

        self.online_loc_metric = online_loc_metric

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor],
        return_cam: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(query, reference, return_cam=return_cam)

    # ------------------------------------------------------------------
    # Loss-weight schedules
    # ------------------------------------------------------------------

    def effective_lambda_con(self, epoch: int | None = None) -> float:
        """Return the linearly-warmed-up contrastive loss weight for ``epoch``.

        Mirrors ``LossesConfig.con_warmup_*`` documentation. ``epoch=None``
        uses ``self.current_epoch`` (the canonical training-step entry point).
        Exposed on the module for unit-testability.
        """
        base = float(self.losses_cfg.lambda_con)
        if base <= 0.0:
            return 0.0
        e = int(self.current_epoch if epoch is None else epoch)
        start = int(self.losses_cfg.con_warmup_start_epoch)
        ramp = int(self.losses_cfg.con_warmup_epochs)
        if e < start:
            return 0.0
        if ramp <= 0 or e >= start + ramp:
            return base
        return base * float(e - start) / float(ramp)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        q = batch["query_image"]
        refs = batch["ref_images"]
        labels = batch["query_label"]
        B = q.size(0)

        # Single forward through the student. Request the attention map only
        # when we actually need it (spatial fusion + lambda_eq > 0); this
        # avoids the dense need_weights=True path on the baseline run.
        want_attn = self._spatial_fusion and self.losses_cfg.lambda_eq > 0
        feats = self.model.extract_merged_features(q, refs, return_attn=want_attn)
        fused = feats["fused"]
        pooled = fused.mean(dim=[2, 3])
        logits = self.model.classifier(pooled)
        L_cls = self.criterion(logits, labels)

        total = L_cls
        comp: dict[str, torch.Tensor] = {"L_cls": L_cls.detach()}

        # ------------------ Equivariance ------------------
        if want_attn:
            attn_orig = feats["attn_map"]
            # Round-robin transform per batch (deterministic, batch-uniform
            # so the per-step compute cost is constant).
            t_choices = self.losses_cfg.equivariance_transforms
            if len(t_choices) > 0:
                t_id = int(t_choices[batch_idx % len(t_choices)])
                q_aug = ET.apply(q, t_id)
                attn_aug = self.model.attention_map(
                    q_aug,
                    ref_merged_cached=feats.get("ref_merged"),
                )
                L_eq = equivariance_loss(attn_orig, attn_aug, t_id)
                total = total + self.losses_cfg.lambda_eq * L_eq
                comp["L_eq"] = L_eq.detach()

        # ------------------ Patch contrastive ------------------
        if self.proj_head is not None:
            lam_con_eff = self.effective_lambda_con()
            # Always log the schedule so MLflow can plot the ramp next to
            # L_con itself. Use a 0-d tensor on the same device as the rest
            # so Lightning's log() doesn't trigger a host->device copy.
            comp["lambda_con_eff"] = torch.tensor(
                lam_con_eff, device=fused.device, dtype=fused.dtype,
            )
            if lam_con_eff > 0.0:
                L_con = patch_contrastive_loss(
                    p3_query=feats["query_merged"],
                    p4_fused=fused,
                    cls_weight=self.model.classifier.weight,
                    labels=labels,
                    proj_head=self.proj_head,
                    top_k=self.losses_cfg.con_top_K,
                    m_negatives=self.losses_cfg.con_M_negatives,
                    temperature=self.losses_cfg.con_temperature,
                )
                total = total + lam_con_eff * L_con
                comp["L_con"] = L_con.detach()

        # ------------------ Self-distillation ------------------
        warmup_done = (
            self.ema_teacher is not None
            and self.current_epoch >= self.losses_cfg.distill_warmup_epochs
        )
        if warmup_done:
            S_student = torch.einsum(
                "nc,bchw->bnhw", self.model.classifier.weight, fused,
            )
            S_teacher = self.ema_teacher(q, refs)  # type: ignore[union-attr]
            P_actual = S_student.shape[-1] * S_student.shape[-2]
            # Resize the center buffer if the model's spatial resolution
            # disagrees with our (image_size // 4) ** 2 estimate (e.g.
            # variable input crops). One-time silent reallocation.
            if self.distill_center is None or self.distill_center.numel() != P_actual:
                self.distill_center = torch.zeros(
                    P_actual, device=S_student.device, dtype=S_student.dtype,
                )
            L_dist = self_distillation_loss(
                s_student=S_student,
                s_teacher=S_teacher,
                labels=labels,
                center=self.distill_center,
                center_beta=self.losses_cfg.distill_center_beta,
                T_teacher=self.losses_cfg.distill_T_teacher,
                T_student=self.losses_cfg.distill_T_student,
            )
            total = total + self.losses_cfg.lambda_distill * L_dist
            comp["L_dist"] = L_dist.detach()

        # ------------------ Logging + mAP ------------------
        preds = torch.sigmoid(logits.detach())
        self.train_mAP.update(preds, labels.int())
        self.log(
            "train/loss", total,
            prog_bar=True, on_step=True, on_epoch=True, batch_size=B,
        )
        for name, val in comp.items():
            self.log(
                f"train/{name}", val,
                prog_bar=False, on_step=True, on_epoch=True, batch_size=B,
            )
        return total

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        """EMA teacher update lives here so the teacher tracks the
        post-optimizer-step student parameters (classic mean-teacher schedule)."""
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.model)

    def on_train_epoch_end(self) -> None:
        self.log("train/mAP", self.train_mAP.compute(), prog_bar=True)
        self.train_mAP.reset()

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch["query_image"], batch["ref_images"], return_cam=False)
        loss = self.criterion(logits, batch["query_label"])
        preds = torch.sigmoid(logits.detach())
        self.val_mAP.update(preds, batch["query_label"].int())
        self.log(
            "val/loss", loss,
            prog_bar=True, on_epoch=True, batch_size=logits.size(0),
        )

    def on_validation_epoch_end(self) -> None:
        self.log("val/mAP", self.val_mAP.compute(), prog_bar=True)
        self.val_mAP.reset()

        if (
            self.online_loc_metric is not None
            and self.online_loc_metric.should_run(self.current_epoch)
            and not self.trainer.sanity_checking  # skip the sanity-check run
        ):
            scalars = self.online_loc_metric.evaluate(self.model, self.device)
            for k, v in scalars.items():
                self.log(
                    f"val/{k}", float(v),
                    prog_bar=False, on_epoch=True,
                )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        # Skip frozen params (EMA teacher) -- AdamW would otherwise allocate
        # state for them even though their grads are always None.
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
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
