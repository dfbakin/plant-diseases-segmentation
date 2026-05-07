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
    attention_concentration_loss,
    attention_marginal_entropy_loss,
    cam_pseudo_mask_loss,
    equivariance_loss,
    patch_contrastive_loss,
    self_distillation_loss,
)


def _warmup_schedule(
    base: float, start: int, ramp: int, current: int,
) -> float:
    """Linear warmup with explicit start epoch.

    Returns ``0`` for ``current < start`` and before any positive ``base``;
    linearly interpolates up to ``base`` over ``ramp`` epochs starting at
    ``start``, then stays at ``base`` afterwards. ``ramp <= 0`` means "jump
    to ``base`` at ``start``".
    """
    if base <= 0.0:
        return 0.0
    if current < start:
        return 0.0
    if ramp <= 0 or current >= start + ramp:
        return base
    return base * float(current - start) / float(ramp)


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
        ref_pool_size: int = 14,
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
            ref_pool_size=ref_pool_size,
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
        return _warmup_schedule(
            base=float(self.losses_cfg.lambda_con),
            start=int(self.losses_cfg.con_warmup_start_epoch),
            ramp=int(self.losses_cfg.con_warmup_epochs),
            current=int(self.current_epoch if epoch is None else epoch),
        )

    def effective_lambda_mask(self, epoch: int | None = None) -> float:
        """Return the linearly-warmed-up pseudo-mask loss weight.

        Same shape as ``effective_lambda_con`` but keyed off
        ``mask_warmup_start_epoch`` / ``mask_warmup_epochs``. Both defaults
        are zero so ``lambda_mask`` applies in full from epoch 0 on; set
        ``mask_warmup_*`` to ramp the supervision gradually on top of a
        warmstart.
        """
        return _warmup_schedule(
            base=float(self.losses_cfg.lambda_mask),
            start=int(self.losses_cfg.mask_warmup_start_epoch),
            ramp=int(self.losses_cfg.mask_warmup_epochs),
            current=int(self.current_epoch if epoch is None else epoch),
        )

    def effective_lambda_ac(self, epoch: int | None = None) -> float:
        """Return the linearly-warmed-up attention-concentration weight.

        Same shape as ``effective_lambda_mask`` but keyed off
        ``ac_warmup_start_epoch`` / ``ac_warmup_epochs``. Both defaults are
        zero so ``lambda_ac`` applies in full from epoch 0 (existing recipes
        are unchanged). Set non-zero to delay L_ac introduction until the
        classifier has built a discriminative spatial signal -- this avoids
        the epoch-3 attention collapse observed in cold-start runs where L_ac
        is applied on top of random MSE logits (the 2026-04-30 highres run
        saturated attn_mean to 0.98 within 3 epochs and pinned val/cam_iou
        at 0.198 for the remaining run).
        """
        return _warmup_schedule(
            base=float(self.losses_cfg.lambda_ac),
            start=int(self.losses_cfg.ac_warmup_start_epoch),
            ramp=int(self.losses_cfg.ac_warmup_epochs),
            current=int(self.current_epoch if epoch is None else epoch),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        q = batch["query_image"]
        refs = batch["ref_images"]
        labels = batch["query_label"]
        B = q.size(0)

        # Single forward through the student. Request the attention map only
        # when we actually need it (spatial fusion + any of: lambda_eq,
        # lambda_ac, lambda_marg_H, or the always-on log_attn_stats
        # diagnostic). This avoids the dense need_weights=True path on the
        # baseline run.
        want_attn = self._spatial_fusion and (
            self.losses_cfg.lambda_eq > 0
            or self.losses_cfg.lambda_ac > 0
            or self.losses_cfg.lambda_marg_H > 0
            or self.losses_cfg.log_attn_stats
        )
        feats = self.model.extract_merged_features(q, refs, return_attn=want_attn)
        fused = feats["fused"]
        pooled = fused.mean(dim=[2, 3])
        logits = self.model.classifier(pooled)
        L_cls = self.criterion(logits, labels)

        total = L_cls
        comp: dict[str, torch.Tensor] = {"L_cls": L_cls.detach()}

        # ------------------ Equivariance (+ attention concentration) ------------------
        if want_attn:
            attn_orig = feats["attn_map"]
            if self.losses_cfg.log_attn_stats:
                # Always-on attention diagnostics. Detached so they never
                # contribute to the training graph (compute is the cost of
                # three reductions over the per-query concentration map,
                # ``< 1ms`` at 224x224). Kept here -- inside ``if want_attn``
                # but BEFORE any loss-coefficient branch -- so the stats
                # are emitted whether or not L_ac / L_eq / L_marg_H is on,
                # giving an apples-to-apples baseline against later
                # aux-loss runs (see §5.14.6 in RESEARCH_CONTEXT.md).
                #
                # ``attn_mean`` mirrors L_ac's negation; ``attn_std``
                # catches the uniform-attention failure mode that
                # ``attn_mean`` alone misses; ``attn_p99`` rises sooner
                # than the mean once any subset of queries pins on a
                # single key. The L_ac branch also writes ``attn_mean``
                # on its own; Lightning de-dupes by metric name on the
                # same step so the duplicate write is harmless.
                a = attn_orig.detach()
                comp["attn_mean"] = a.mean()
                comp["attn_std"] = a.std()
                comp["attn_p99"] = a.flatten().quantile(0.99)
            if self.losses_cfg.lambda_eq > 0:
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
            if self.losses_cfg.lambda_ac > 0:
                # D1: push attn_map away from its uniform fixed point. The
                # gradient flows through ``attn_orig`` back into the SCA
                # module, so it's essential that ``attn_orig`` was computed
                # in the current forward pass (which it is above).
                #
                # The schedule ``lam_ac_eff = effective_lambda_ac()`` linearly
                # ramps L_ac's contribution (default: full weight from epoch
                # 0, matching legacy D1 recipes). Setting ``ac_warmup_*`` on
                # cold-start runs is now the recommended default because L_ac
                # has a trivial minimum at attn_map == 1 everywhere that is
                # otherwise reached in 2-3 epochs before the classifier has
                # built useful spatial features.
                lam_ac_eff = self.effective_lambda_ac()
                comp["lambda_ac_eff"] = torch.tensor(
                    lam_ac_eff, device=fused.device, dtype=fused.dtype,
                )
                L_ac = attention_concentration_loss(attn_orig)
                if lam_ac_eff > 0.0:
                    total = total + lam_ac_eff * L_ac
                comp["L_ac"] = L_ac.detach()
                # Always-logged diagnostic: mean concentration is the
                # positive-valued mirror of L_ac. Tracked even during L_ac
                # warmup (when lam_ac_eff == 0) so operators can watch for
                # the attn_mean > 0.95 collapse signature throughout training.
                comp["attn_mean"] = attn_orig.detach().mean()
            if self.losses_cfg.lambda_marg_H > 0:
                # D4 (RQ2): L_marg_H = -mean(M) + beta * KL(mu || U).
                # Unlike L_ac this penalises single-key dominance via the
                # marginal of attention weights over keys, so it has no
                # mode-collapse fixed point. Requires the full (B, P, N_k)
                # attention tensor, not just the concentration summary.
                attn_w = feats["attn_w"]
                L_marg_H = attention_marginal_entropy_loss(
                    attn_w, beta=self.losses_cfg.marg_H_beta,
                )
                total = total + self.losses_cfg.lambda_marg_H * L_marg_H
                comp["L_marg_H"] = L_marg_H.detach()

        # ------------------ Pseudo-mask CAM supervision (D2) ------------------
        lam_mask_eff = self.effective_lambda_mask()
        # Log the schedule so MLflow sees the ramp even when lam_mask is 0.
        comp["lambda_mask_eff"] = torch.tensor(
            lam_mask_eff, device=fused.device, dtype=fused.dtype,
        )
        if lam_mask_eff > 0.0:
            L_mask = cam_pseudo_mask_loss(
                p3_query=feats["query_merged"],
                p4_fused=fused,
                cls_weight=self.model.classifier.weight,
                labels=labels,
                alpha_pos=self.losses_cfg.mask_alpha_pos,
                beta_neg=self.losses_cfg.mask_beta_neg,
                # Deprecated alias wins over ``mask_combiner`` when the
                # user has explicitly set it (legacy behaviour). ``None``
                # means "use mask_combiner" (the new D4 path).
                use_intersection=self.losses_cfg.mask_use_intersection,
                mask_combiner=self.losses_cfg.mask_combiner,
            )
            total = total + lam_mask_eff * L_mask
            comp["L_mask"] = L_mask.detach()

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
                    anchor_source=self.losses_cfg.con_anchor_source,
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
        # ``sync_dist=True`` averages each scalar across DDP ranks before it
        # reaches the MLflow logger. torchmetrics objects (``train_mAP``,
        # ``val_mAP``) auto-sync via their internal ``.compute()`` and do
        # NOT need this flag; setting it on them would double-count.
        self.log(
            "train/loss", total,
            prog_bar=True, on_step=True, on_epoch=True, batch_size=B,
            sync_dist=True,
        )
        for name, val in comp.items():
            self.log(
                f"train/{name}", val,
                prog_bar=False, on_step=True, on_epoch=True, batch_size=B,
                sync_dist=True,
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
        # torchmetrics' ``.compute()`` already gathers across DDP ranks;
        # do NOT pass ``sync_dist=True`` here or the value would be
        # averaged a second time.
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
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        # torchmetrics auto-syncs across ranks; no ``sync_dist`` here.
        self.log("val/mAP", self.val_mAP.compute(), prog_bar=True)
        self.val_mAP.reset()

        # ``OnlineCAMIoU.evaluate`` runs on EVERY rank (lockstep) so
        # that every metric logged through ``self.log`` is present on
        # every rank's ``trainer.callback_metrics`` dict. The
        # alternative (compute on rank 0 only, log with
        # ``rank_zero_only=True``) caused a 2026-05-06 deadlock under
        # DDP: ``ModelCheckpoint(monitor="val/cam_iou_best")`` took
        # different code paths on rank 0 (metric present -> save ckpt
        # -> ``strategy.barrier()``, which is an ``AllReduce(1)`` in
        # NCCL) vs rank 1 (metric absent -> no save -> no barrier),
        # and the asymmetric collective hung for 10 min before the
        # NCCL watchdog killed both processes. Recomputing on every
        # rank is safe because ``evaluate()`` is deterministic on the
        # same query subset + seed + model weights; the 2x compute is
        # parallelised across separate GPUs so wall-clock impact is
        # negligible (~30-60 s on rank 0 and rank 1 simultaneously).
        #
        # OOM defense: ``evaluate()`` does its own forward pass through
        # the SPDNet, which at high resolution + large rps materialises
        # a ``(B, heads, Q, K)`` attention weight tensor in fp32. With
        # the 5090 chain at rps=56 + 896², this is ~5 GiB at
        # eval_batch_size=2 and ~20 GiB at the legacy default of 8 -- the
        # 2026-05-06 P1' run died at the end of epoch 0 because rank 0
        # OOMed here. The primary fix is calibrating
        # ``online_loc_eval_batch_size`` to match training micro-batch
        # (set in ``scripts/run_phase5_5090_chain.sh``); this try/except
        # is defense-in-depth so an allocator-fragmentation OOM
        # downgrades to a logged warning instead of killing the run.
        # Free training-cached memory first to give the eval forward
        # the largest possible budget; the cost (~tens of ms) is
        # immaterial vs. a full run kill.
        #
        # Cross-rank OOM coordination: an OOM may strike one rank
        # without the other (allocator state diverges across ranks
        # because of empty_cache timing differences). If we let the
        # asymmetry through, rank 0 might log 3 scalars while rank 1
        # logs 0 -- exactly the rank-asymmetric ``self.log`` pattern
        # that triggered the original deadlock. So we all-reduce a
        # success flag (MIN: 1 only if every rank succeeded) and skip
        # the log calls everywhere whenever any rank OOMed.
        if (
            self.online_loc_metric is not None
            and self.online_loc_metric.should_run(self.current_epoch)
            and not self.trainer.sanity_checking
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                scalars = self.online_loc_metric.evaluate(self.model, self.device)
            except torch.cuda.OutOfMemoryError as oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import logging
                rank = (
                    self.trainer.global_rank
                    if hasattr(self, "trainer") and self.trainer is not None
                    else -1
                )
                logging.getLogger(__name__).warning(
                    "OnlineCAMIoU.evaluate OOM at epoch %d (rank %s): %s. "
                    "Skipping online metric for this epoch (all ranks); "
                    "training continues. Lower "
                    "losses.online_loc_eval_batch_size if this recurs.",
                    self.current_epoch, rank, oom,
                )
                scalars = {}

            # Cross-rank OOM coordination via MIN-reduce on a 1.0/0.0
            # success flag. If torch.distributed isn't initialised we
            # are running single-process so the flag passes through
            # unmodified.
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                succeeded = torch.tensor(
                    [1.0 if scalars else 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
                torch.distributed.all_reduce(
                    succeeded, op=torch.distributed.ReduceOp.MIN
                )
                if succeeded.item() < 0.5:
                    # At least one rank failed; force every rank to
                    # skip the log so the per-rank ``self.log`` count
                    # stays in lockstep.
                    if scalars:
                        import logging
                        logging.getLogger(__name__).warning(
                            "OnlineCAMIoU: another rank OOMed at epoch "
                            "%d; skipping log on this rank too to keep "
                            "DDP collectives symmetric.",
                            self.current_epoch,
                        )
                    scalars = {}

            for k, v in scalars.items():
                self.log(
                    f"val/{k}", float(v),
                    prog_bar=False, on_epoch=True,
                    sync_dist=True,
                )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        # Silently-inverted cosine guard. CosineAnnealingLR with
        # ``eta_min >= base_lr`` interpolates *upward* across its T_max window
        # instead of decaying: the highres896 run on 2026-04-30 wasted ~10 h
        # before we noticed (``trainer.min_lr=1e-5`` + linear-scaling rule
        # pulling peak to 7.8e-6 put the floor above the peak).
        # Fail loudly before any forward pass so the trap can't recur.
        if self.min_lr >= self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr:g}) must be strictly below the (already "
                f"batch-scaled) base learning rate ({self.learning_rate:g}). "
                "CosineAnnealingLR would otherwise ascend from base_lr to "
                "eta_min rather than decay. Reduce trainer.min_lr or raise "
                "model.learning_rate so that base_lr * batch_size / 256 > min_lr."
            )
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
