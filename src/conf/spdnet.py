"""Hydra configuration dataclasses for SPDNet Siamese training."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SPDNetModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    fpn_channels: int = 256
    num_classes: int = 115
    input_size: int = 448
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    mse_reduction: int = 4
    fusion_mode: str = "token"
    # Side length of the SCA reference key grid. ``0`` means "auto-scale
    # with image_size" -- specifically ``max(14, image_size // 44)``. NOTE
    # the FPN-merge that feeds SCA is at /4 stride, not /8 -- so at 896²
    # the query grid is 224×224 = 50176 tokens. The attention buffer is
    # ``(B, num_heads, Q, K)`` so memory grows linearly in K². On a 24-
    # 32 GiB card with bf16-mixed, the empirical ceilings (verified by
    # ``scripts/smoke_test_spdnet_highres.py``) are:
    #
    # * 448², batch=16, rps=14 (legacy), aux ON  -> ~9 GiB
    # * 448², batch=16, rps=20, aux ON           -> ~13 GiB
    # * 896², batch=6,  rps=14 (legacy), no-aux  -> ~23 GiB
    # * 896², batch=6,  rps=20, no-aux           -> ~26 GiB
    # * 896², batch=6,  rps=28, no-aux           -> OOM on 32 GiB
    # * 896², batch=2,  rps=40, aux ON           -> ~22 GiB
    # * 896², batch=2,  rps=56, aux ON           -> OOM on 32 GiB
    #
    # The auto rule (``image_size // 44``) gives rps=14 at 448 (legacy
    # behaviour, regression-free) and rps=20 at 896 (Q:K = 224²:20² = 125:1
    # vs legacy 256:1, modest improvement, fits at batch=6 cls-only). For
    # aux-loss highres runs you'll typically want rps=40 explicitly with
    # batch=2 and accum=15 to hit eff_batch=30; the launcher script
    # ``scripts/run_phase5_lr_fix.sh`` sets this directly.
    ref_pool_size: int = 0
    # Optional explicit override of the optimizer's base LR after
    # batch-size scaling. When >0 this REPLACES
    # ``learning_rate * effective_batch / 256`` in ``train_spdnet.py``;
    # when 0 (default) the legacy scaling rule is used (see
    # ``effective_batch_for_lr`` and §5.14.2 Trap 1 in
    # ``RESEARCH_CONTEXT.md``). Lets us pin a known-good LR (e.g.
    # 6.25e-5 to match the 448 calibration) for highres runs without
    # going through the eff-batch arithmetic each time.
    learning_rate_override: float = 0.0


@dataclass
class SPDNetDataConfig:
    root: str = "data/plantsegv3"
    pv_root: str = "data/plant-village"
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    include_plantvillage: bool = False
    num_references: int = 1
    augmentation: str = "heavy"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class SPDNetTrainerConfig:
    max_epochs: int = 80
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-5
    # Save a second "best" checkpoint chosen by val/cam_iou_best (the
    # online macro disease-IoU metric emitted by OnlineCAMIoU). The
    # default-best checkpoint ranks by val/map, which historically has
    # been a poor proxy for localization quality on WSSS; keeping both
    # writes only ~900 MB extra per run and lets downstream eval pick
    # the checkpoint that was strongest on the Phase-5 headline metric.
    # Takes effect in ``src/train_spdnet.py`` (a second ModelCheckpoint
    # callback). Safe to disable for experiments that don't emit the
    # online CAM-IoU metric (set ``spatial_losses.online_loc_eval=False``
    # as well).
    save_best_cam_iou: bool = True

    # Lightning trainer ``strategy`` argument. ``"auto"`` (the default)
    # leaves the choice to Lightning -- single GPU with ``devices=1``,
    # DDP otherwise. Set explicitly to ``"ddp"`` to force the
    # ``DDPStrategy`` constructor in ``src/train_spdnet.py`` (which then
    # passes ``find_unused_parameters`` and ``gradient_as_bucket_view``);
    # any other string (``"deepspeed"``, ``"fsdp"``, ...) is forwarded
    # verbatim to Lightning for users who know what they're doing.
    strategy: str = "auto"
    # When DDP is selected (either because ``devices > 1`` and
    # ``strategy in {"auto", "ddp"}``, or because the user explicitly
    # set ``strategy="ddp"``), pass this flag to ``DDPStrategy``. Must
    # stay ``True`` for SPDNet because several aux-loss branches are
    # entered conditionally (``proj_head`` only when ``lambda_con > 0``,
    # ``ema_teacher`` only when ``lambda_distill > 0``, attention buffers
    # only when ``want_attn``). DDP otherwise raises "expected to have
    # finished reduction in the prior iteration before starting a new
    # one" because rank 1 receives parameters whose ``.grad`` was never
    # touched on rank 0's batch.
    find_unused_parameters: bool = True

    # Convert all ``BatchNorm{1,2,3}d`` layers (in the ResNet-50 backbone
    # and the FPN/MSE necks that inherit BN momentum) to
    # ``torch.nn.SyncBatchNorm`` before DDP wraps the module. Lightning
    # forwards this to ``torch.nn.SyncBatchNorm.convert_sync_batchnorm``
    # in ``Strategy.setup``; per-step BN statistics + running stats then
    # use the WORLD batch (per-rank batch * world size) instead of the
    # noisy per-rank micro-batch.
    #
    # Why we want it on for the 5090 chain:
    #   The Phase-5 5090 recipe runs at ``data.batch_size=2`` per rank
    #   (eff_batch=32 via accum=8 across 2 GPUs) because rps=56 + 896²
    #   only fits at 24.6 GiB when micro-batch <= 2. Per-rank BN sees
    #   just 2 samples per forward, well below the ~16-32 micro-batch
    #   range the ImageNet-pretrained ResNet-50 BN was tuned for.
    #   ``sync_batchnorm=True`` restores an effective BN sample of 4
    #   (2 ranks * batch 2) -- still small, but matches the lower end
    #   of the single-card 448 baseline that produced val/mAP=0.888.
    #
    # Cost: roughly one extra all-reduce per BN forward and one per
    # backward. On a 2-rank PCIe link this adds ~3-5% to step time;
    # comfortably masked by the already-amortised gradient all-reduce
    # at the end of the step.
    #
    # No effect when ``devices <= 1`` (Lightning silently skips the
    # convert), so single-card configs stay unchanged. Default ``False``
    # preserves all legacy behaviour; the launcher
    # ``scripts/run_phase5_5090_chain.sh`` flips it on for P1'/P2'.
    sync_batchnorm: bool = False

    # NCCL collective timeout in seconds. Forwarded to
    # ``DDPStrategy(timeout=...)`` (which itself maps to
    # ``torch.distributed.init_process_group(timeout=...)``). The
    # default Lightning value is 1800 s (30 min) and that's exactly
    # how long the 2026-05-06 P1' run spent waiting on a dead rank
    # 0 after an OOM in OnlineCAMIoU before NCCL finally tripped. We
    # ship a tighter 600 s (10 min) default so future deadlocks fail
    # fast: a healthy SPDNet step is 0.3-0.5 s on this hardware,
    # OnlineCAMIoU.evaluate is ~30-60 s on rank 0, the longest
    # legitimate inter-collective gap is a checkpoint save (which
    # writes a few-hundred-MB ckpt to disk; ~5 s). 600 s leaves an
    # order-of-magnitude headroom over any of those while still
    # killing the run inside 10 min when something genuinely deadlocks.
    # ``0`` falls back to Lightning's default. No effect when
    # ``devices <= 1``.
    ddp_timeout_seconds: int = 600


@dataclass
class SPDNetSpatialLossesConfig:
    """Auxiliary spatial losses + EMA teacher + online localization metric.

    Defaults reproduce the "no aux losses, no online metric" classification
    baseline (every loss weight == 0, online metric kill-switched off);
    enable individually for the experiments described in the plan
    (`spdnet_auxiliary_spatial_losses_*.plan.md`).

    Field groups:

    * Equivariance loss (``lambda_eq``, ``equivariance_transforms``).
    * Patch-contrastive loss (``lambda_con``, ``con_*``, ``con_position``).
    * Self-distillation (``lambda_distill``, ``distill_*``, ``ema_alpha``).
    * Online localization metric (``online_loc_*``).
    """

    # ----- Equivariance loss -----
    lambda_eq: float = 0.0
    # Allowed transform IDs from src.wsss.spdnet.equivariance_transforms:
    # 0=identity, 1=hflip, 2=rot90, 3=rot180, 4=rot270.
    equivariance_transforms: tuple[int, ...] = (1, 2, 3, 4)

    # ----- Attention concentration regulariser (D1) -----
    # Pushes the per-query attention concentration map ``attn_map`` in [0, 1]
    # away from the uniform fixed point that ``L_eq`` cannot break out of
    # (diagnostic measured std=0.004 per image on the eq-only checkpoint).
    # Loss is ``L_ac = -mean(attn_map)`` so that gradient descent maximises
    # concentration. Enabling ``lambda_ac > 0`` automatically turns on the
    # ``return_attn=True`` forward path.
    #
    # ``ac_warmup_start_epoch`` / ``ac_warmup_epochs`` behave identically to
    # ``mask_warmup_*``: L_ac is multiplied by a 0 -> lambda_ac linear ramp
    # across ``[start_epoch, start_epoch + ramp)``. Both defaults are zero so
    # existing recipes keep their original epoch-0 behaviour; set non-zero to
    # delay L_ac until the classifier has built discriminative features. The
    # 2026-04-30 cold-start run collapsed attn_mean at epoch 3 because L_ac
    # was applied from epoch 0 on random MSE logits -- an issue that is not
    # visible from any checkpoint metric alone.
    lambda_ac: float = 0.0
    ac_warmup_start_epoch: int = 0
    ac_warmup_epochs: int = 0

    # ----- Marginal-entropy attention regulariser (D4 / RQ2) -----
    # Mode-collapse-free alternative to ``L_ac``. Combines the per-query
    # concentration term with a per-key marginal-entropy KL to uniform:
    #   L_marg_H = -mean(M) + marg_H_beta * (log N - H(mu))
    # Fixed point is "each query sharp AND marginal over keys uniform",
    # i.e. different queries pick different keys -- no D1-style mode
    # collapse. See ``reports/notes/rq2_attention_regularizer_analysis.md``
    # for derivation and ranking. Enabling ``lambda_marg_H > 0`` turns on
    # the ``return_attn=True`` forward path so the full (B, P, N) attention
    # weights are available.
    lambda_marg_H: float = 0.0
    marg_H_beta: float = 0.25

    # ----- Pseudo-mask CAM supervision (D2) -----
    # Directly supervise the active-class CAM slice against a class-agnostic
    # saliency pseudo-mask derived from channel variance of the pre-fusion
    # feature map (``p3_query``). Pre-training diagnostic (200 val images)
    # showed chvar top-alpha pseudo-masks hit IoU=0.26-0.28 vs GT at feat
    # resolution, above the single-threshold CAM IoU of ~0.25.
    # ``mask_combiner`` selects how positives are built (``chvar_top_alpha``
    # vs ``cam_top_alpha``); negatives are always the bottom-``mask_beta_neg``
    # of chvar. Middle positions are not supervised. Loss is MSE on per-image
    # min-max normalised CAM against the {0, 1} target, weighted by the
    # pos+neg union mask.
    lambda_mask: float = 0.0
    mask_alpha_pos: float = 0.25
    mask_beta_neg: float = 0.5
    # Combiner for the positive pseudo-mask. Must be one of:
    #   "intersection" - pos = chvar_top AND cam_top  (old D2 default;
    #                    highest precision, lowest coverage)
    #   "chvar_only"   - pos = chvar_top              (class-agnostic only)
    #   "union"        - pos = chvar_top OR cam_top   (D4 per RQ5;
    #                    +3 pp IoU vs GT at feat-res)
    mask_combiner: str = "intersection"
    # DEPRECATED alias for ``mask_combiner``. When left at its default
    # ``None`` the new ``mask_combiner`` field is used. Setting this field
    # explicitly to ``True`` maps to ``"intersection"`` and ``False`` maps
    # to ``"chvar_only"`` (the only two modes the old boolean could
    # express); this override wins over ``mask_combiner`` so that pre-D4
    # configs keep their original semantics. Will be removed in a future
    # release.
    mask_use_intersection: Optional[bool] = None
    mask_warmup_start_epoch: int = 0
    mask_warmup_epochs: int = 0

    # ----- Patch contrastive loss -----
    lambda_con: float = 0.0
    con_top_K: int = 8
    con_M_negatives: int = 16
    con_temperature: float = 0.07
    con_projection_dim: int = 128
    # Where to take the patch features from. Currently only "P3_query_merged"
    # is implemented (matches the spec).
    con_position: str = "P3_query_merged"
    # Source for the per-image anchor positions used by ``L_con``. Must be
    # one of ``"classifier"`` (default; matches the spec) or
    # ``"union_cls_chvar"`` (D3 ablation: anchor rank = element-wise max of
    # classifier-CAM rank and channel-variance rank). The second option
    # broadens anchor coverage when the two sources disagree (measured
    # Jaccard=0.15 at K=8 on the eq-only checkpoint). Background negatives
    # continue to use the classifier score alone so we keep a stable "definitely
    # not disease" anchor-to-negative polarity.
    con_anchor_source: str = "classifier"
    # Linear warmup for ``lambda_con``. The effective weight used in
    # ``training_step`` is scheduled as::
    #
    #   e < con_warmup_start_epoch                          -> 0
    #   start <= e < start + con_warmup_epochs              -> lambda_con * (e - start) / ramp
    #   e >= start + con_warmup_epochs                      -> lambda_con
    #
    # Defaults ``start=0, ramp=0`` reproduce the original no-warmup
    # behaviour (lambda_con applied in full from epoch 0 onward). A typical
    # "classifier first, then contrastive" recipe uses ``start=14, ramp=7``.
    con_warmup_start_epoch: int = 0
    con_warmup_epochs: int = 0

    # ----- Self-distillation -----
    lambda_distill: float = 0.0
    ema_alpha: float = 0.999
    distill_T_teacher: float = 0.04
    distill_T_student: float = 0.1
    distill_center_beta: float = 0.9
    distill_warmup_epochs: int = 10

    # ----- Online localization metric -----
    online_loc_eval_enabled: bool = True
    online_loc_eval_subset_size: int = 100
    online_loc_eval_seed: int = 1234
    online_loc_eval_every_n_epochs: int = 1
    # Per-batch size of the rank-0 ``OnlineCAMIoU.evaluate`` forward
    # pass through the SPDNet. CRITICAL: this MUST be tuned together
    # with ``model.ref_pool_size`` and ``data.image_size``, because the
    # SCA attention weights are materialised in fp32 at
    # ``(B, num_heads, Q, K)`` where ``Q = (image_size/4)^2`` and
    # ``K = ref_pool_size^2``. The buffer dominates the eval VRAM
    # footprint:
    #
    #   image_size=896, rps=56 (Q=224^2, K=56^2):
    #     bs=2 -> ~5  GiB attention weights   PASS on 5090 (32 GiB)
    #     bs=4 -> ~10 GiB                     PASS (tight)
    #     bs=8 -> ~20 GiB                     OOM (training residual + this)
    #
    #   image_size=448, rps=14-20 (Q=112^2, K<=20^2):
    #     bs=8 -> ~0.5 GiB                    PASS, plenty of headroom
    #
    # The 2026-05-06 P1' run on the 5090 chain crashed at end of epoch
    # 0 because the launcher set rps=56 + image_size=896 but inherited
    # this default of 8: the rank-0 evaluate() OOMed, rank 1 then sat
    # on the next ALLREDUCE for 30 min waiting for a dead rank 0
    # (NCCL watchdog default), and the 90-minute training run
    # was discarded. The launcher
    # ``scripts/run_phase5_5090_chain.sh`` now sets this explicitly to
    # 2 via the ``ONLINE_LOC_EVAL_BS`` env knob; ``lightning.py``'s
    # ``on_validation_epoch_end`` additionally wraps evaluate() in
    # try/except OOM as defense in depth.
    online_loc_eval_batch_size: int = 8
    online_loc_gt_binary_dir: str = "outputs/plantseg_binary_mc115/gt_binary_val"

    # ----- Always-on attention diagnostics -----
    # Force the ``return_attn=True`` forward path even when no aux loss
    # currently needs the attention map (in particular: pure classifier
    # runs at ``lambda_eq = lambda_ac = lambda_marg_H = 0``). Logs three
    # extra training scalars every step so the "pure" attention
    # distribution can be compared directly against later aux-loss runs:
    #
    #   train/attn_mean  -- same number ``L_ac`` negates; saturation > 0.95
    #                       reproduces the D1 mode-collapse signature.
    #   train/attn_std   -- catches the "uniform attention" failure mode
    #                       that ``attn_mean`` alone misses (mean stays
    #                       near 0.5 while std collapses toward 0).
    #   train/attn_p99   -- 99th-percentile concentration; rises faster
    #                       than the mean once any subset of queries
    #                       starts pinning on a single key.
    #
    # Memory cost is the SCA attention buffer
    # ``(B, num_heads, Q, K)`` materialised in fp32 for the
    # ``need_weights=True`` path. At 896² with batch=8, rps=56 this is
    # ``8 * 1 * 50176 * 3136 * 4 bytes ~ 5 GiB`` per card before
    # backprop graph; budget verified in §5 of the run plan. Default
    # False so legacy classification-only experiments stay on the lean
    # ``need_weights=False`` path.
    log_attn_stats: bool = False


@dataclass
class SPDNetConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    experiment_name: str = "spdnet_plantseg"
    seed: int = 42

    model: SPDNetModelConfig = field(default_factory=SPDNetModelConfig)
    data: SPDNetDataConfig = field(default_factory=SPDNetDataConfig)
    trainer: SPDNetTrainerConfig = field(default_factory=SPDNetTrainerConfig)
    losses: SPDNetSpatialLossesConfig = field(
        default_factory=SPDNetSpatialLossesConfig,
    )

    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "${experiment_name}"
    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
