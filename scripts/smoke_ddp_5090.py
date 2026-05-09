"""Self-contained DDP smoke for the 2x RTX 5090 host.

Verifies the pre-launch invariants of the Phase-5 5090 chain WITHOUT
needing the real plantsegv3 dataset (which may still be DVC-pulling):

* ``Trainer(strategy="ddp", devices=2)`` launches both ranks, NCCL
  initialises, and a 2-epoch fit runs to completion on synthetic data
  shaped exactly like ``siamese_collate_fn`` output.
* SPDNetModule's conditional aux-loss branches (``proj_head`` /
  ``ema_teacher`` / attention-buffer path) do NOT trigger DDP's
  "expected to have finished reduction" assertion -- this is the
  bug ``find_unused_parameters=True`` prevents.
* ``losses.log_attn_stats=True`` actually emits ``train/attn_mean``,
  ``train/attn_std``, ``train/attn_p99`` from rank 0 (Lightning's
  ``log()`` with ``sync_dist=True`` averages across ranks before they
  reach the logger).
* Validation epoch end does not deadlock when the OnlineCAMIoU branch
  is exercised + ``ModelCheckpoint(monitor="val/cam_iou_best")`` is
  active. The 2026-05-06 deadlock fired exactly here: rank-0-only
  evaluate + ``rank_zero_only=True`` log made the metric live on
  rank 0 only, ModelCheckpoint then saved on rank 0 (issuing a
  ``strategy.barrier()`` -> ``AllReduce(1)``) while rank 1 skipped,
  and the asymmetric collective hung NCCL until the 600 s watchdog
  killed both processes. The smoke uses a stub OnlineCAMIoU that
  returns 3 fake scalars on every rank to reproduce the exact code
  path; if the symmetric-eval contract breaks again the smoke fails
  inside its own NCCL timeout instead of bringing down a 12-h run.

This is a strict pre-launch gate -- if it fails, the overnight chain
will fail in the same way, so we want to discover that in 60 seconds
on synthetic data, not 30 minutes into a real run.

Usage:
    python scripts/smoke_ddp_5090.py
    python scripts/smoke_ddp_5090.py --image-size 448 --rps 20 --batch 4
    python scripts/smoke_ddp_5090.py --image-size 896 --rps 56 --batch 8
        # peak-VRAM check; logs ``cuda.max_memory_allocated`` per rank.
    python scripts/smoke_ddp_5090.py --no-with-online-loc
        # skip the OnlineCAMIoU stub (kept for pre-fix bisection).

Exit codes:
    0  smoke passed (DDP fit + attn stats logged, peak VRAM OK,
       OnlineCAMIoU symmetric path completes without deadlock)
    1  trainer.fit raised (incl. NCCL watchdog deadlock if any)
    2  expected log keys missing
    3  peak VRAM > VRAM_BUDGET_GB (sanity ceiling, default 30 GiB)
    4  Trainer(sync_batchnorm=True) did not convert backbone BN layers
    5  OnlineCAMIoU stub did NOT log val/cam_iou_* on every rank
       (i.e. the symmetric contract is broken)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.conf.spdnet import SPDNetSpatialLossesConfig  # noqa: E402
from src.train_spdnet import _resolve_trainer_strategy  # noqa: E402
from src.wsss.spdnet.dataset import siamese_collate_fn  # noqa: E402
from src.wsss.spdnet.lightning import SPDNetModule  # noqa: E402


class _SyntheticSiameseDataset(Dataset):
    """Random tensors shaped exactly like the real dataset's ``__getitem__``.

    Returns ``{"query": {image, label, name}, "references": [{image,
    label, name}, ...]}`` so that ``siamese_collate_fn`` produces the
    correct flat batch layout.
    """

    def __init__(self, n: int, image_size: int, num_classes: int, num_refs: int = 1):
        self.n = n
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_refs = num_refs

    def __len__(self) -> int:
        return self.n

    def _sample(self, idx: int) -> dict:
        # Deterministic per-index so DDP rank shards see different but
        # reproducible content. Each sample has 1-3 active labels.
        g = torch.Generator().manual_seed(idx + 1)
        img = torch.randn(3, self.image_size, self.image_size, generator=g)
        active = torch.randint(0, self.num_classes, (3,), generator=g).tolist()
        label = torch.zeros(self.num_classes)
        for c in set(active):
            label[c] = 1.0
        return {"image": img, "label": label, "name": f"smoke_{idx:05d}"}

    def __getitem__(self, idx: int) -> dict:
        return {
            "query": self._sample(idx),
            "references": [self._sample(idx + 100 + r) for r in range(self.num_refs)],
        }


class _StubOnlineLocMetric:
    """Duck-typed stand-in for :class:`OnlineCAMIoU`.

    ``SPDNetModule.on_validation_epoch_end`` calls
    ``self.online_loc_metric.should_run(epoch)`` and
    ``.evaluate(model, device)``. Returning a fixed dict from every
    rank exercises the symmetric ``self.log(... sync_dist=True)``
    pattern. With the 2026-05-06 buggy code (rank-0-only branch +
    ``rank_zero_only=True`` log) the smoke would deadlock here; with
    the fix it logs on every rank, all-reduces a per-key mean (each
    rank contributes the same value -> the mean equals the value),
    and ModelCheckpoint(monitor="val/cam_iou_best") sees the metric
    on every rank's ``callback_metrics`` and takes a symmetric save
    path.

    Determinism is critical: each rank computing the SAME scalars on
    the SAME query subset is precisely why we can drop the rank-0
    gate. We mimic that by returning constants here so the test
    isn't affected by per-rank randomness in the synthetic dataset.
    """

    def __init__(self, every_n_epochs: int = 1):
        self.every_n_epochs = max(1, int(every_n_epochs))

    def should_run(self, epoch: int) -> bool:
        return epoch % self.every_n_epochs == 0

    def evaluate(self, model, device) -> dict:  # noqa: ARG002
        return {
            "cam_iou_best": 0.42,
            "cam_iou_best_thr": 0.55,
            "cam_iou_auc": 0.31,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=448,
                        help="Query/reference H=W. Default 448 (cheap).")
    parser.add_argument("--rps", type=int, default=20,
                        help="ref_pool_size for SCA. Default 20 (auto at 896).")
    parser.add_argument("--batch", type=int, default=4,
                        help="Per-rank batch. Default 4.")
    parser.add_argument("--devices", type=int, default=2,
                        help="DDP world size. Default 2.")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Tiny num_classes for a quick smoke. Default 8.")
    parser.add_argument("--num-train", type=int, default=16,
                        help="Synthetic dataset size. Default 16.")
    parser.add_argument("--num-val", type=int, default=8,
                        help="Synthetic val dataset size. Default 8.")
    parser.add_argument("--vram-budget-gb", type=float, default=30.0,
                        help="Peak ``cuda.max_memory_allocated`` budget per rank.")
    parser.add_argument("--sync-batchnorm", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pass sync_batchnorm=True to Trainer. Default True; "
                             "verifies that BN layers get rewritten to SyncBatchNorm "
                             "by Lightning before DDP wraps the module.")
    parser.add_argument("--with-online-loc", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Wire a stub OnlineCAMIoU + "
                             "ModelCheckpoint(monitor='val/cam_iou_best') so the "
                             "smoke exercises the exact rank-symmetric eval path "
                             "that deadlocked on 2026-05-06. Default True.")
    parser.add_argument("--max-epochs", type=int, default=2,
                        help="Number of epochs to run. Default 2 -- one epoch is "
                             "enough for the eval+log path to fire, but "
                             "ModelCheckpoint's 'is this better than best?' compare "
                             "+ save+barrier sequence only happens from epoch 1 "
                             "onwards (epoch 0 always saves 'last' but not 'best' "
                             "if score didn't improve).")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=180,
                        help="Tighter NCCL watchdog timeout for the smoke. "
                             "Default 180 s; the symmetric path should clear in "
                             "under a minute, so 3x that catches deadlocks fast.")
    # Aux-loss knobs. Default 0 keeps the legacy classifier-only smoke;
    # set non-zero to verify the cls + mask + eq stack (P3' P3'-equivalent
    # on synthetic data, which is the cheapest possible VRAM + DDP +
    # second-forward verification before launching a 5-7 h real run).
    parser.add_argument("--lambda-mask", type=float, default=0.0,
                        help="L_mask coefficient. Non-zero exercises "
                             "cam_pseudo_mask_loss in training_step.")
    parser.add_argument("--lambda-eq", type=float, default=0.0,
                        help="L_eq coefficient. Non-zero exercises the "
                             "second-forward attention_map(q_aug) path AND "
                             "the equivariance_loss MSE -- adds ~30%% per-step "
                             "VRAM at the train-time SCA buffer scale.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[smoke] CUDA unavailable; cannot run a 2-GPU DDP smoke.", file=sys.stderr)
        return 1
    if torch.cuda.device_count() < args.devices:
        print(
            f"[smoke] Need {args.devices} GPUs, found "
            f"{torch.cuda.device_count()}.", file=sys.stderr,
        )
        return 1

    # NCCL hygiene matches scripts/run_phase5_5090_chain.sh.
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    L.seed_everything(0, workers=True)

    train_ds = _SyntheticSiameseDataset(
        n=args.num_train, image_size=args.image_size,
        num_classes=args.num_classes, num_refs=1,
    )
    val_ds = _SyntheticSiameseDataset(
        n=args.num_val, image_size=args.image_size,
        num_classes=args.num_classes, num_refs=1,
    )
    common_loader = dict(
        batch_size=args.batch, num_workers=0, pin_memory=False,
        collate_fn=siamese_collate_fn,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)

    losses_cfg = SPDNetSpatialLossesConfig(
        lambda_eq=float(args.lambda_eq),
        lambda_con=0.0, lambda_distill=0.0,
        lambda_ac=0.0, lambda_marg_H=0.0,
        lambda_mask=float(args.lambda_mask),
        # ``mask_combiner=union`` matches the P2'/P3' production recipe.
        # ``mask_warmup_*=0`` keeps the loss at full strength from epoch
        # 0 so a 2-epoch smoke actually exercises the full L_mask path.
        mask_combiner="union",
        mask_warmup_start_epoch=0,
        mask_warmup_epochs=0,
        online_loc_eval_enabled=bool(args.with_online_loc),
        log_attn_stats=True,
    )
    online_metric = _StubOnlineLocMetric() if args.with_online_loc else None
    module = SPDNetModule(
        num_classes=args.num_classes, fpn_channels=64, mse_reduction=4,
        pretrained=False, learning_rate=1e-4, weight_decay=0.05,
        warmup_epochs=0, min_lr=1e-7,
        fusion_mode="spatial", losses_cfg=losses_cfg,
        online_loc_metric=online_metric,
        image_size=args.image_size, ref_pool_size=args.rps,
    )

    smoke_dir = ROOT / "outputs" / "_phase5_5090_smoke_ddp"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(save_dir=str(smoke_dir), name="csv")

    strategy = _resolve_trainer_strategy(
        strategy="ddp", devices=args.devices, find_unused_parameters=True,
        ddp_timeout_seconds=int(args.ddp_timeout_seconds),
    )

    # ModelCheckpoint(monitor="val/cam_iou_best") is the exact callback
    # that took the rank-asymmetric save path on 2026-05-06. Wiring it
    # in here (only when the OnlineCAMIoU stub is active) lets the
    # smoke deadlock if the symmetric-eval contract regresses.
    callbacks: list = []
    enable_ckpt = False
    if args.with_online_loc:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(smoke_dir / "ckpts"),
                filename="best_cam_iou_smoke",
                monitor="val/cam_iou_best",
                mode="max",
                save_top_k=1,
                save_last=False,
            )
        )
        enable_ckpt = True
    trainer = L.Trainer(
        max_epochs=int(args.max_epochs),
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        use_distributed_sampler=True,
        sync_batchnorm=bool(args.sync_batchnorm),
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=csv_logger,
        callbacks=callbacks,
        enable_checkpointing=enable_ckpt,
        enable_progress_bar=False,
        default_root_dir=str(smoke_dir),
        num_sanity_val_steps=0,
    )

    print(
        f"[smoke] launching DDP fit: image_size={args.image_size} "
        f"rps={args.rps} batch={args.batch} devices={args.devices}",
        flush=True,
    )
    try:
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(module, train_loader, val_loader)
    except Exception as e:  # pragma: no cover - smoke
        print(f"[smoke] trainer.fit raised: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if not trainer.is_global_zero:
        # Non-rank-0 returns silently; rank 0 does the validation.
        return 0

    # ---- SyncBatchNorm conversion check (only when devices > 1).
    # Lightning's ``DDPStrategy.setup`` calls
    # ``torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`` at fit
    # start when ``Trainer(sync_batchnorm=True)``. CRUCIALLY, on
    # ``teardown()`` (which runs inside ``trainer.fit``) the strategy
    # reverts SyncBN back to a ``_BatchNormXd`` sentinel (see
    # ``lightning.pytorch.plugins.layer_sync._BatchNormXd``,
    # ``DDPStrategy.teardown`` line 411-420). So by the time we land
    # here ALL ``SyncBatchNorm`` instances have already been reverted
    # and a naive ``isinstance(m, SyncBatchNorm)`` check would always
    # be zero -- regardless of whether SyncBN ran during training.
    #
    # The right post-fit signature of "SyncBN was applied":
    #   * Every original ``BatchNormXd`` in the backbone is now an
    #     instance of ``_BatchNormXd`` (Lightning's revert sentinel
    #     subclass of ``_BatchNorm``). Plain ``nn.BatchNorm{1,2,3}d``
    #     means SyncBN was NEVER applied (the convert wasn't reached
    #     or _layer_sync was None on the strategy).
    #
    # When ``sync_batchnorm=False`` the BN modules stay as
    # ``BatchNormXd`` (still ``_BatchNorm`` subclasses, but NOT the
    # ``_BatchNormXd`` sentinel), so the assertion polarity flips.
    from lightning.pytorch.plugins.layer_sync import _BatchNormXd
    from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm
    from torch.nn.modules.batchnorm import _BatchNorm

    fit_model = trainer.lightning_module.model  # SPDNetModule.model
    bn_mods = [m for m in fit_model.modules() if isinstance(m, _BatchNorm)]
    n_bn_total = len(bn_mods)
    n_sync_bn = sum(1 for m in bn_mods if isinstance(m, SyncBatchNorm))
    n_revert_sentinel = sum(1 for m in bn_mods if isinstance(m, _BatchNormXd))
    # "Original" plain BN = anything that's a _BatchNorm but neither
    # SyncBatchNorm nor the revert sentinel. Use the concrete
    # BatchNorm{1,2,3}d as the strict matcher to keep the polarity
    # explicit.
    n_plain_orig = sum(
        1 for m in bn_mods
        if isinstance(m, (BatchNorm1d, BatchNorm2d, BatchNorm3d))
        and not isinstance(m, (SyncBatchNorm, _BatchNormXd))
    )
    print(
        f"[smoke] BatchNorm modules total={n_bn_total} sync={n_sync_bn} "
        f"revert_sentinel={n_revert_sentinel} plain={n_plain_orig} "
        f"(sync_batchnorm={args.sync_batchnorm}, devices={args.devices})",
        flush=True,
    )
    if args.sync_batchnorm and args.devices > 1:
        if n_bn_total == 0:
            print("[smoke] FAIL: no BN modules found on the fit model -- "
                  "SyncBN check is meaningless. Did the backbone change?",
                  file=sys.stderr)
            return 4
        # Every BN must have gone through the apply+revert cycle.
        if n_revert_sentinel != n_bn_total:
            print(
                f"[smoke] FAIL: Trainer(sync_batchnorm=True) under DDP did "
                f"NOT apply SyncBatchNorm to all backbone BN layers. "
                f"Expected {n_bn_total} _BatchNormXd revert-sentinel modules, "
                f"got {n_revert_sentinel} (plain orig BN remaining: "
                f"{n_plain_orig}).",
                file=sys.stderr,
            )
            return 4
    elif not args.sync_batchnorm:
        # Inverse polarity: with the flag off we must NOT find the
        # revert sentinel anywhere -- that'd mean some other code path
        # silently flipped SyncBN on.
        if n_revert_sentinel > 0:
            print(
                f"[smoke] FAIL: sync_batchnorm=False but found "
                f"{n_revert_sentinel} _BatchNormXd revert sentinels -- "
                f"someone else applied SyncBN.",
                file=sys.stderr,
            )
            return 4

    logged = trainer.logged_metrics
    # Lightning fans ``on_step=True, on_epoch=True`` logs into
    # ``<key>_step`` and ``<key>_epoch`` variants, so we accept either.
    def _has(k: str) -> bool:
        return k in logged or f"{k}_step" in logged or f"{k}_epoch" in logged

    expected = ["train/attn_mean", "train/attn_std", "train/attn_p99",
                "train/loss", "train/L_cls", "val/loss", "val/mAP"]
    if float(args.lambda_mask) > 0:
        expected.append("train/L_mask")
    if float(args.lambda_eq) > 0:
        expected.append("train/L_eq")
    missing = [k for k in expected if not _has(k)]
    if missing:
        print(f"[smoke] FAIL: missing keys in trainer.logged_metrics: {missing}",
              file=sys.stderr)
        print(f"[smoke] logged keys present: {sorted(logged.keys())}", file=sys.stderr)
        return 2

    def _val(k: str) -> float:
        for cand in (k, f"{k}_epoch", f"{k}_step"):
            if cand in logged:
                return float(logged[cand])
        raise KeyError(k)

    # OnlineCAMIoU symmetric-path contract check. With the stub wired
    # in, every rank should have logged val/cam_iou_* via sync_dist=True
    # so the keys appear in trainer.logged_metrics on every rank. If the
    # symmetric contract regressed (someone re-added is_global_zero or
    # rank_zero_only=True) the run would deadlock and we'd never get
    # here -- but we also keep an explicit content check as a tripwire
    # in case Lightning's behaviour changes.
    if args.with_online_loc:
        cam_iou_expected = [
            "val/cam_iou_best", "val/cam_iou_best_thr", "val/cam_iou_auc",
        ]
        cam_iou_missing = [k for k in cam_iou_expected if not _has(k)]
        if cam_iou_missing:
            print(
                f"[smoke] FAIL: OnlineCAMIoU stub was active but "
                f"val/cam_iou_* keys are missing on rank 0: "
                f"{cam_iou_missing}. The symmetric-eval contract is "
                f"broken (likely is_global_zero gate or rank_zero_only=True "
                f"snuck back into on_validation_epoch_end).",
                file=sys.stderr,
            )
            print(f"[smoke] logged keys present: {sorted(logged.keys())}",
                  file=sys.stderr)
            return 5
        # Content check: each rank logs the same constant scalars from
        # the stub, so after sync_dist=True averaging the value should
        # equal the constant (within float noise).
        for key, exp_val in [("val/cam_iou_best", 0.42),
                             ("val/cam_iou_best_thr", 0.55),
                             ("val/cam_iou_auc", 0.31)]:
            got = _val(key)
            if abs(got - exp_val) > 1e-3:
                print(
                    f"[smoke] FAIL: {key}={got:.4f} but expected "
                    f"~{exp_val:.4f} (constant from stub). sync_dist "
                    f"is averaging differently than expected; "
                    f"investigate before relaunching.",
                    file=sys.stderr,
                )
                return 5

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[smoke] peak CUDA memory (rank 0): {peak:.2f} GiB", flush=True)
    print(f"[smoke] sample logged metrics:")
    for key in sorted(expected):
        v = _val(key)
        print(f"            {key:24s}  {v:+.6f}")

    if peak > args.vram_budget_gb:
        print(
            f"[smoke] FAIL: peak VRAM {peak:.2f} GiB exceeds budget "
            f"{args.vram_budget_gb} GiB. Reduce batch or rps.",
            file=sys.stderr,
        )
        return 3

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
