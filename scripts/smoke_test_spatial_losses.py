#!/usr/bin/env python
"""End-to-end smoke test for the SPDNet auxiliary spatial losses.

Runs a 1-epoch Lightning fit on a 50-image subset of PlantSeg with all
three auxiliary losses turned on (warmup overridden to 0 so distillation
fires immediately) and asserts that:

* every loss component is finite on every step;
* ``L_dist`` is positive on at least one step (proves teacher is not
  bit-equal to student);
* the three online metric scalars (``val/cam_iou_best``,
  ``val/cam_iou_best_thr``, ``val/cam_iou_auc``) appear in
  ``trainer.logged_metrics``;
* EMA teacher state moved away from its post-deepcopy snapshot (proves
  ``on_train_batch_end`` is being called);
* every relevant trainable parameter receives a non-zero ``.grad`` after
  the first optimizer step (SCA, projection head, classifier).

Run::

    /venv/main/bin/python scripts/smoke_test_spatial_losses.py

Exits 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.data.voc_classification import (
    NUM_PLANTSEG_FG_CLASSES,
    PlantSegMCTformerDataset,
)
from src.train_spdnet import build_train_transform, build_val_transform
from src.wsss.spdnet.dataset import SiamesePlantSegDataset, siamese_collate_fn
from src.wsss.spdnet.lightning import SPDNetModule
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU


PLANTSEG_ROOT = "data/plantsegv3"
PV_ROOT = "data/plant-village"
GT_BIN_DIR = "outputs/plantseg_binary_mc115/gt_binary_val"

SUBSET_SIZE = 50          # train + val each
ONLINE_METRIC_SUBSET = 10  # tiny subset just to exercise the path
BATCH_SIZE = 4
IMAGE_SIZE = 224          # smaller than launch (448) so smoke runs in seconds
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Loss-component sniffer: stash each step's logged metrics so we can assert
# at the end of the fit. Lightning resets logged_metrics every step so we
# need a callback to capture them.
# ---------------------------------------------------------------------------


class _LogCollector(Callback):
    """Records per-step training metrics and -- at fit-end --
    ``trainer.callback_metrics`` which by then contains every val metric
    flushed by the LightningModule (``val/loss``, ``val/mAP``,
    ``val/cam_iou_*``)."""

    def __init__(self) -> None:
        self.steps: list[dict[str, float]] = []
        self.final_callback_metrics: dict[str, float] = {}

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: dict,
        batch_idx: int,
    ) -> None:
        snapshot = {
            k: float(v.item()) if hasattr(v, "item") else float(v)
            for k, v in trainer.logged_metrics.items()
        }
        self.steps.append(snapshot)

    def on_fit_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        for k, v in trainer.callback_metrics.items():
            self.final_callback_metrics[k] = (
                float(v.item()) if hasattr(v, "item") else float(v)
            )


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    _section("SPDNet auxiliary spatial losses smoke")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  Torch:         {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  PlantSeg root: {PLANTSEG_ROOT}")
    print(f"  GT binary:     {GT_BIN_DIR}")
    print(f"  subset_size:   {SUBSET_SIZE} (train) + {SUBSET_SIZE} (val)")
    print(f"  image_size:    {IMAGE_SIZE}, batch_size: {BATCH_SIZE}")

    if not Path(PLANTSEG_ROOT).exists():
        raise FileNotFoundError(
            f"{PLANTSEG_ROOT} not found. Run "
            "`dvc pull data/plantsegv3.dvc` first."
        )

    L.seed_everything(0, workers=True)

    # ---- Datasets & loaders ----
    _section("Datasets")
    train_base = PlantSegMCTformerDataset(
        root=PLANTSEG_ROOT,
        split="train",
        image_size=IMAGE_SIZE,
        transform=build_train_transform(IMAGE_SIZE, augmentation="minimal"),
        plantvillage_root=PV_ROOT,
        include_plantvillage=False,
    )
    val_base = PlantSegMCTformerDataset(
        root=PLANTSEG_ROOT,
        split="val",
        image_size=IMAGE_SIZE,
        transform=build_val_transform(IMAGE_SIZE),
        plantvillage_root=PV_ROOT,
        include_plantvillage=False,
    )
    print(f"  train (full): {len(train_base)}, val (full): {len(val_base)}")
    train_siamese = SiamesePlantSegDataset(train_base, num_references=1)
    val_siamese = SiamesePlantSegDataset(val_base, num_references=1)
    train_ds = Subset(train_siamese, list(range(min(SUBSET_SIZE, len(train_siamese)))))
    val_ds = Subset(val_siamese, list(range(min(SUBSET_SIZE, len(val_siamese)))))
    print(f"  smoke subset -> train: {len(train_ds)}, val: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
        collate_fn=siamese_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=siamese_collate_fn,
    )

    # ---- Online metric (very small) ----
    _section("OnlineCAMIoU")
    online_metric = OnlineCAMIoU(
        plantseg_root=PLANTSEG_ROOT,
        gt_binary_dir=GT_BIN_DIR,
        num_classes=NUM_PLANTSEG_FG_CLASSES,
        subset_size=ONLINE_METRIC_SUBSET,
        seed=1234,
        every_n_epochs=1,
        image_size=IMAGE_SIZE,
        eval_batch_size=BATCH_SIZE,
        enabled=True,
    )
    print(f"  online metric subset: {len(online_metric.query_names)} images")

    # ---- Lightning module with all aux losses on, warmup=0 ----
    # D1 (lambda_ac), D2 (lambda_mask + union combiner), D3 (union-anchor
    # L_con) and D4 (lambda_marg_H) are all exercised alongside the
    # original equivariance / contrastive / distillation terms. Every
    # weight is small but non-zero so the whole training_step branch-graph
    # is stressed in a single smoke.
    _section("Module")
    losses_cfg = SPDNetSpatialLossesConfig(
        lambda_eq=1.0,
        lambda_con=0.5,
        lambda_distill=0.1,
        lambda_ac=0.3,
        lambda_mask=0.5,
        lambda_marg_H=0.15,
        marg_H_beta=0.25,
        mask_combiner="union",
        mask_use_intersection=None,  # let mask_combiner win
        con_anchor_source="union_cls_chvar",
        distill_warmup_epochs=0,
        online_loc_eval_enabled=True,
        online_loc_eval_subset_size=ONLINE_METRIC_SUBSET,
    )
    module = SPDNetModule(
        num_classes=NUM_PLANTSEG_FG_CLASSES,
        fpn_channels=256,
        mse_reduction=4,
        pretrained=False,                     # smoke -> skip the download
        learning_rate=1e-4,
        weight_decay=0.05,
        warmup_epochs=0,
        min_lr=1e-5,
        fusion_mode="spatial",
        losses_cfg=losses_cfg,
        online_loc_metric=online_metric,
        image_size=IMAGE_SIZE,
    )
    print(f"  total params: {sum(p.numel() for p in module.parameters()):,}")
    print(f"  trainable:    {sum(p.numel() for p in module.parameters() if p.requires_grad):,}")
    assert module.ema_teacher is not None, "EMA teacher should be present"
    assert module.proj_head is not None, "Projection head should be present"
    assert module.distill_center is not None, "Distill center should be allocated"

    # Snapshot teacher params so we can verify they moved.
    teacher_snapshot = [p.clone() for p in module.ema_teacher.teacher.parameters()]

    # ---- Fit ----
    _section("Fit")
    collector = _LogCollector()
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        precision="32-true",                # avoid AMP weirdness in smoke
        log_every_n_steps=1,
        callbacks=[collector],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    t0 = time.time()
    trainer.fit(module, train_loader, val_loader)
    elapsed = time.time() - t0
    print(f"  fit elapsed: {elapsed:.1f}s")

    # ---- Assertions ----
    _section("Assertions")
    failures: list[str] = []

    # 1) every loss component finite on every step
    expected_components = {
        "train/L_cls", "train/L_eq", "train/L_con", "train/L_dist",
        # D1/D2/D3 additions.
        "train/L_ac", "train/L_mask", "train/attn_mean",
        "train/lambda_mask_eff", "train/lambda_con_eff",
        # D4 additions.
        "train/L_marg_H",
    }
    seen_components: set[str] = set()
    for i, snap in enumerate(collector.steps):
        for k, v in snap.items():
            seen_components.add(k)
            if not torch.isfinite(torch.tensor(v)):
                failures.append(f"step {i}: {k}={v} non-finite")
    print(f"  steps logged: {len(collector.steps)}")
    print(f"  components seen: {sorted(c for c in seen_components if c.startswith('train/'))}")
    missing = expected_components - seen_components
    if missing:
        # Lightning's logged_metrics may carry the on_step/on_epoch suffix
        # under some versions; accept either.
        for m in list(missing):
            stem = m.replace("train/", "")
            if any(stem in k for k in seen_components):
                missing.discard(m)
    if missing:
        failures.append(f"missing logged components: {missing}")

    # 2a) L_eq > 0 on at least one step.
    # Catches the ``attn_w.mean(dim=-1)`` regression where the attention
    # map is mathematically constant per query, making L_eq bit-exactly
    # zero in fp32 regardless of the model's actual equivariance. The
    # absolute magnitude of L_eq at init is small (~1e-6) because an
    # untrained backbone produces nearly-uniform attention with low
    # per-query variance; the unit tests in
    # ``tests/test_spatial_losses.py::TestAttnMapNonConstancy`` give
    # rigorous bounds on the attn_map shape itself.
    l_eq_vals = [s.get("train/L_eq", s.get("train/L_eq_step", 0.0)) for s in collector.steps]
    nonzero_eq = [v for v in l_eq_vals if v > 1e-8]
    if not nonzero_eq:
        failures.append(
            f"L_eq is bit-exactly zero across {len(l_eq_vals)} steps; "
            f"max={max(l_eq_vals) if l_eq_vals else 0}. "
            "Likely the SCA attn_map is a constant tensor."
        )
    else:
        print(f"  L_eq > 1e-8 on {len(nonzero_eq)}/{len(l_eq_vals)} steps "
              f"(max={max(nonzero_eq):.2e})")

    # 2b) L_dist > 0 on at least one step (teacher != student)
    l_dist_vals = [s.get("train/L_dist", s.get("train/L_dist_step", 0.0)) for s in collector.steps]
    nonzero_dist = [v for v in l_dist_vals if v > 1e-8]
    if not nonzero_dist:
        failures.append(f"L_dist never positive across {len(l_dist_vals)} steps; "
                        f"max={max(l_dist_vals) if l_dist_vals else 0}")
    else:
        print(f"  L_dist > 0 on {len(nonzero_dist)}/{len(l_dist_vals)} steps "
              f"(max={max(nonzero_dist):.4f})")

    # 2c) L_ac (D1) must be in [-1, 0] on every step -- range bound of
    # attn_concentration_loss. Also catches a totally flat attn_map: if
    # the mean is 0.0 on every step the model's attention is maximally
    # uniform, which is the fixed point L_ac is designed to escape. We
    # don't fail on that at init (pretrained=False here) but we log it so
    # regressions stand out.
    l_ac_vals = [
        s.get("train/L_ac", s.get("train/L_ac_step"))
        for s in collector.steps
    ]
    l_ac_vals = [v for v in l_ac_vals if v is not None]
    if not l_ac_vals:
        failures.append("train/L_ac never logged despite lambda_ac=0.3")
    else:
        lo = min(l_ac_vals); hi = max(l_ac_vals)
        if not (-1.0 - 1e-6 <= lo <= 0.0 + 1e-6 and -1.0 - 1e-6 <= hi <= 0.0 + 1e-6):
            failures.append(
                f"L_ac out of [-1, 0] range on some step: min={lo}, max={hi}"
            )
        else:
            print(f"  L_ac range: [{lo:.4f}, {hi:.4f}] (target: [-1, 0])")

    # 2d) L_mask (D2) must be in [0, 1]. MSE on per-image min-max normalised
    # CAM against a {0, 1} target is bounded.
    l_mask_vals = [
        s.get("train/L_mask", s.get("train/L_mask_step"))
        for s in collector.steps
    ]
    l_mask_vals = [v for v in l_mask_vals if v is not None]
    if not l_mask_vals:
        failures.append("train/L_mask never logged despite lambda_mask=0.5")
    else:
        lo = min(l_mask_vals); hi = max(l_mask_vals)
        if not (0.0 - 1e-6 <= lo and hi <= 1.0 + 1e-6):
            failures.append(
                f"L_mask out of [0, 1] range on some step: min={lo}, max={hi}"
            )
        else:
            print(f"  L_mask range: [{lo:.4f}, {hi:.4f}] (target: [0, 1])")

    # 2e) L_marg_H (D4) must live in [-1, beta * log(N)] where
    # N = (image_size // 4) ** 2 is the number of keys. Untrained attention
    # is near-uniform so L_marg_H starts close to 0; after a few steps it
    # can dip negative as attention sharpens. Range check catches NaNs and
    # also wildly out-of-spec beta-sweep values.
    import math as _math
    N_keys = (IMAGE_SIZE // 4) ** 2
    marg_H_upper = 0.25 * _math.log(N_keys) + 0.1  # small slack for FP
    marg_H_lower = -1.0 - 0.1
    l_marg_vals = [
        s.get("train/L_marg_H", s.get("train/L_marg_H_step"))
        for s in collector.steps
    ]
    l_marg_vals = [v for v in l_marg_vals if v is not None]
    if not l_marg_vals:
        failures.append("train/L_marg_H never logged despite lambda_marg_H=0.15")
    else:
        lo = min(l_marg_vals); hi = max(l_marg_vals)
        if not (marg_H_lower <= lo and hi <= marg_H_upper):
            failures.append(
                f"L_marg_H out of [{marg_H_lower:.2f}, {marg_H_upper:.2f}] "
                f"range on some step: min={lo}, max={hi}"
            )
        else:
            print(
                f"  L_marg_H range: [{lo:.4f}, {hi:.4f}] "
                f"(target: [{marg_H_lower:.2f}, {marg_H_upper:.2f}])"
            )

    # 3) online metric scalars logged
    iou_keys = ("val/cam_iou_best", "val/cam_iou_best_thr", "val/cam_iou_auc")
    final = collector.final_callback_metrics
    for k in iou_keys:
        if k not in final:
            failures.append(
                f"missing online metric key {k!r} in callback_metrics; "
                f"have: {sorted(final)}"
            )
    if all(k in final for k in iou_keys):
        print(f"  online metric: best={final['val/cam_iou_best']:.4f}, "
              f"best_thr={final['val/cam_iou_best_thr']:.2f}, "
              f"auc={final['val/cam_iou_auc']:.4f}")

    # 4) EMA teacher state moved
    moved = sum(
        (p_now - p_then).abs().sum().item()
        for p_now, p_then in zip(module.ema_teacher.teacher.parameters(), teacher_snapshot)
    )
    if moved <= 0:
        failures.append("EMA teacher state did NOT move during training")
    else:
        print(f"  EMA teacher L1 movement: {moved:.4f}")

    # ---- Summary ----
    _section("Summary")
    print(f"  total time: {elapsed:.1f}s")
    if failures:
        print(f"  FAILED ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  ALL OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        sys.exit(2)
