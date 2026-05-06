"""Anchor- and attention-quality diagnostic for an SPDNet checkpoint.

Given a converged SPDNet spatial checkpoint, answer the question that sits at
the core of the ``L_con`` null-result debate: do the top-K CAM anchors that
``patch_contrastive_loss`` uses as positives actually fall on disease pixels?

The script reuses the ``OnlineCAMIoU`` data pipeline (same val subset,
references, GT masks, transforms) and adds, per image:

* ``classifier_cam`` anchors: the exact top-K positions that ``L_con`` uses
  (``argsort(W_cls[c_active] @ p4_fused)``, ``K=8`` by default).
* Four reference anchor sources for comparison:
    - ``attn_map`` top-K (per-query attention concentration)
    - ``p3_merged_chmean`` top-K (channel mean of pre-fusion features)
    - ``p3_merged_chvar`` top-K (channel variance of pre-fusion features)
    - ``random`` top-K (averaged over ``--n-random-trials`` draws)

Per-image metrics:

* ``precision_at_K``  -- fraction of top-K anchors inside the GT mask.
* ``recall_gt``       -- fraction of GT pixels covered by the top-K anchors.
* ``iou``             -- ``|anchor ∩ gt| / |anchor ∪ gt|`` at feature res.
* ``cam_spatial_entropy`` -- ``H(softmax(S_anchor / T))``, threshold-free
  "is the CAM sharp or diffuse?" scalar (nats).
* ``attn_map_mean/std/q50/q90/max`` -- distribution of per-query attention
  concentration ``1 - H(p_key|q)/log N_key``.
* ``cam_pearson_gt``  -- Pearson correlation between min-max normalised
  ``S_anchor`` and the binary GT mask at feature resolution.

Aggregates: mean / std / p25 / p50 / p75 over the subset, plus per-anchor
source summaries that make it trivially obvious whether the classifier or
any cheaper heuristic is actually pointing at disease.

Output: a JSON blob + a one-screen stdout summary. Optional ``--out-dir`` for
a CSV of per-image rows, useful for downstream pandas analysis.

Usage (eq-only ckpt, 200 images, batch 4 on a single GPU)::

    python scripts/diagnose_spdnet_anchors.py \
        --ckpt outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt \
        --subset-size 200 \
        --out-dir outputs/diagnostics/anchors_eq72

Why this is the right diagnostic:

1. If ``classifier_cam precision_at_K`` is low, then ``L_con`` is fed noisy
   anchors -- its InfoNCE gradient points nowhere useful, and no amount of
   temperature / warmup tuning saves it. That's the circularity concern.
2. If ``p3_chmean`` or ``p3_chvar`` precision_at_K is materially higher than
   ``classifier_cam``, the feature map already contains localization signal
   that the classifier is actively discarding.
3. If ``attn_map`` precision_at_K is also low, the equivariance target is
   uninformative and explains why ``L_eq`` can't break out of its
   near-uniform fixed point.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.conf.spdnet import SPDNetSpatialLossesConfig  # noqa: E402
from src.wsss.spdnet.lightning import SPDNetModule  # noqa: E402
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anchor-source helpers.
# ---------------------------------------------------------------------------


def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``scores``: ``(B, P)`` -> top-k indices ``(B, K)`` (descending)."""
    return torch.topk(scores, k=k, dim=1).indices


def _indices_to_mask(indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``indices``: ``(B, K)`` flat (0..HW-1) -> ``(B, H, W)`` {0,1} float."""
    B, K = indices.shape
    out = torch.zeros(B, H * W, dtype=torch.float32, device=indices.device)
    out.scatter_(1, indices, 1.0)
    return out.view(B, H, W)


def _precision_recall_iou(
    anchor_mask: torch.Tensor, gt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image anchor-vs-GT metrics. All tensors ``(B, H, W)`` in ``{0, 1}``.

    Returns ``(precision_at_K, recall_gt, iou, gt_area)``. Each is ``(B,)``
    on CPU for clean aggregation. ``gt_area == 0`` short-circuits to 1.0
    precision / NaN IoU (images with no disease aren't informative).
    """
    B = anchor_mask.shape[0]
    anc = anchor_mask.reshape(B, -1)
    gt = gt_mask.reshape(B, -1).float()

    inter = (anc * gt).sum(dim=1)
    anc_area = anc.sum(dim=1).clamp_min(1.0)
    gt_area = gt.sum(dim=1)
    union = ((anc + gt) > 0).float().sum(dim=1)

    prec = inter / anc_area
    rec = torch.where(gt_area > 0, inter / gt_area.clamp_min(1.0), torch.full_like(gt_area, float("nan")))
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.full_like(union, float("nan")))
    return prec.cpu(), rec.cpu(), iou.cpu(), gt_area.cpu()


# ---------------------------------------------------------------------------
# Forward pass that exposes the same internals ``L_con`` uses.
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_diag(
    model,
    q: torch.Tensor,
    r: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One diagnostic forward pass.

    Returns all intermediate tensors that matter for the anchor / attention
    question: ``p3_query`` (pre-fusion), ``p4_fused`` (post-fusion),
    ``attn_map`` (per-query concentration in [0, 1]), ``attn_w`` (raw
    softmax (B, P, N_key)), ``cls_weight`` (C, C_in), ``logits`` (B, C).
    """
    sca = model.spatial_attn

    feats = model.extract_features(q)
    fpn_out = model.fpn(feats)
    mse_out = [model.mse(p) for p in fpn_out]
    query_merged = model._merge_fpn(mse_out)                       # (B, C_in, H, W)

    r_fpn = model.fpn(model.extract_features(r))
    r_mse = [model.mse(p) for p in r_fpn]
    ref_merged = model._merge_fpn(r_mse)                           # (B, C_in, Hr, Wr)

    B, C_in, H, W = query_merged.shape
    ref_pooled = sca.pool(ref_merged)                              # (B, C_in, h, w)

    q_tok = query_merged.flatten(2).permute(0, 2, 1)               # (B, P, C_in)
    kv_tok = ref_pooled.flatten(2).permute(0, 2, 1)                # (B, N, C_in)
    q_tok = sca.norm_q(q_tok)
    kv_tok = sca.norm_kv(kv_tok)

    # Fused output (training-path) + raw attn weights (eval-path) to match
    # ``_spatial_attn_with_map`` exactly. Second MHA call is dropout-free so
    # ``attn_w`` remains a valid probability distribution per query.
    attended, _ = sca.cross_attn(q_tok, kv_tok, kv_tok)
    attended = attended.permute(0, 2, 1).view(B, C_in, H, W)
    fused = query_merged + sca.gate * attended

    saved = sca.cross_attn.training
    sca.cross_attn.eval()
    try:
        _, attn_w = sca.cross_attn(
            q_tok, kv_tok, kv_tok, need_weights=True, average_attn_weights=True,
        )
    finally:
        sca.cross_attn.train(saved)

    log_N = math.log(attn_w.shape[-1])
    p = attn_w.clamp_min(1e-12)
    neg_ent = (p * p.log()).sum(dim=-1)                            # (B, P)
    attn_map = (1.0 + neg_ent / log_N).view(B, H, W)               # in [0, 1]

    logits = model.classifier(fused.amax(dim=(2, 3)))

    return {
        "p3_query": query_merged,
        "p4_fused": fused,
        "attn_map": attn_map,
        "attn_w": attn_w,
        "logits": logits,
        "cls_weight": model.classifier.weight.detach(),
    }


# ---------------------------------------------------------------------------
# Per-image diagnostic aggregator.
# ---------------------------------------------------------------------------


SOURCE_NAMES = [
    "classifier_cam",
    "attn_map",
    "p3_chmean",
    "p3_chvar",
    "random",
]


def _summarise(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p25", "p50", "p75", "n")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


def run_diagnostic(
    ckpt_path: Path,
    plantseg_root: Path,
    gt_binary_dir: Path,
    subset_size: int,
    top_k: int,
    n_random_trials: int,
    eval_batch_size: int,
    device: torch.device,
    num_classes: int = 115,
    image_size: int = 448,
    seed: int = 1234,
) -> dict:
    """Run the whole diagnostic. Returns a nested-dict report."""
    log.info("Loading SPDNet eq-only checkpoint: %s", ckpt_path)
    module = SPDNetModule(
        num_classes=num_classes,
        fpn_channels=256,
        fusion_mode="spatial",
        losses_cfg=SPDNetSpatialLossesConfig(lambda_eq=0.0, lambda_con=0.0, lambda_distill=0.0),
        online_loc_metric=None,
        image_size=image_size,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(sd.get("state_dict", sd), strict=False)
    if missing:
        log.warning("load_state_dict missing %d keys (first 3): %s", len(missing), missing[:3])
    if unexpected:
        log.warning("load_state_dict unexpected %d keys (first 3): %s", len(unexpected), unexpected[:3])
    module.eval().to(device)
    model = module.model

    log.info("Building OnlineCAMIoU subset (size=%d)", subset_size)
    loc = OnlineCAMIoU(
        plantseg_root=str(plantseg_root),
        gt_binary_dir=str(gt_binary_dir),
        num_classes=num_classes,
        subset_size=subset_size,
        seed=seed,
        every_n_epochs=1,
        image_size=image_size,
        eval_batch_size=eval_batch_size,
    )
    N = loc.query_images.shape[0]
    log.info("Running on %d query/ref pairs at %dx%d", N, image_size, image_size)

    # Per-image accumulators, keyed by anchor-source name.
    prec: dict[str, list[float]] = {s: [] for s in SOURCE_NAMES}
    rec: dict[str, list[float]] = {s: [] for s in SOURCE_NAMES}
    iou: dict[str, list[float]] = {s: [] for s in SOURCE_NAMES}

    # Per-image table for downstream pandas analysis. ``_ent`` and
    # ``pearson`` referenced below are defined inside each batch iteration
    # (see CAM sharpness / Pearson blocks).
    per_image_rows: list[dict] = []

    # Per-K precision sweep for the classifier. Tells us whether picking
    # fewer anchors (e.g. top-1) would be cleaner than the K=8 default.
    K_SWEEP = [1, 2, 4, 8, 16, 32]
    prec_by_k_cls: dict[int, list[float]] = {k: [] for k in K_SWEEP}
    prec_by_k_chvar: dict[int, list[float]] = {k: [] for k in K_SWEEP}
    # Overlap between classifier and chvar top-K anchors (Jaccard on sets).
    overlap_cls_chvar: list[float] = []

    # Classifier-anchor only: also track CAM sharpness and attention stats.
    cam_entropy_T1: list[float] = []
    cam_entropy_T01: list[float] = []
    cam_pearson_gt: list[float] = []
    attn_stats: dict[str, list[float]] = {k: [] for k in ("mean", "std", "q50", "q90", "max")}
    attn_entropy_nats: list[float] = []
    logits_active: list[float] = []
    logits_top1_is_active: list[float] = []
    gt_area_frac: list[float] = []
    active_counts: list[int] = []

    # Anchor positions x classifier_cam only, for spatial heatmap aggregate.
    anchor_hits_map = torch.zeros(image_size, image_size, dtype=torch.float32)
    anchor_total = 0
    gt_hits_map = torch.zeros(image_size, image_size, dtype=torch.float32)

    for start in range(0, N, eval_batch_size):
        stop = min(start + eval_batch_size, N)
        q = loc.query_images[start:stop].to(device, non_blocking=True)
        r = loc.ref_images[start:stop].to(device, non_blocking=True)
        labels_b = loc.query_labels[start:stop].to(device, non_blocking=True)  # (b, C)
        gt_masks_img = loc.query_masks[start:stop].to(device, non_blocking=True)  # (b, H, W)

        feats = forward_diag(model, q, r)
        p3 = feats["p3_query"]                                    # (b, C_in, Hf, Wf)
        p4 = feats["p4_fused"]                                    # (b, C_in, Hf, Wf)
        attn_map = feats["attn_map"]                              # (b, Hf, Wf) in [0, 1]
        attn_w = feats["attn_w"]                                  # (b, Pf, Nk)
        cls_w = feats["cls_weight"]                               # (C, C_in)
        logits = feats["logits"]                                  # (b, C)

        b, C_in, Hf, Wf = p4.shape
        Pf = Hf * Wf

        # Active class per image (same as _first_active_class in L_con).
        valid = labels_b.sum(dim=1) > 0
        active_first = labels_b.argmax(dim=1).long().masked_fill(~valid, 0)  # (b,)
        active_counts.extend(int(x) for x in labels_b.sum(dim=1).cpu().tolist())

        # GT mask at feature resolution (for anchor-vs-GT metrics).
        gt_feat = F.interpolate(
            gt_masks_img.unsqueeze(1), size=(Hf, Wf), mode="nearest",
        ).squeeze(1)                                               # (b, Hf, Wf)
        gt_feat_bin = (gt_feat > 0).float()
        gt_flat = gt_feat_bin.reshape(b, -1)

        # Top-K from each anchor source.
        # 1) classifier_cam: gather per-class score on the active class, min-max.
        S_full = torch.einsum("nc,bchw->bnhw", cls_w, p4)          # (b, C, Hf, Wf)
        idx_c = active_first[:, None, None, None].expand(-1, 1, Hf, Wf)
        S_act = torch.gather(S_full, 1, idx_c).squeeze(1).reshape(b, -1)  # (b, Pf)
        s_min = S_act.amin(dim=1, keepdim=True)
        s_max = S_act.amax(dim=1, keepdim=True)
        S_act_norm = (S_act - s_min) / (s_max - s_min + 1e-8)

        anchor_idx: dict[str, torch.Tensor] = {}
        anchor_idx["classifier_cam"] = _topk_indices(S_act_norm, top_k)

        # 2) attn_map top-K.
        anchor_idx["attn_map"] = _topk_indices(attn_map.reshape(b, -1), top_k)

        # 3) p3_chmean top-K (pre-fusion channel mean).
        p3_chmean = p3.mean(dim=1).reshape(b, -1)
        anchor_idx["p3_chmean"] = _topk_indices(p3_chmean, top_k)

        # 4) p3_chvar top-K (pre-fusion channel variance).
        p3_chvar = p3.var(dim=1, unbiased=False).reshape(b, -1)
        anchor_idx["p3_chvar"] = _topk_indices(p3_chvar, top_k)

        # K-sweep: recompute precision@K over a range of K values. Reuses
        # the sorted index tensor from S_act_norm and chvar.
        _, sorted_cls = torch.sort(S_act_norm, dim=1, descending=True)
        _, sorted_chvar = torch.sort(p3_chvar, dim=1, descending=True)
        for k_val in K_SWEEP:
            for src_name, idx_sorted, pool in [
                ("classifier_cam", sorted_cls, prec_by_k_cls),
                ("p3_chvar", sorted_chvar, prec_by_k_chvar),
            ]:
                idxs = idx_sorted[:, :k_val]
                mask = _indices_to_mask(idxs, Hf, Wf).to(gt_feat_bin.device)
                p_, _, _, ga = _precision_recall_iou(mask, gt_feat_bin)
                valid_gt = (ga > 0).numpy()
                for j in range(b):
                    if valid_gt[j]:
                        pool[k_val].append(float(p_[j].item()))

        # Overlap between classifier and chvar anchors: Jaccard |A & B|/|A u B|.
        cls_set = [set(anchor_idx["classifier_cam"][j].cpu().tolist()) for j in range(b)]
        chvar_set = [set(anchor_idx["p3_chvar"][j].cpu().tolist()) for j in range(b)]
        for j in range(b):
            if (gt_flat[j].sum() > 0).item():
                a, c = cls_set[j], chvar_set[j]
                if a or c:
                    overlap_cls_chvar.append(len(a & c) / len(a | c))

        # 5) random: average precision/recall/iou over N_RANDOM_TRIALS.
        rng = torch.Generator(device="cpu").manual_seed(seed + start)

        # Compute metrics per source.
        for src, idx_tensor in anchor_idx.items():
            mask = _indices_to_mask(idx_tensor, Hf, Wf).to(gt_feat_bin.device)
            p_, r_, i_, ga = _precision_recall_iou(mask, gt_feat_bin)
            # Skip images with no GT (rec undefined). Mirror OnlineCAMIoU's
            # "empty-GT -> perfect IoU" treatment would hide the null case;
            # we'd rather *report* the subset where the question is
            # well-posed.
            valid_gt = (ga > 0).numpy()
            for j in range(b):
                if valid_gt[j]:
                    prec[src].append(float(p_[j].item()))
                    rec[src].append(float(r_[j].item()))
                    iou[src].append(float(i_[j].item()))

        # Random: averaged over trials.
        r_prec, r_rec, r_iou = [], [], []
        for _t in range(n_random_trials):
            perm = torch.stack([
                torch.randperm(Pf, generator=rng)[:top_k] for _ in range(b)
            ]).to(gt_feat_bin.device)                              # (b, K)
            mask = _indices_to_mask(perm, Hf, Wf).to(gt_feat_bin.device)
            p_, r_, i_, ga = _precision_recall_iou(mask, gt_feat_bin)
            r_prec.append(p_.numpy())
            r_rec.append(r_.numpy())
            r_iou.append(i_.numpy())
        r_prec = np.nanmean(np.stack(r_prec), axis=0)
        r_rec = np.nanmean(np.stack(r_rec), axis=0)
        r_iou = np.nanmean(np.stack(r_iou), axis=0)
        ga_np = gt_flat.sum(dim=1).cpu().numpy()
        for j in range(b):
            if ga_np[j] > 0:
                prec["random"].append(float(r_prec[j]))
                rec["random"].append(float(r_rec[j]))
                iou["random"].append(float(r_iou[j]))

        # CAM sharpness (classifier_cam): entropy of softmax(S_act / T).
        def _ent(T: float) -> torch.Tensor:
            probs = torch.softmax(S_act / T, dim=1)
            return -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)

        cam_entropy_T1.extend([float(v) for v in _ent(1.0).cpu().tolist()])
        cam_entropy_T01.extend([float(v) for v in _ent(0.1).cpu().tolist()])

        # Pearson correlation between S_act_norm and GT at feature res.
        S_centered = S_act_norm - S_act_norm.mean(dim=1, keepdim=True)
        gt_c = gt_flat - gt_flat.mean(dim=1, keepdim=True)
        denom = (S_centered.pow(2).sum(dim=1).sqrt()
                 * gt_c.pow(2).sum(dim=1).sqrt()).clamp_min(1e-8)
        pearson = (S_centered * gt_c).sum(dim=1) / denom
        cam_pearson_gt.extend([float(v) for v in pearson.cpu().tolist()])

        # Attention-map summary per image.
        am_flat = attn_map.reshape(b, -1).cpu().numpy()
        attn_stats["mean"].extend(am_flat.mean(axis=1).tolist())
        attn_stats["std"].extend(am_flat.std(axis=1).tolist())
        attn_stats["q50"].extend(np.percentile(am_flat, 50, axis=1).tolist())
        attn_stats["q90"].extend(np.percentile(am_flat, 90, axis=1).tolist())
        attn_stats["max"].extend(am_flat.max(axis=1).tolist())
        # Raw attention entropy (nats) averaged over query positions.
        attn_p = attn_w.clamp_min(1e-12)
        ent_q = -(attn_p * attn_p.log()).sum(dim=-1)                # (b, P)
        attn_entropy_nats.extend([float(v) for v in ent_q.mean(dim=1).cpu().tolist()])

        # Classifier sanity: is the active class within logits.top-3?
        y = active_first.cpu().numpy()
        top3 = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()
        logits_active.extend([float(logits[j, y[j]].item()) for j in range(b)])
        logits_top1_is_active.extend([int(y[j] == top3[j, 0]) for j in range(b)])

        gt_area_frac.extend([float(v) for v in (gt_flat.sum(dim=1) / Pf).cpu().tolist()])

        # Per-image row (for CSV dump).
        ga_np = gt_flat.sum(dim=1).cpu().numpy()
        for j in range(b):
            if ga_np[j] <= 0:
                continue
            row = {
                "stem": loc.query_names[start + j],
                "gt_area_frac": float(ga_np[j] / Pf),
                "active_class": int(active_first[j].cpu().item()),
                "prec_classifier_cam": float((_indices_to_mask(anchor_idx["classifier_cam"][j:j+1], Hf, Wf).to(gt_feat_bin.device) * gt_feat_bin[j]).sum().item() / top_k),
                "prec_attn_map": float((_indices_to_mask(anchor_idx["attn_map"][j:j+1], Hf, Wf).to(gt_feat_bin.device) * gt_feat_bin[j]).sum().item() / top_k),
                "prec_p3_chmean": float((_indices_to_mask(anchor_idx["p3_chmean"][j:j+1], Hf, Wf).to(gt_feat_bin.device) * gt_feat_bin[j]).sum().item() / top_k),
                "prec_p3_chvar": float((_indices_to_mask(anchor_idx["p3_chvar"][j:j+1], Hf, Wf).to(gt_feat_bin.device) * gt_feat_bin[j]).sum().item() / top_k),
                "cam_entropy_T1_nats": float(_ent(1.0)[j].cpu().item()),
                "pearson_cam_gt": float(pearson[j].cpu().item()),
                "attn_map_mean": float(attn_map[j].mean().item()),
                "attn_map_max": float(attn_map[j].max().item()),
                "top1_is_active": int(active_first[j].item() == int(torch.topk(logits[j], 1).indices[0].item())),
            }
            per_image_rows.append(row)

        # Spatial heatmap of classifier-anchor hits, upsampled to image res
        # for human-readable inspection downstream.
        anchor_mask_c = _indices_to_mask(anchor_idx["classifier_cam"], Hf, Wf)
        anchor_mask_c_full = F.interpolate(
            anchor_mask_c.unsqueeze(1), size=(image_size, image_size),
            mode="nearest",
        ).squeeze(1).cpu()
        anchor_hits_map += anchor_mask_c_full.sum(dim=0)
        anchor_total += int(anchor_mask_c_full.shape[0])
        gt_hits_map += gt_masks_img.cpu().sum(dim=0)

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    report: dict = {
        "config": {
            "ckpt": str(ckpt_path),
            "subset_size": int(subset_size),
            "images_with_gt": int(len(prec["classifier_cam"])),
            "top_k": int(top_k),
            "image_size": int(image_size),
            "feature_hw": [Hf, Wf],
            "feature_P": int(Pf),
            "n_random_trials": int(n_random_trials),
        },
        "chance": {
            "mean_gt_area_frac": float(np.mean(gt_area_frac)) if gt_area_frac else float("nan"),
            "median_gt_area_frac": float(np.median(gt_area_frac)) if gt_area_frac else float("nan"),
            "expected_precision_at_K_uniform": float(np.mean(gt_area_frac)) if gt_area_frac else float("nan"),
            "note": "Uniform-random precision@K = E[fg_frac]. Precision@K below that "
                   "means an anchor source is WORSE than random.",
        },
        "sources": {
            src: {
                "precision_at_K": _summarise(prec[src]),
                "recall_gt": _summarise(rec[src]),
                "iou": _summarise(iou[src]),
            }
            for src in SOURCE_NAMES
        },
        "classifier_cam_sharpness": {
            "cam_entropy_softmax_T=1.0_nats": _summarise(cam_entropy_T1),
            "cam_entropy_softmax_T=0.1_nats": _summarise(cam_entropy_T01),
            "max_uniform_entropy_nats": float(math.log(Pf)),
            "pearson_cam_vs_gt": _summarise(cam_pearson_gt),
        },
        "attn_map_distribution": {
            "mean": _summarise(attn_stats["mean"]),
            "std": _summarise(attn_stats["std"]),
            "q50": _summarise(attn_stats["q50"]),
            "q90": _summarise(attn_stats["q90"]),
            "max": _summarise(attn_stats["max"]),
            "raw_attention_entropy_nats": _summarise(attn_entropy_nats),
            "max_uniform_entropy_nats": float(math.log(attn_w.shape[-1])),
        },
        "classifier_head_sanity": {
            "logits_on_active_class": _summarise(logits_active),
            "top1_is_active_class": float(np.mean(logits_top1_is_active)) if logits_top1_is_active else float("nan"),
            "mean_num_active_classes_per_image": float(np.mean(active_counts)) if active_counts else float("nan"),
        },
        "k_sweep": {
            "classifier_cam": {str(k): _summarise(prec_by_k_cls[k]) for k in K_SWEEP},
            "p3_chvar": {str(k): _summarise(prec_by_k_chvar[k]) for k in K_SWEEP},
        },
        "anchor_agreement": {
            "jaccard_classifier_vs_chvar": _summarise(overlap_cls_chvar),
            "note": "Jaccard of top-K=%d position sets. Low -> classifier and chvar "
                    "find DIFFERENT disease positions; high -> redundant." % int(top_k),
        },
    }
    return report, anchor_hits_map, gt_hits_map, anchor_total, per_image_rows


def _print_summary(report: dict) -> None:
    cfg = report["config"]
    log.info("=" * 70)
    log.info("Anchor-quality diagnostic for %s", cfg["ckpt"])
    log.info("  subset_size=%d, images_with_gt=%d, top_k=%d, feat=%s, P=%d",
             cfg["subset_size"], cfg["images_with_gt"], cfg["top_k"],
             cfg["feature_hw"], cfg["feature_P"])
    log.info("  mean disease area (feat-res): %.4f -> uniform-random precision@K = %.4f",
             report["chance"]["mean_gt_area_frac"],
             report["chance"]["expected_precision_at_K_uniform"])
    log.info("-" * 70)
    log.info("%-18s | prec@K   recall_gt   IoU   (mean +/- std)", "source")
    log.info("-" * 70)
    for src, s in report["sources"].items():
        p = s["precision_at_K"]
        r = s["recall_gt"]
        i = s["iou"]
        log.info("%-18s | %.4f+-%.4f  %.4f+-%.4f  %.4f+-%.4f",
                 src, p["mean"], p["std"], r["mean"], r["std"],
                 i["mean"], i["std"])
    log.info("-" * 70)
    sharp = report["classifier_cam_sharpness"]
    log.info("classifier_cam CAM spatial entropy (nats), max uniform = %.3f:",
             sharp["max_uniform_entropy_nats"])
    log.info("  T=1.0: %.3f+-%.3f (p50=%.3f)  -- close to max -> diffuse CAM",
             sharp["cam_entropy_softmax_T=1.0_nats"]["mean"],
             sharp["cam_entropy_softmax_T=1.0_nats"]["std"],
             sharp["cam_entropy_softmax_T=1.0_nats"]["p50"])
    log.info("  T=0.1: %.3f+-%.3f (p50=%.3f)  -- tempered CAM, closer to argmax",
             sharp["cam_entropy_softmax_T=0.1_nats"]["mean"],
             sharp["cam_entropy_softmax_T=0.1_nats"]["std"],
             sharp["cam_entropy_softmax_T=0.1_nats"]["p50"])
    log.info("  Pearson(CAM, GT) at feature res: %.4f+-%.4f (p25=%.3f, p75=%.3f)",
             sharp["pearson_cam_vs_gt"]["mean"],
             sharp["pearson_cam_vs_gt"]["std"],
             sharp["pearson_cam_vs_gt"]["p25"],
             sharp["pearson_cam_vs_gt"]["p75"])
    attn = report["attn_map_distribution"]
    log.info("attn_map (per-query concentration, [0,1]):")
    log.info("  image-mean: %.4f+-%.4f  std: %.4f  p90: %.4f  max: %.4f",
             attn["mean"]["mean"], attn["mean"]["std"],
             attn["std"]["mean"], attn["q90"]["mean"], attn["max"]["mean"])
    log.info("  raw key-softmax entropy per query: %.3f / %.3f nats (actual/uniform) "
             "-> %.1f%% of max",
             attn["raw_attention_entropy_nats"]["mean"],
             attn["max_uniform_entropy_nats"],
             100.0 * attn["raw_attention_entropy_nats"]["mean"] / attn["max_uniform_entropy_nats"])
    cls = report["classifier_head_sanity"]
    log.info("classifier sanity: top1=active in %.1f%% of images; avg active labels %.2f",
             100.0 * cls["top1_is_active_class"],
             cls["mean_num_active_classes_per_image"])
    log.info("-" * 70)
    log.info("precision @ K sweep (mean over images with GT):")
    ks = sorted(int(k) for k in report["k_sweep"]["classifier_cam"].keys())
    log.info("  %-18s " + "  ".join(f"K={k:>3}" for k in ks), "source")
    for src in ("classifier_cam", "p3_chvar"):
        row = report["k_sweep"][src]
        vals = "  ".join(f"{row[str(k)]['mean']:>5.3f}" for k in ks)
        log.info("  %-18s %s", src, vals)
    ag = report["anchor_agreement"]["jaccard_classifier_vs_chvar"]
    log.info("Jaccard(classifier_top-K, chvar_top-K): %.3f+-%.3f (p50=%.3f) "
             "-- low = complementary localizations",
             ag["mean"], ag["std"], ag["p50"])
    log.info("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path, help="SPDNet Lightning .ckpt")
    p.add_argument("--plantseg-root", default="data/plantsegv3", type=Path)
    p.add_argument("--gt-binary-dir", default="outputs/plantseg_binary_mc115/gt_binary_val", type=Path)
    p.add_argument("--subset-size", default=200, type=int)
    p.add_argument("--top-k", default=8, type=int)
    p.add_argument("--n-random-trials", default=8, type=int)
    p.add_argument("--eval-batch-size", default=4, type=int)
    p.add_argument("--image-size", default=448, type=int)
    p.add_argument("--num-classes", default=115, type=int)
    p.add_argument("--out-dir", default=None, type=Path, help="If set, dump report.json + heatmaps.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    report, anchor_hits, gt_hits, anchor_total, per_image_rows = run_diagnostic(
        ckpt_path=args.ckpt,
        plantseg_root=args.plantseg_root,
        gt_binary_dir=args.gt_binary_dir,
        subset_size=args.subset_size,
        top_k=args.top_k,
        n_random_trials=args.n_random_trials,
        eval_batch_size=args.eval_batch_size,
        device=device,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    _print_summary(report)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
        np.save(args.out_dir / "anchor_hits_heatmap.npy", anchor_hits.numpy())
        np.save(args.out_dir / "gt_hits_heatmap.npy", gt_hits.numpy())
        _dump_per_image_csv(args.out_dir / "per_image.csv", per_image_rows)
        log.info("Dumped report.json + heatmaps + per_image.csv to %s (anchor_total=%d)",
                 args.out_dir, anchor_total)


def _dump_per_image_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
