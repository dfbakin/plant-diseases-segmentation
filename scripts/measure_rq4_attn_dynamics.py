"""RQ4 — Attention-map dynamics measurement (read-only).

For each of a configurable set of SPDNet checkpoints (plus a freshly
initialised "init" model with a pretrained ResNet50 backbone and random
fusion/classifier), compute on the shared 100-image val subset used by
``OnlineCAMIoU``:

* ``H_q``  -- per-query attention entropy (nats).
* ``M_q``  -- per-query normalised concentration in ``[0, 1]``
  (``1 - H_q / log N_k``).
* ``mu_k`` -- per-key marginal of attention (``mean_{b,q} attn_{b,q,k}``),
  plus its entropy as a diagnostic for mode-collapse on a single key.
* ``argmax_k``: count of how many queries pick each key as argmax. If a
  single key dominates, we're in the D1-style patch-to-key collapse.

The script classifies each checkpoint into one of three uniformity
regimes and writes a per-checkpoint JSON plus a CSV trajectory.

Additionally, for the ``eq_only`` checkpoint only, it performs a
"rogue-idea feasibility" probe: computes gradient magnitudes of
``L_cls`` and ``L_ac`` on the cross-attention in-projection weight and
reports the ratio and cosine angle. This tells us whether adding a
small ``L_ac`` at training time would interfere with classification.

Usage::

    python scripts/measure_rq4_attn_dynamics.py \\
        --plantseg-root data/plantsegv3 \\
        --gt-binary-dir outputs/plantseg_binary_mc115/gt_binary_val \\
        --out-dir outputs/diagnostics/rq4_attn_dynamics \\
        --subset-size 100 --eval-batch-size 8

Outputs:
    outputs/diagnostics/rq4_attn_dynamics/<label>.json
    outputs/diagnostics/rq4_attn_dynamics/trajectory.csv
    reports/notes/rq4_attn_dynamics.md  (human-readable narrative)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.conf.spdnet import SPDNetSpatialLossesConfig  # noqa: E402
from src.wsss.spdnet.lightning import SPDNetModule  # noqa: E402
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU  # noqa: E402
from src.wsss.spdnet.spatial_losses import (  # noqa: E402
    attention_concentration_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rq4")


# ----- Checkpoint catalog (label -> file path). None path = fresh init. -----

DEFAULT_CKPTS: list[tuple[str, str | None]] = [
    ("init", None),
    (
        "eq_only",
        "outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/"
        "epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt",
    ),
    (
        "eq_con_warmstart",
        "outputs/spdnet_aux_losses/spdnet_spatial_eq_con_warmstart_20260425/"
        "checkpoints/last.ckpt",
    ),
    (
        "eq_con_warmstart_ep0",
        "outputs/spdnet_aux_losses/spdnet_spatial_eq_con_warmstart_20260425/"
        "checkpoints/epoch=epoch=00-val_mAP=val/mAP=0.8604.ckpt",
    ),
    (
        "D1",
        "outputs/spdnet_aux_losses/spdnet_spatial_d1_ac_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D1_ep0",
        "outputs/spdnet_aux_losses/spdnet_spatial_d1_ac_warmstart_20260427/"
        "checkpoints/epoch=epoch=00-val_mAP=val/mAP=0.8541.ckpt",
    ),
    (
        "D2",
        "outputs/spdnet_aux_losses/spdnet_spatial_d2_mask_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D3",
        "outputs/spdnet_aux_losses/spdnet_spatial_d3_d2plus_union_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D4_main",
        "outputs/spdnet_aux_losses/spdnet_spatial_d4_main_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D4_attn_only",
        "outputs/spdnet_aux_losses/spdnet_spatial_d4_attn_only_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D4_ac_safe",
        "outputs/spdnet_aux_losses/spdnet_spatial_d4_ac_safe_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
    (
        "D4_int",
        "outputs/spdnet_aux_losses/spdnet_spatial_d4_int_warmstart_20260427/"
        "checkpoints/last.ckpt",
    ),
]


def _build_module(ckpt_path: Path | None, num_classes: int, image_size: int) -> SPDNetModule:
    """Instantiate SPDNetModule; load ckpt weights when provided."""
    module = SPDNetModule(
        num_classes=num_classes,
        fpn_channels=256,
        fusion_mode="spatial",
        losses_cfg=SPDNetSpatialLossesConfig(
            lambda_eq=0.0, lambda_con=0.0, lambda_distill=0.0,
        ),
        online_loc_metric=None,
        image_size=image_size,
    )
    if ckpt_path is not None:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing, unexpected = module.load_state_dict(
            sd.get("state_dict", sd), strict=False,
        )
        if missing:
            log.warning("[%s] load missing %d keys (first 3): %s",
                        ckpt_path.name, len(missing), missing[:3])
        if unexpected:
            log.warning("[%s] load unexpected %d keys (first 3): %s",
                        ckpt_path.name, len(unexpected), unexpected[:3])
    return module


@torch.no_grad()
def _forward_attn_weights(
    model, q: torch.Tensor, r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(attn_w (B, P, N_k), attn_map (B, H, W) in [0, 1])``.

    Uses the second (eval-mode) MHA call from ``SpatialCrossAttention``
    so dropout is disabled and ``attn_w`` rows are true probability
    distributions.
    """
    sca = model.spatial_attn
    feats = model.extract_features(q)
    fpn_out = model.fpn(feats)
    mse_out = [model.mse(p) for p in fpn_out]
    query_merged = model._merge_fpn(mse_out)

    r_fpn = model.fpn(model.extract_features(r))
    r_mse = [model.mse(p) for p in r_fpn]
    ref_merged = model._merge_fpn(r_mse)

    B, C_in, H, W = query_merged.shape
    ref_pooled = sca.pool(ref_merged)
    q_tok = query_merged.flatten(2).permute(0, 2, 1)
    kv_tok = ref_pooled.flatten(2).permute(0, 2, 1)
    q_tok = sca.norm_q(q_tok)
    kv_tok = sca.norm_kv(kv_tok)

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
    neg_ent = (p * p.log()).sum(dim=-1)                             # (B, P)
    attn_map = (1.0 + neg_ent / log_N).view(B, H, W)                # in [0, 1]
    return attn_w, attn_map


def _classify_regime(
    h_mean: float, log_N: float, m_mean: float, mu_ent: float,
) -> str:
    """Label the uniformity regime for a checkpoint."""
    h_ratio = h_mean / log_N            # 1.0 = uniform per query
    mu_ratio = mu_ent / log_N           # 1.0 = uniform over keys
    if h_ratio > 0.85 and mu_ratio > 0.85:
        return "patch-to-patch uniform (both H_q and H(mu) near log N_k)"
    if m_mean > 0.8 and mu_ratio < 0.25:
        return "patch-to-key collapse (queries sharp, single key dominant)"
    if m_mean > 0.5 and mu_ratio < 0.5:
        return "mild collapse (sharp per-query, uneven key usage)"
    if m_mean < 0.3 and mu_ratio > 0.7:
        return "structurally diffuse (low per-query concentration, spread keys)"
    return "structured/intermediate"


def measure_ckpt(
    module: SPDNetModule,
    loc: OnlineCAMIoU,
    device: torch.device,
    eval_batch_size: int,
) -> dict:
    """Collect attention-map diagnostics for one checkpoint."""
    module.eval().to(device)
    model = module.model

    N = loc.query_images.shape[0]
    per_q_H: list[float] = []                      # (N_total_queries,) list of H(p_q)
    per_q_M: list[float] = []                      # normalised concentration
    marginal_sum: torch.Tensor | None = None
    marginal_count: int = 0
    argmax_hist: torch.Tensor | None = None
    N_key_global: int | None = None
    log_N_global: float | None = None

    for start in range(0, N, eval_batch_size):
        stop = min(start + eval_batch_size, N)
        q = loc.query_images[start:stop].to(device, non_blocking=True)
        r = loc.ref_images[start:stop].to(device, non_blocking=True)
        attn_w, attn_map = _forward_attn_weights(model, q, r)
        B, P, N_key = attn_w.shape
        if N_key_global is None:
            N_key_global = int(N_key)
            log_N_global = float(math.log(N_key_global))
            marginal_sum = torch.zeros(N_key, device=device, dtype=torch.float64)
            argmax_hist = torch.zeros(N_key, device=device, dtype=torch.float64)

        p = attn_w.clamp_min(1e-12)
        H_q = -(p * p.log()).sum(dim=-1)             # (B, P) in [0, log N]
        M_q = 1.0 - H_q / log_N_global               # (B, P) in [0, 1]
        per_q_H.extend(H_q.flatten().cpu().tolist())
        per_q_M.extend(M_q.flatten().cpu().tolist())

        marginal_sum.add_(attn_w.double().sum(dim=(0, 1)))
        marginal_count += B * P

        argmax_idx = attn_w.argmax(dim=-1)           # (B, P)
        argmax_hist.scatter_add_(
            0,
            argmax_idx.flatten().long(),
            torch.ones_like(argmax_idx.flatten(), dtype=torch.float64),
        )

    assert marginal_sum is not None and argmax_hist is not None
    assert log_N_global is not None and N_key_global is not None

    mu = (marginal_sum / float(marginal_count)).cpu().numpy()          # (N_key,)
    arg_hist = argmax_hist.cpu().numpy()
    arg_prob = arg_hist / max(1, arg_hist.sum())
    H_q_arr = np.asarray(per_q_H, dtype=np.float64)
    M_q_arr = np.asarray(per_q_M, dtype=np.float64)
    mu_safe = np.clip(mu, 1e-12, None)
    mu_entropy = float(-(mu_safe * np.log(mu_safe)).sum())
    arg_safe = np.clip(arg_prob, 1e-12, None)
    arg_entropy = float(-(arg_safe * np.log(arg_safe)).sum())

    return {
        "n_queries_total": int(H_q_arr.size),
        "N_key": int(N_key_global),
        "log_N_key": float(log_N_global),
        "H_q_mean_nats": float(H_q_arr.mean()),
        "H_q_std_nats": float(H_q_arr.std()),
        "H_q_median_nats": float(np.median(H_q_arr)),
        "H_q_over_logN_mean": float(H_q_arr.mean() / log_N_global),
        "M_q_mean": float(M_q_arr.mean()),
        "M_q_std": float(M_q_arr.std()),
        "M_q_min": float(M_q_arr.min()),
        "M_q_max": float(M_q_arr.max()),
        "M_q_p10": float(np.percentile(M_q_arr, 10)),
        "M_q_p50": float(np.percentile(M_q_arr, 50)),
        "M_q_p90": float(np.percentile(M_q_arr, 90)),
        "mu_entropy_nats": mu_entropy,
        "mu_entropy_over_logN": mu_entropy / log_N_global,
        "mu_max": float(mu.max()),
        "mu_min": float(mu.min()),
        "mu_top5_frac": float(np.sort(mu)[-5:].sum()),
        "argmax_key_entropy_nats": arg_entropy,
        "argmax_key_entropy_over_logN": arg_entropy / log_N_global,
        "argmax_top1_share": float(arg_prob.max()),
        "argmax_top5_share": float(np.sort(arg_prob)[-5:].sum()),
        "regime": _classify_regime(
            h_mean=float(H_q_arr.mean()),
            log_N=log_N_global,
            m_mean=float(M_q_arr.mean()),
            mu_ent=mu_entropy,
        ),
    }


def rogue_feasibility_probe(
    module: SPDNetModule,
    loc: OnlineCAMIoU,
    device: torch.device,
    n_images: int,
    eval_batch_size: int,
) -> dict:
    """Measure |grad L_cls| vs |grad L_ac| on cross-attn in-proj weight.

    Run on eq_only. Uses the module's training step (minus the optimizer
    update) to stay bug-compatible with the training forward.
    """
    module.train()
    module.to(device)
    model = module.model
    sca_params = list(model.spatial_attn.cross_attn.parameters())
    sca_inproj = model.spatial_attn.cross_attn.in_proj_weight

    N = min(n_images, loc.query_images.shape[0])
    norms_cls: list[float] = []
    norms_ac: list[float] = []
    cos_vals: list[float] = []

    for start in range(0, N, eval_batch_size):
        stop = min(start + eval_batch_size, N)
        q = loc.query_images[start:stop].to(device, non_blocking=True)
        r = loc.ref_images[start:stop].to(device, non_blocking=True)
        labels = loc.query_labels[start:stop].to(device, non_blocking=True)

        # Forward with attention map, gradients enabled.
        feats = model.extract_merged_features(q, [r], return_attn=True)
        fused = feats["fused"]
        attn_map = feats["attn_map"]
        logits = model.classifier(fused.mean(dim=[2, 3]))
        L_cls = module.criterion(logits, labels)
        L_ac = attention_concentration_loss(attn_map)

        g_cls = torch.autograd.grad(
            L_cls, sca_inproj, retain_graph=True, create_graph=False,
        )[0].detach().flatten()
        g_ac = torch.autograd.grad(
            L_ac, sca_inproj, retain_graph=False, create_graph=False,
        )[0].detach().flatten()
        # Clear the computation graph so the next batch starts clean.
        del feats, fused, attn_map, logits, L_cls, L_ac

        n_cls = float(g_cls.norm().item())
        n_ac = float(g_ac.norm().item())
        denom = max(n_cls * n_ac, 1e-12)
        cos_theta = float((g_cls @ g_ac).item() / denom)
        norms_cls.append(n_cls)
        norms_ac.append(n_ac)
        cos_vals.append(cos_theta)

    nc = np.asarray(norms_cls)
    na = np.asarray(norms_ac)
    cs = np.asarray(cos_vals)
    return {
        "n_batches": int(nc.size),
        "grad_L_cls_norm_mean": float(nc.mean()),
        "grad_L_cls_norm_std": float(nc.std()),
        "grad_L_ac_norm_mean": float(na.mean()),
        "grad_L_ac_norm_std": float(na.std()),
        "ratio_ac_over_cls_mean": float((na / np.clip(nc, 1e-12, None)).mean()),
        "cos_angle_mean": float(cs.mean()),
        "cos_angle_std": float(cs.std()),
        "param_name": "spatial_attn.cross_attn.in_proj_weight",
        "param_numel": int(sca_inproj.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantseg-root", default="data/plantsegv3")
    parser.add_argument(
        "--gt-binary-dir",
        default="outputs/plantseg_binary_mc115/gt_binary_val",
    )
    parser.add_argument("--out-dir", default="outputs/diagnostics/rq4_attn_dynamics")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=115)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rogue-eq-only", action="store_true", default=True,
        help="Run the rogue-idea feasibility probe on eq_only.",
    )
    parser.add_argument("--rogue-n-images", type=int, default=48)
    parser.add_argument(
        "--skip-ckpts", nargs="*", default=[],
        help="Labels to skip (e.g. --skip-ckpts eq_con_warmstart_ep0)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    log.info("Building shared OnlineCAMIoU subset (size=%d)", args.subset_size)
    loc = OnlineCAMIoU(
        plantseg_root=str(args.plantseg_root),
        gt_binary_dir=str(args.gt_binary_dir),
        num_classes=args.num_classes,
        subset_size=args.subset_size,
        seed=args.seed,
        every_n_epochs=1,
        image_size=args.image_size,
        eval_batch_size=args.eval_batch_size,
    )
    log.info("Loaded %d query-reference pairs", loc.query_images.shape[0])

    trajectory_rows: list[dict] = []
    rogue_result: dict | None = None

    for label, rel_path in DEFAULT_CKPTS:
        if label in args.skip_ckpts:
            log.info("Skipping %s (user request)", label)
            continue
        ckpt_path: Path | None = None
        if rel_path is not None:
            ckpt_path = (repo_root / rel_path).resolve()
            if not ckpt_path.exists():
                log.warning("MISSING ckpt for %s: %s", label, ckpt_path)
                continue

        log.info("Measuring %s (ckpt=%s)", label, ckpt_path)
        module = _build_module(ckpt_path, args.num_classes, args.image_size)
        metrics = measure_ckpt(module, loc, device, args.eval_batch_size)
        metrics["label"] = label
        metrics["ckpt_path"] = str(ckpt_path) if ckpt_path is not None else None
        (out_dir / f"{label}.json").write_text(json.dumps(metrics, indent=2))
        trajectory_rows.append(metrics)

        if label == "eq_only" and args.rogue_eq_only:
            log.info("Running rogue feasibility probe on eq_only")
            rogue_result = rogue_feasibility_probe(
                module, loc, device, args.rogue_n_images, args.eval_batch_size,
            )
            (out_dir / "rogue_feasibility.json").write_text(
                json.dumps(rogue_result, indent=2)
            )
            log.info("rogue: |grad L_cls|=%.3e  |grad L_ac|=%.3e  ratio=%.2f  cos=%.3f",
                     rogue_result["grad_L_cls_norm_mean"],
                     rogue_result["grad_L_ac_norm_mean"],
                     rogue_result["ratio_ac_over_cls_mean"],
                     rogue_result["cos_angle_mean"])

        # Free memory between checkpoints.
        del module
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Wide CSV for easy inspection.
    csv_path = out_dir / "trajectory.csv"
    if trajectory_rows:
        cols = [
            "label", "n_queries_total", "N_key", "log_N_key",
            "H_q_mean_nats", "H_q_std_nats", "H_q_over_logN_mean",
            "M_q_mean", "M_q_std", "M_q_p10", "M_q_p50", "M_q_p90",
            "mu_entropy_nats", "mu_entropy_over_logN", "mu_max",
            "argmax_top1_share", "argmax_top5_share", "regime",
        ]
        lines = [",".join(cols)]
        for row in trajectory_rows:
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            lines.append(",".join(vals))
        csv_path.write_text("\n".join(lines) + "\n")
        log.info("wrote %s (%d rows)", csv_path, len(trajectory_rows))

    # Compact human-readable summary stdout.
    log.info("\n--- RQ4 summary ---")
    for row in trajectory_rows:
        log.info(
            "%-24s N_key=%d  H_q/logN=%.3f  M_q_mean=%.3f  H(mu)/logN=%.3f  "
            "argmax_top1=%.3f  regime=%s",
            row["label"], row["N_key"], row["H_q_over_logN_mean"],
            row["M_q_mean"], row["mu_entropy_over_logN"],
            row["argmax_top1_share"], row["regime"],
        )

    if rogue_result is not None:
        log.info("rogue probe: ratio |grad L_ac|/|grad L_cls| = %.3f, cos = %.3f",
                 rogue_result["ratio_ac_over_cls_mean"],
                 rogue_result["cos_angle_mean"])


if __name__ == "__main__":
    main()
