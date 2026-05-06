"""RQ2 — empirical ranking of candidate attention regularizers.

For each of four candidate variants of the D1 attention-concentration
loss, compute the variant's loss value on the actually-observed
attention tensors of ``init``, ``eq_only``, ``D1``, ``D2``, ``D3`` on
the shared 100-image val subset. We want a variant whose loss value
prefers (is LOWER on) ``eq_only`` over ``D1`` (the mode-collapse
failure) and over ``init`` (the trivially-uniform baseline).

Variants:

* ``L_ac`` (current D1): ``-mean(M_{b,q})``
* ``L_var``: ``-var_{b,q}(M_{b,q})`` averaged over batch.
* ``L_tv``:  ``+mean(|dM/dx| + |dM/dy|)`` (total variation)
* ``L_marg_H``: ``-mean(M) + beta * (log N_k - H(mu_k))`` with
  ``mu_k = mean_{b,q} attn_{b,q,k}``. Minimized when queries are sharp
  AND spread their peaks across many keys.
* ``L_entropy_balanced``: ``-mean(M) + beta * argmax-share(max_k mu_k)``
  Direct penalty on the "one key dominates" failure mode.

Each variant's value is scale-dependent; rankings are what matter.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rq2")


CKPTS: list[tuple[str, str | None]] = [
    ("init", None),
    (
        "eq_only",
        "outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/"
        "epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt",
    ),
    (
        "D1",
        "outputs/spdnet_aux_losses/spdnet_spatial_d1_ac_warmstart_20260427/"
        "checkpoints/last.ckpt",
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
]


def _load(ckpt_rel: str | None, image_size: int, num_classes: int) -> SPDNetModule:
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
    if ckpt_rel is not None:
        path = (repo_root / ckpt_rel).resolve()
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        module.load_state_dict(sd.get("state_dict", sd), strict=False)
    return module


@torch.no_grad()
def _attn_tensor(
    model, q: torch.Tensor, r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Return ``(attn_w (B, P, N_k), M (B, H, W), (H, W))``."""
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
    neg_ent = (p * p.log()).sum(dim=-1)                                  # (B, P)
    M = (1.0 + neg_ent / log_N).view(B, H, W)
    return attn_w, M, (H, W)


# ---------------------------------------------------------------------------
# Regularizer variants.  Each takes (attn_w (B, P, N_k), M (B, H, W), HW=(H, W))
# and returns a scalar python float (lower = "more preferred" by the variant).
# ---------------------------------------------------------------------------


def L_ac(attn_w: torch.Tensor, M: torch.Tensor, hw) -> float:
    """Current D1 loss: ``-mean(M)``."""
    return float(-M.mean().item())


def L_var(attn_w: torch.Tensor, M: torch.Tensor, hw) -> float:
    """``-mean_b var_{q}(M_{b, q})`` i.e. reward spatial variance of M."""
    v = M.flatten(1).var(dim=1, unbiased=False).mean()
    return float(-v.item())


def L_tv(attn_w: torch.Tensor, M: torch.Tensor, hw) -> float:
    """Minus total-variation smoothness: ``-mean(|dM/dx| + |dM/dy|)``.

    Lower value = smoother M. This is a smoothness PRIOR, not a structure
    prior; included to show it does the opposite of what D1 needs.
    """
    dx = (M[:, :, 1:] - M[:, :, :-1]).abs().mean()
    dy = (M[:, 1:, :] - M[:, :-1, :]).abs().mean()
    return float((dx + dy).item())


def L_marg_H(
    attn_w: torch.Tensor, M: torch.Tensor, hw, beta: float = 0.5,
) -> float:
    """``-mean(M) + beta * (log N_k - H(mu_k))``.

    ``mu_k = mean_{b,q} attn_{b,q,k}`` — the per-key marginal. The second
    term is KL(mu || Uniform); zero iff keys are used uniformly, > 0 when
    a small number of keys monopolise attention. The joint minimum is
    "every query peaks on a DIFFERENT key so mu stays uniform".
    """
    B, P, N_k = attn_w.shape
    log_N = math.log(N_k)
    mu = attn_w.mean(dim=(0, 1)).clamp_min(1e-12)                        # (N_k,)
    H_mu = -(mu * mu.log()).sum()
    kl_to_uniform = log_N - H_mu.item()
    return float(-M.mean().item() + beta * kl_to_uniform)


def L_argmax_share(
    attn_w: torch.Tensor, M: torch.Tensor, hw, beta: float = 1.0,
) -> float:
    """Direct penalty on the "single key dominates" failure.

    ``-mean(M) + beta * max_k(mu_k)``. ``max_k mu_k`` = 1/N_k when
    uniform, = 1 when all queries fire on key k*. Perfectly penalises
    mode collapse.
    """
    B, P, N_k = attn_w.shape
    mu = attn_w.mean(dim=(0, 1))                                         # (N_k,)
    mu_max = mu.max().item()
    return float(-M.mean().item() + beta * mu_max)


def L_dispersion(
    attn_w: torch.Tensor, M: torch.Tensor, hw, beta: float = 1.0,
) -> float:
    """Reward per-query sharpness AND different-query peak diversity.

    ``-mean(M) + beta * (1 - argmax-key-diversity)``

    where argmax-key-diversity = (unique argmax keys / total queries),
    clamped.
    """
    argmax_idx = attn_w.argmax(dim=-1)                                   # (B, P)
    flat = argmax_idx.flatten()
    unique_keys = int(flat.unique().numel())
    n_keys = int(attn_w.shape[-1])
    diversity = unique_keys / n_keys
    return float(-M.mean().item() + beta * (1.0 - diversity))


VARIANTS = {
    "L_ac (current D1)": L_ac,
    "L_var (spatial variance of M)": L_var,
    "L_tv (smoothness, expected WORSE)": L_tv,
    "L_marg_H beta=0.10": lambda a, m, h: L_marg_H(a, m, h, beta=0.10),
    "L_marg_H beta=0.20": lambda a, m, h: L_marg_H(a, m, h, beta=0.20),
    "L_marg_H beta=0.25": lambda a, m, h: L_marg_H(a, m, h, beta=0.25),
    "L_marg_H beta=0.30": lambda a, m, h: L_marg_H(a, m, h, beta=0.30),
    "L_marg_H beta=0.50": lambda a, m, h: L_marg_H(a, m, h, beta=0.50),
    "L_argmax_share beta=1.0": L_argmax_share,
    "L_argmax_share beta=2.0": lambda a, m, h: L_argmax_share(a, m, h, beta=2.0),
    "L_dispersion beta=1.0": L_dispersion,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantseg-root", default="data/plantsegv3")
    parser.add_argument(
        "--gt-binary-dir",
        default="outputs/plantseg_binary_mc115/gt_binary_val",
    )
    parser.add_argument("--out-dir", default="outputs/diagnostics/rq2_attention_variants")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=115)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Building shared OnlineCAMIoU subset (size=%d)", args.subset_size)
    loc = OnlineCAMIoU(
        plantseg_root=str(args.plantseg_root),
        gt_binary_dir=str(args.gt_binary_dir),
        num_classes=args.num_classes,
        subset_size=args.subset_size,
        seed=args.seed,
        every_n_epochs=1,
        image_size=args.image_size,
        eval_batch_size=args.batch_size,
    )
    N = loc.query_images.shape[0]

    results: dict[str, dict[str, float]] = {}                            # label -> variant -> value
    for label, rel_path in CKPTS:
        log.info("Loading %s", label)
        module = _load(rel_path, args.image_size, args.num_classes).to(device).eval()
        # Accumulate losses over batches (attn tensor is too big to keep all).
        variant_running: dict[str, list[float]] = {v: [] for v in VARIANTS}
        for start in range(0, N, args.batch_size):
            stop = min(start + args.batch_size, N)
            q = loc.query_images[start:stop].to(device, non_blocking=True)
            r = loc.ref_images[start:stop].to(device, non_blocking=True)
            attn_w, M, hw = _attn_tensor(module.model, q, r)
            for name, fn in VARIANTS.items():
                variant_running[name].append(fn(attn_w, M, hw))
        # Mean across batches.
        results[label] = {v: float(np.mean(vs)) for v, vs in variant_running.items()}
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "ranking.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s/ranking.json", out_dir)

    # Pretty print
    headers = ["variant"] + [label for label, _ in CKPTS]
    rows = []
    for v_name in VARIANTS:
        row = [v_name]
        for label, _ in CKPTS:
            row.append(f"{results[label][v_name]:+.4f}")
        rows.append(row)
    # Markdown table for the notes.
    widths = [max(len(c) for c in col) for col in zip(*([headers] + rows))]
    def _fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
    md_lines = [_fmt(headers), "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"]
    for row in rows:
        md_lines.append(_fmt(row))
    md_block = "\n".join(md_lines)
    (out_dir / "ranking_table.md").write_text(md_block + "\n")
    log.info("Wrote ranking table:\n%s", md_block)

    # Ranking analysis: for each variant, does eq_only have lower loss than D1?
    verdicts = {}
    for v_name in VARIANTS:
        v_init = results["init"][v_name]
        v_eq = results["eq_only"][v_name]
        v_d1 = results["D1"][v_name]
        v_d2 = results["D2"][v_name]
        v_d3 = results["D3"][v_name]
        prefers_eq_over_d1 = v_eq < v_d1
        prefers_eq_over_init = v_eq < v_init
        prefers_structured = v_eq < min(v_init, v_d1, v_d2, v_d3)
        verdicts[v_name] = {
            "init": v_init,
            "eq_only": v_eq,
            "D1": v_d1,
            "D2": v_d2,
            "D3": v_d3,
            "ranks_eq_below_D1": bool(prefers_eq_over_d1),
            "ranks_eq_below_init": bool(prefers_eq_over_init),
            "ranks_eq_best": bool(prefers_structured),
        }
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    log.info("wrote verdicts.json")


if __name__ == "__main__":
    main()
