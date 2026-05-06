"""RQ1 — Gradient-magnitude spectrum across all auxiliary SPDNet losses.

Loads the eq_only checkpoint, enables every auxiliary loss at
lambda=1.0, runs ``--n-steps`` forward passes on the 100-image val
subset (batches of ``--batch-size``), and for each loss computes the
per-parameter-group gradient norm and its cosine angle against the
classification gradient.

Outputs:

* ``outputs/diagnostics/grad_spectrum/eq_only/grad_norms.csv``
    long format: (step, loss_name, param_group, l2, rms,
    cos_vs_cls, n_params)
* ``outputs/diagnostics/grad_spectrum/eq_only/pairwise_angles.csv``
    long format: (step, param_group, loss_a, loss_b, cos_theta)
* ``outputs/diagnostics/grad_spectrum/eq_only/SUMMARY.md``
    headline table with recommended lambda_k_star values.

The full val subset is not shuffled between steps to keep the run
reproducible; gradient variance across steps reflects only the
stochastic content (dropout in main MHA path, label balance).

Usage::

    python scripts/measure_grad_spectrum.py \
        --ckpt outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt \
        --n-steps 20 --batch-size 8

No optimizer updates. No training. Entirely read-only probe.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.conf.spdnet import SPDNetSpatialLossesConfig  # noqa: E402
from src.wsss.spdnet import equivariance_transforms as ET  # noqa: E402
from src.wsss.spdnet.lightning import SPDNetModule  # noqa: E402
from src.wsss.spdnet.online_loc_metric import OnlineCAMIoU  # noqa: E402
from src.wsss.spdnet.spatial_losses import (  # noqa: E402
    EMATeacher,
    ProjectionHead,
    attention_concentration_loss,
    attention_marginal_entropy_loss,
    cam_pseudo_mask_loss,
    equivariance_loss,
    patch_contrastive_loss,
    self_distillation_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rq1")


# D4 adds L_marg_H; measure it alongside the other five losses so we can
# pick lambda_marg_H with the same equalise-RMS recipe.
LOSS_NAMES = ["L_cls", "L_eq", "L_con", "L_ac", "L_marg_H", "L_mask", "L_dist"]


@dataclass
class ParamGroup:
    name: str
    params: list[torch.nn.Parameter]

    def numel(self) -> int:
        return sum(int(p.numel()) for p in self.params)

    def flatten_grad(self, grad_list: list[torch.Tensor | None]) -> torch.Tensor:
        """Concatenate per-parameter grad tensors into a single flat vector.

        ``None`` grads are replaced with zeros (loss independent of param).
        """
        flats = []
        for p, g in zip(self.params, grad_list):
            if g is None:
                flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flats.append(g.detach().flatten())
        return torch.cat(flats)


def build_param_groups(module: SPDNetModule) -> list[ParamGroup]:
    """Return the three canonical parameter groups for gradient accounting."""
    fusion_params = list(module.model.spatial_attn.parameters())
    cls_params = [module.model.classifier.weight, module.model.classifier.bias]
    proj_params: list[torch.nn.Parameter] = []
    if getattr(module, "proj_head", None) is not None:
        proj_params = list(module.proj_head.parameters())
    return [
        ParamGroup("fusion", fusion_params),
        ParamGroup("classifier", cls_params),
        ParamGroup("proj_head", proj_params),
        ParamGroup(
            "all_trainable",
            [p for p in module.parameters() if p.requires_grad],
        ),
    ]


def forward_all_losses(
    module: SPDNetModule,
    q: torch.Tensor,
    r: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: int,
) -> dict[str, torch.Tensor]:
    """Reproduce the training-step forward and return each loss scalar.

    Only differences from ``training_step``:
      * EMA teacher warmup is ignored -- L_dist is always computed if
        the teacher exists.
      * The contrastive warmup/schedules are ignored -- we log raw loss
        magnitudes, not the effective-lambda products.

    Every loss in the returned dict shares a single forward graph so
    ``torch.autograd.grad(L_k, params, retain_graph=True)`` can be
    called five times in a row.
    """
    model = module.model
    cfg = module.losses_cfg

    feats = model.extract_merged_features(q, [r], return_attn=True)
    fused = feats["fused"]
    attn_orig = feats["attn_map"]
    attn_w = feats["attn_w"]
    pooled = fused.mean(dim=[2, 3])
    logits = model.classifier(pooled)
    L_cls = module.criterion(logits, labels)

    t_choices = cfg.equivariance_transforms if cfg.equivariance_transforms else [0]
    t_id = int(t_choices[batch_idx % len(t_choices)])
    q_aug = ET.apply(q, t_id)
    attn_aug = model.attention_map(
        q_aug, ref_merged_cached=feats.get("ref_merged"),
    )
    L_eq = equivariance_loss(attn_orig, attn_aug, t_id)

    L_ac = attention_concentration_loss(attn_orig)
    L_marg_H = attention_marginal_entropy_loss(attn_w, beta=cfg.marg_H_beta)

    L_mask = cam_pseudo_mask_loss(
        p3_query=feats["query_merged"],
        p4_fused=fused,
        cls_weight=model.classifier.weight,
        labels=labels,
        alpha_pos=cfg.mask_alpha_pos,
        beta_neg=cfg.mask_beta_neg,
        use_intersection=cfg.mask_use_intersection,
        mask_combiner=cfg.mask_combiner,
    )

    L_con = patch_contrastive_loss(
        p3_query=feats["query_merged"],
        p4_fused=fused,
        cls_weight=model.classifier.weight,
        labels=labels,
        proj_head=module.proj_head,
        top_k=cfg.con_top_K,
        m_negatives=cfg.con_M_negatives,
        temperature=cfg.con_temperature,
        anchor_source=cfg.con_anchor_source,
    )

    if module.ema_teacher is not None:
        S_student = torch.einsum(
            "nc,bchw->bnhw", model.classifier.weight, fused,
        )
        S_teacher = module.ema_teacher(q, [r])
        P_actual = S_student.shape[-1] * S_student.shape[-2]
        if module.distill_center is None or module.distill_center.numel() != P_actual:
            module.distill_center = torch.zeros(
                P_actual, device=S_student.device, dtype=S_student.dtype,
            )
        L_dist = self_distillation_loss(
            s_student=S_student,
            s_teacher=S_teacher,
            labels=labels,
            center=module.distill_center,
            center_beta=cfg.distill_center_beta,
            T_teacher=cfg.distill_T_teacher,
            T_student=cfg.distill_T_student,
        )
    else:
        L_dist = torch.zeros((), device=q.device, dtype=q.dtype)

    return {
        "L_cls": L_cls,
        "L_eq": L_eq,
        "L_ac": L_ac,
        "L_marg_H": L_marg_H,
        "L_mask": L_mask,
        "L_con": L_con,
        "L_dist": L_dist,
    }


def measure_step(
    module: SPDNetModule,
    q: torch.Tensor,
    r: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: int,
    param_groups: list[ParamGroup],
) -> tuple[list[dict], list[dict], dict[str, float]]:
    """One measurement forward -> grads for every (loss, param_group) pair.

    Returns:
        rows_norms: long-format rows for ``grad_norms.csv``
        rows_angles: pairwise cosine angles between loss-pairs
        raw_losses: per-loss scalar value (for bookkeeping / debug)
    """
    losses = forward_all_losses(module, q, r, labels, batch_idx)
    raw_losses = {k: float(v.detach().item()) for k, v in losses.items()}

    # Per (loss, param_group) gradient vectors.  Shape: dict[(loss, group)] -> flat tensor.
    grad_vecs: dict[tuple[str, str], torch.Tensor] = {}
    for loss_name in LOSS_NAMES:
        loss_tensor = losses[loss_name]
        # Some losses degenerate to grad-preserving zero; torch.autograd.grad
        # still returns zero tensors for them.
        for group in param_groups:
            grads = torch.autograd.grad(
                loss_tensor,
                group.params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            grad_vecs[(loss_name, group.name)] = group.flatten_grad(grads)

    rows_norms: list[dict] = []
    for loss_name in LOSS_NAMES:
        for group in param_groups:
            v = grad_vecs[(loss_name, group.name)]
            n_params = v.numel()
            l2 = float(v.norm().item())
            rms = float((v.pow(2).sum().item() / max(n_params, 1)) ** 0.5)
            v_cls = grad_vecs[("L_cls", group.name)]
            denom = max(float(v.norm().item()) * float(v_cls.norm().item()), 1e-18)
            cos = float((v @ v_cls).item() / denom)
            rows_norms.append({
                "batch_idx": batch_idx,
                "loss_name": loss_name,
                "param_group": group.name,
                "l2": l2,
                "rms": rms,
                "cos_vs_cls": cos,
                "n_params": n_params,
                "raw_loss_value": raw_losses[loss_name],
            })

    # Pairwise angles for RQ3.
    rows_angles: list[dict] = []
    for i, a in enumerate(LOSS_NAMES):
        for b in LOSS_NAMES[i + 1 :]:
            for group in param_groups:
                va = grad_vecs[(a, group.name)]
                vb = grad_vecs[(b, group.name)]
                denom = max(float(va.norm().item()) * float(vb.norm().item()), 1e-18)
                cos = float((va @ vb).item() / denom)
                rows_angles.append({
                    "batch_idx": batch_idx,
                    "loss_a": a,
                    "loss_b": b,
                    "param_group": group.name,
                    "cos_theta": cos,
                })

    # Free graph references.
    for v in losses.values():
        del v
    return rows_norms, rows_angles, raw_losses


def recommend_lambdas(
    rows: list[dict], target_group: str, target_ratio_vs_cls: float = 1.0,
) -> dict[str, float]:
    """Return ``lambda_k^\\star`` that equalize RMS(lambda_k * grad L_k).

    lambda_k^\\star = target_ratio_vs_cls * RMS(grad L_cls) / RMS(grad L_k).
    """
    by_loss: dict[str, list[float]] = {ln: [] for ln in LOSS_NAMES}
    for row in rows:
        if row["param_group"] != target_group:
            continue
        by_loss[row["loss_name"]].append(row["rms"])
    rms_cls = float(np.mean(by_loss["L_cls"]))
    out = {}
    for ln in LOSS_NAMES:
        rms_k = float(np.mean(by_loss[ln])) if by_loss[ln] else 0.0
        if ln == "L_cls":
            out[ln] = 1.0
        elif rms_k <= 1e-20:
            out[ln] = float("nan")
        else:
            out[ln] = target_ratio_vs_cls * rms_cls / rms_k
    return out


def write_summary(
    out_dir: Path,
    rows_norms: list[dict],
    rows_angles: list[dict],
    ckpt_path: Path,
    n_steps: int,
    batch_size: int,
    active_cfg: SPDNetSpatialLossesConfig,
    recommended: dict[str, dict[str, float]],
) -> None:
    """Write SUMMARY.md with headline tables."""
    summary = out_dir / "SUMMARY.md"

    by_group = {"fusion": {}, "classifier": {}, "proj_head": {}, "all_trainable": {}}
    for row in rows_norms:
        g = row["param_group"]
        ln = row["loss_name"]
        by_group[g].setdefault(ln, {"l2": [], "rms": [], "cos": [], "raw": []})
        by_group[g][ln]["l2"].append(row["l2"])
        by_group[g][ln]["rms"].append(row["rms"])
        by_group[g][ln]["cos"].append(row["cos_vs_cls"])
        by_group[g][ln]["raw"].append(row["raw_loss_value"])

    lines: list[str] = []
    lines.append("# RQ1 — gradient-magnitude spectrum on eq_only")
    lines.append("")
    lines.append(f"Checkpoint: `{ckpt_path}`")
    lines.append(f"Batches: {n_steps}, batch size: {batch_size}")
    lines.append(f"All aux lambdas set to 1.0; raw loss values are logged unscaled.")
    lines.append("")
    lines.append(
        f"Active losses_cfg snapshot: lambda_eq={active_cfg.lambda_eq}, "
        f"lambda_con={active_cfg.lambda_con}, lambda_ac={active_cfg.lambda_ac}, "
        f"lambda_mask={active_cfg.lambda_mask}, lambda_distill={active_cfg.lambda_distill}"
    )
    lines.append("")

    for group_name in ("fusion", "classifier", "proj_head", "all_trainable"):
        stats = by_group[group_name]
        if not stats:
            continue
        lines.append(f"## Param group: `{group_name}`")
        lines.append("")
        lines.append(
            "| loss | raw mean | L2 mean | RMS mean | RMS rel to cls | cos vs cls (mean) |"
        )
        lines.append("|---|---|---|---|---|---|")
        rms_cls = float(np.mean(stats["L_cls"]["rms"])) if stats.get("L_cls") else float("nan")
        for ln in LOSS_NAMES:
            if ln not in stats:
                continue
            l2 = float(np.mean(stats[ln]["l2"]))
            rms = float(np.mean(stats[ln]["rms"]))
            cos = float(np.mean(stats[ln]["cos"]))
            raw = float(np.mean(stats[ln]["raw"]))
            rel = rms / rms_cls if rms_cls > 0 else float("nan")
            lines.append(
                f"| `{ln}` | {raw:.3e} | {l2:.3e} | {rms:.3e} | {rel:.3f} | {cos:.3f} |"
            )
        lines.append("")

    lines.append("## Recommended lambdas (equalize RMS to L_cls)")
    lines.append("")
    lines.append("| param group | " + " | ".join(f"λ*({ln})" for ln in LOSS_NAMES if ln != "L_cls") + " |")
    lines.append("|---|" + "---|" * (len(LOSS_NAMES) - 1))
    for gn in ("fusion", "classifier", "proj_head", "all_trainable"):
        rec = recommended.get(gn, {})
        cells = [f"{gn}"] + [
            (f"{rec.get(ln, float('nan')):.4f}" if not math.isnan(rec.get(ln, float("nan"))) else "n/a")
            for ln in LOSS_NAMES if ln != "L_cls"
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Pairwise angle summary on fusion group.
    angles_fusion = [row for row in rows_angles if row["param_group"] == "fusion"]
    if angles_fusion:
        lines.append("## Pairwise cosine angles on `fusion` parameters")
        lines.append("")
        lines.append("| loss_a vs loss_b | cos mean | frac obtuse (cos<0) |")
        lines.append("|---|---|---|")
        by_pair: dict[tuple[str, str], list[float]] = {}
        for r in angles_fusion:
            by_pair.setdefault((r["loss_a"], r["loss_b"]), []).append(r["cos_theta"])
        for (a, b), vs in by_pair.items():
            arr = np.asarray(vs)
            obtuse = float((arr < 0).mean())
            lines.append(f"| {a} × {b} | {float(arr.mean()):.3f} | {obtuse:.2f} |")
        lines.append("")

    summary.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=(
            "outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/"
            "epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt"
        ),
    )
    parser.add_argument("--plantseg-root", default="data/plantsegv3")
    parser.add_argument(
        "--gt-binary-dir",
        default="outputs/plantseg_binary_mc115/gt_binary_val",
    )
    parser.add_argument("--out-dir", default="outputs/diagnostics/grad_spectrum/eq_only")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=115)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("device=%s", device)
    log.info("Loading eq_only ckpt: %s", args.ckpt)

    # Enable ALL aux losses at lambda=1 so each loss is actually computed.
    cfg = SPDNetSpatialLossesConfig(
        lambda_eq=1.0,
        lambda_con=1.0,
        lambda_ac=1.0,
        lambda_marg_H=1.0,
        marg_H_beta=0.25,
        lambda_mask=1.0,
        lambda_distill=1.0,
        distill_warmup_epochs=0,
    )
    module = SPDNetModule(
        num_classes=args.num_classes,
        fpn_channels=256,
        fusion_mode="spatial",
        losses_cfg=cfg,
        online_loc_metric=None,
        image_size=args.image_size,
    )
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(
        sd.get("state_dict", sd), strict=False,
    )
    # The checkpoint has no proj_head or ema_teacher; those start fresh.
    log.info("load_state_dict: missing=%d, unexpected=%d", len(missing), len(unexpected))
    module.train().to(device)

    # Rebuild proj_head + EMA teacher fresh if they weren't created at __init__.
    if module.proj_head is None:
        module.proj_head = ProjectionHead(
            in_channels=256,
            out_channels=cfg.con_projection_dim,
        ).to(device)
    if module.ema_teacher is None:
        module.ema_teacher = EMATeacher(module.model, alpha=cfg.ema_alpha).to(device)
        # One update so teacher != student-at-init-of-ema isn't an issue;
        # EMATeacher.__init__ already copies student -> teacher, so no-op.

    # Build val subset for inputs.
    log.info("Building OnlineCAMIoU subset (size=%d)", args.subset_size)
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
    log.info("loaded %d images", N)

    param_groups = build_param_groups(module)
    for g in param_groups:
        log.info("param group `%s`: %d params total", g.name, g.numel())

    rows_norms: list[dict] = []
    rows_angles: list[dict] = []
    raw_history: list[dict] = []

    batches = []
    for start in range(0, N, args.batch_size):
        stop = min(start + args.batch_size, N)
        batches.append((start, stop))
    log.info("n_batches available=%d, will use %d steps", len(batches), args.n_steps)

    for step in range(args.n_steps):
        start, stop = batches[step % len(batches)]
        q = loc.query_images[start:stop].to(device, non_blocking=True)
        r = loc.ref_images[start:stop].to(device, non_blocking=True)
        labels = loc.query_labels[start:stop].to(device, non_blocking=True)
        rn, ra, raw = measure_step(module, q, r, labels, step, param_groups)
        rows_norms.extend(rn)
        rows_angles.extend(ra)
        raw_history.append({"step": step, **raw})
        log.info(
            "step %d raw: cls=%.3e eq=%.3e ac=%.3e marg=%.3e mask=%.3e "
            "con=%.3e dist=%.3e",
            step, raw["L_cls"], raw["L_eq"], raw["L_ac"], raw["L_marg_H"],
            raw["L_mask"], raw["L_con"], raw["L_dist"],
        )

    # Long-format CSVs.
    norms_csv = out_dir / "grad_norms.csv"
    with norms_csv.open("w") as f:
        f.write("step,loss_name,param_group,l2,rms,cos_vs_cls,n_params,raw_loss\n")
        for row in rows_norms:
            f.write(
                f"{row['batch_idx']},{row['loss_name']},{row['param_group']},"
                f"{row['l2']:.6e},{row['rms']:.6e},{row['cos_vs_cls']:.6f},"
                f"{row['n_params']},{row['raw_loss_value']:.6e}\n"
            )
    log.info("wrote %s", norms_csv)

    angles_csv = out_dir / "pairwise_angles.csv"
    with angles_csv.open("w") as f:
        f.write("step,loss_a,loss_b,param_group,cos_theta\n")
        for row in rows_angles:
            f.write(
                f"{row['batch_idx']},{row['loss_a']},{row['loss_b']},"
                f"{row['param_group']},{row['cos_theta']:.6f}\n"
            )
    log.info("wrote %s", angles_csv)

    # Recommended lambdas per param group.
    recommended: dict[str, dict[str, float]] = {}
    for gn in ("fusion", "classifier", "proj_head", "all_trainable"):
        recommended[gn] = recommend_lambdas(rows_norms, target_group=gn)
    (out_dir / "recommended_lambdas.json").write_text(
        json.dumps(recommended, indent=2)
    )

    write_summary(
        out_dir, rows_norms, rows_angles, Path(args.ckpt),
        args.n_steps, args.batch_size, cfg, recommended,
    )
    log.info("wrote %s", out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
