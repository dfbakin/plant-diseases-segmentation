"""RQ5 — Teacher-ceiling measurement under strict WSSS.

For 100-200 val images, compute IoU-vs-GT for a catalog of
WSSS-compliant teacher signals and their combinations:

Single-source teachers (per image, binary mask at top-alpha fraction,
IoU vs GT binary mask):
  * p3_chvar (pre-fusion channel variance of eq_only FPN features)
  * cam_classifier from eq_only
  * cam_classifier from D2
  * fused_chvar (post-fusion channel variance)
  * fused_l2norm (post-fusion channel L2-norm)

Combined teachers:
  * chvar ∩ cam (D2 recipe)
  * chvar ∪ cam
  * max-rank(chvar, cam) (D3 anchor recipe)
  * soft blend alpha*chvar + (1-alpha)*cam at best alpha

EMA synthetic teacher:
  * alpha*cam_eq + (1-alpha)*cam_D2 over alpha in {0.2, 0.5, 0.8}

Every teacher is evaluated at multiple alpha fractions (top-alpha
selection) AND at a full threshold sweep for comparability with
`val/cam_iou_auc`. Output gives max IoU per teacher.

WeakCLIP masks are optional (--weakclip-dir); if absent we note it and
skip.

Outputs:
  outputs/diagnostics/rq5_teachers/teacher_ious.csv
  outputs/diagnostics/rq5_teachers/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.conf.spdnet import SPDNetSpatialLossesConfig  # noqa: E402
from src.wsss.spdnet.lightning import SPDNetModule  # noqa: E402
from src.wsss.spdnet.online_loc_metric import (  # noqa: E402
    OnlineCAMIoU,
    compute_iou_sweep,
    DEFAULT_THRESHOLDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rq5")


CKPTS = {
    "eq_only": (
        "outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/"
        "epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt"
    ),
    "D2": (
        "outputs/spdnet_aux_losses/spdnet_spatial_d2_mask_warmstart_20260427/"
        "checkpoints/last.ckpt"
    ),
    "D3": (
        "outputs/spdnet_aux_losses/spdnet_spatial_d3_d2plus_union_warmstart_20260427/"
        "checkpoints/last.ckpt"
    ),
}


def _load(ckpt_rel: str, image_size: int, num_classes: int) -> SPDNetModule:
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
    path = (repo_root / ckpt_rel).resolve()
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    module.load_state_dict(sd.get("state_dict", sd), strict=False)
    return module


@torch.no_grad()
def _compute_features(
    module: SPDNetModule,
    loc: OnlineCAMIoU,
    device: torch.device,
    image_size: int,
) -> dict[str, torch.Tensor]:
    """Forward the 100-image subset, return per-image teacher maps.

    Every returned map is ``(N, H, W)`` at image resolution, normalised
    per-image to ``[0, 1]``.  Maps returned:

      - ``cam``: classifier CAM on active class
      - ``chvar_p3``: Var_c(p3_query)
      - ``chvar_fused``: Var_c(fused)
      - ``l2_fused``: ||fused[c, h, w]||_2 along channels
    """
    module.eval().to(device)
    model = module.model
    N = loc.query_images.shape[0]
    H_img = W_img = image_size

    out = {
        "cam": torch.zeros(N, H_img, W_img, dtype=torch.float32),
        "chvar_p3": torch.zeros(N, H_img, W_img, dtype=torch.float32),
        "chvar_fused": torch.zeros(N, H_img, W_img, dtype=torch.float32),
        "l2_fused": torch.zeros(N, H_img, W_img, dtype=torch.float32),
    }

    for start in range(0, N, loc.eval_batch_size):
        stop = min(start + loc.eval_batch_size, N)
        q = loc.query_images[start:stop].to(device, non_blocking=True)
        r = loc.ref_images[start:stop].to(device, non_blocking=True)
        labels_b = loc.query_labels[start:stop].to(device, non_blocking=True)

        feats = model.extract_merged_features(q, [r])
        p3 = feats["query_merged"]                                  # (B, C, Hf, Wf)
        fused = feats["fused"]
        cls_w = model.classifier.weight

        S = torch.einsum("nc,bchw->bnhw", cls_w, fused)
        label_mask = labels_b.bool().unsqueeze(-1).unsqueeze(-1)
        S_masked = S.masked_fill(~label_mask, float("-inf"))
        cam = S_masked.max(dim=1).values

        chvar_p3 = p3.var(dim=1, unbiased=False)
        chvar_fused = fused.var(dim=1, unbiased=False)
        l2_fused = fused.norm(dim=1)

        for name, tensor in [
            ("cam", cam), ("chvar_p3", chvar_p3),
            ("chvar_fused", chvar_fused), ("l2_fused", l2_fused),
        ]:
            t = F.interpolate(
                tensor.unsqueeze(1), size=(H_img, W_img),
                mode="bilinear", align_corners=False,
            ).squeeze(1).float()
            mn = t.amin(dim=(1, 2), keepdim=True)
            mx = t.amax(dim=(1, 2), keepdim=True)
            t_n = (t - mn) / (mx - mn + 1e-8)
            out[name][start:stop] = t_n.detach().cpu()
    return out


def topk_mask_from_map(
    norm_map: torch.Tensor, alpha: float,
) -> torch.Tensor:
    """Top-alpha binary mask per image.

    ``norm_map``: ``(N, H, W)`` in [0, 1]. Returns ``(N, H, W)`` in {0, 1}
    float32, where each image's top-``alpha`` fraction of positions by
    score are marked 1.
    """
    N, H, W = norm_map.shape
    P = H * W
    k = max(int(P * alpha), 1)
    flat = norm_map.reshape(N, P)
    thr, _ = torch.topk(flat, k=k, dim=1, largest=True)
    thr = thr[:, -1:]
    mask = (flat >= thr).float().view(N, H, W)
    return mask


def iou_from_binary(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor,
) -> float:
    """Mean per-image IoU over a batch. ``{0,1}`` inputs."""
    N = pred_mask.shape[0]
    pred_f = pred_mask.reshape(N, -1)
    gt_f = (gt_mask > 0).float().reshape(N, -1)
    inter = (pred_f * gt_f).sum(dim=1)
    union = ((pred_f + gt_f) > 0).float().sum(dim=1)
    iou = torch.where(
        union > 0,
        inter / union.clamp_min(1.0),
        torch.ones_like(union),
    )
    return float(iou.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantseg-root", default="data/plantsegv3")
    parser.add_argument(
        "--gt-binary-dir",
        default="outputs/plantseg_binary_mc115/gt_binary_val",
    )
    parser.add_argument("--out-dir", default="outputs/diagnostics/rq5_teachers")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=115)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--weakclip-dir", default="", type=str,
        help="directory with per-image WeakCLIP masks (filename = stem.png). "
             "Empty = skip. None available in the current workspace.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    gt_masks = loc.query_masks                                       # (N, H, W)
    gt_bin = (gt_masks > 0).float()
    log.info("loaded %d images; GT area frac mean=%.3f",
             N, float(gt_bin.mean().item()))

    # Compute teacher maps per checkpoint.
    teacher_maps: dict[str, dict[str, torch.Tensor]] = {}
    for label, rel in CKPTS.items():
        log.info("Forward %s", label)
        module = _load(rel, args.image_size, args.num_classes)
        teacher_maps[label] = _compute_features(module, loc, device, args.image_size)
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    alphas = [0.15, 0.20, 0.25, 0.30]
    rows: list[dict] = []

    # --- Single-source top-alpha sweeps. ---
    single_sources: list[tuple[str, torch.Tensor]] = [
        ("chvar_p3 (eq_only)", teacher_maps["eq_only"]["chvar_p3"]),
        ("chvar_fused (eq_only)", teacher_maps["eq_only"]["chvar_fused"]),
        ("l2_fused (eq_only)", teacher_maps["eq_only"]["l2_fused"]),
        ("cam (eq_only)", teacher_maps["eq_only"]["cam"]),
        ("cam (D2)", teacher_maps["D2"]["cam"]),
        ("cam (D3)", teacher_maps["D3"]["cam"]),
        ("chvar_p3 (D2)", teacher_maps["D2"]["chvar_p3"]),
        ("chvar_p3 (D3)", teacher_maps["D3"]["chvar_p3"]),
    ]
    for name, m in single_sources:
        for alpha in alphas:
            pred = topk_mask_from_map(m, alpha)
            iou = iou_from_binary(pred, gt_bin)
            rows.append({
                "teacher": name, "kind": "top_alpha",
                "alpha": alpha, "iou": iou,
            })
        # Full threshold sweep.
        ious_per_thr = compute_iou_sweep(m, gt_bin, DEFAULT_THRESHOLDS)
        auc = float(torch.trapz(ious_per_thr, DEFAULT_THRESHOLDS).item())
        best = float(ious_per_thr.max().item())
        best_thr = float(DEFAULT_THRESHOLDS[int(ious_per_thr.argmax().item())].item())
        rows.append({
            "teacher": name, "kind": "threshold_sweep_best",
            "alpha": None, "iou": best,
            "best_thr": best_thr, "auc": auc,
        })

    # --- Combined teachers at each alpha. ---
    cam_eq = teacher_maps["eq_only"]["cam"]
    cam_d2 = teacher_maps["D2"]["cam"]
    chvar_eq = teacher_maps["eq_only"]["chvar_p3"]
    for alpha in alphas:
        m_ch = topk_mask_from_map(chvar_eq, alpha)
        m_cam_eq = topk_mask_from_map(cam_eq, alpha)
        m_cam_d2 = topk_mask_from_map(cam_d2, alpha)

        # Intersection (D2 default).
        inter_eq = m_ch * m_cam_eq
        inter_d2 = m_ch * m_cam_d2
        rows.append({"teacher": "chvar ∩ cam_eq", "kind": "top_alpha_intersection",
                     "alpha": alpha, "iou": iou_from_binary(inter_eq, gt_bin)})
        rows.append({"teacher": "chvar ∩ cam_D2", "kind": "top_alpha_intersection",
                     "alpha": alpha, "iou": iou_from_binary(inter_d2, gt_bin)})
        # Union.
        uni_eq = ((m_ch + m_cam_eq) > 0).float()
        uni_d2 = ((m_ch + m_cam_d2) > 0).float()
        rows.append({"teacher": "chvar ∪ cam_eq", "kind": "top_alpha_union",
                     "alpha": alpha, "iou": iou_from_binary(uni_eq, gt_bin)})
        rows.append({"teacher": "chvar ∪ cam_D2", "kind": "top_alpha_union",
                     "alpha": alpha, "iou": iou_from_binary(uni_d2, gt_bin)})

        # Union-of-ranks (D3 recipe): per position, take max rank across the two.
        Nq, Hi, Wi = chvar_eq.shape
        P = Hi * Wi
        rank_ch = chvar_eq.reshape(Nq, P).argsort(dim=1).argsort(dim=1).float()
        rank_cam_eq = cam_eq.reshape(Nq, P).argsort(dim=1).argsort(dim=1).float()
        rank_cam_d2 = cam_d2.reshape(Nq, P).argsort(dim=1).argsort(dim=1).float()
        combined_eq = torch.maximum(rank_ch, rank_cam_eq).view(Nq, Hi, Wi)
        combined_d2 = torch.maximum(rank_ch, rank_cam_d2).view(Nq, Hi, Wi)
        # Normalize so topk_mask_from_map works uniformly.
        combined_eq = combined_eq / combined_eq.reshape(Nq, -1).amax(dim=1)[:, None, None]
        combined_d2 = combined_d2 / combined_d2.reshape(Nq, -1).amax(dim=1)[:, None, None]
        rows.append({"teacher": "max_rank(chvar, cam_eq)", "kind": "union_of_ranks",
                     "alpha": alpha, "iou": iou_from_binary(
                         topk_mask_from_map(combined_eq, alpha), gt_bin)})
        rows.append({"teacher": "max_rank(chvar, cam_D2)", "kind": "union_of_ranks",
                     "alpha": alpha, "iou": iou_from_binary(
                         topk_mask_from_map(combined_d2, alpha), gt_bin)})

    # --- Optimal soft blend of chvar + cam (threshold-sweep AUC). ---
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = ratio * chvar_eq + (1 - ratio) * cam_eq
        # Re-normalise per image.
        mn = blended.reshape(N, -1).min(dim=1).values[:, None, None]
        mx = blended.reshape(N, -1).max(dim=1).values[:, None, None]
        blended = (blended - mn) / (mx - mn + 1e-8)
        ious_per_thr = compute_iou_sweep(blended, gt_bin, DEFAULT_THRESHOLDS)
        auc = float(torch.trapz(ious_per_thr, DEFAULT_THRESHOLDS).item())
        rows.append({
            "teacher": f"{ratio:.2f}*chvar + {1-ratio:.2f}*cam_eq",
            "kind": "soft_blend_auc",
            "alpha": None,
            "iou": float(ious_per_thr.max().item()),
            "auc": auc,
        })

    # --- EMA synthetic teacher: alpha*cam_eq + (1-alpha)*cam_D2 ---
    for a in [0.2, 0.5, 0.8]:
        ema = a * cam_eq + (1 - a) * cam_d2
        mn = ema.reshape(N, -1).min(dim=1).values[:, None, None]
        mx = ema.reshape(N, -1).max(dim=1).values[:, None, None]
        ema = (ema - mn) / (mx - mn + 1e-8)
        ious = compute_iou_sweep(ema, gt_bin, DEFAULT_THRESHOLDS)
        auc = float(torch.trapz(ious, DEFAULT_THRESHOLDS).item())
        rows.append({
            "teacher": f"EMA synthetic (alpha={a:.1f}*eq_only + {1-a:.1f}*D2)",
            "kind": "ema_synthetic",
            "alpha": None,
            "iou": float(ious.max().item()),
            "auc": auc,
        })

    # --- Non-WSSS reference: GT itself at every threshold = 1.0 ---
    rows.append({
        "teacher": "GT (non-WSSS reference)",
        "kind": "reference",
        "alpha": None,
        "iou": 1.0,
        "auc": 1.0,
    })

    # Write CSV.
    csv_path = out_dir / "teacher_ious.csv"
    cols = ["teacher", "kind", "alpha", "iou", "best_thr", "auc"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(
                str(row.get(c, "")) if not isinstance(row.get(c), float)
                else f"{row[c]:.6f}"
                for c in cols
            ) + "\n")
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    # Summary markdown with ranked tables.
    summary = out_dir / "SUMMARY.md"
    lines = []
    lines.append("# RQ5 — Teacher-ceiling measurement under strict WSSS")
    lines.append("")
    lines.append(f"Images: {N}  |  image_size: {args.image_size}  |  GT area frac mean: {float(gt_bin.mean().item()):.3f}")
    lines.append("")
    lines.append("## Single-source teachers: best top-alpha IoU")
    lines.append("")
    lines.append("| teacher | best alpha | best IoU (top-alpha) | AUC (threshold sweep) | best_thr |")
    lines.append("|---|---|---|---|---|")
    for name, _ in single_sources:
        alpha_rows = [r for r in rows if r["teacher"] == name and r["kind"] == "top_alpha"]
        sweep_row = [r for r in rows if r["teacher"] == name and r["kind"] == "threshold_sweep_best"]
        if not alpha_rows or not sweep_row:
            continue
        best = max(alpha_rows, key=lambda r: r["iou"])
        sw = sweep_row[0]
        lines.append(
            f"| {name} | {best['alpha']:.2f} | **{best['iou']:.4f}** | "
            f"{sw['auc']:.4f} | {sw['best_thr']:.2f} |"
        )
    lines.append("")

    # Combined teachers: pick best alpha per combination kind.
    lines.append("## Combined teachers (at top-alpha)")
    lines.append("")
    lines.append("| combination | best alpha | best IoU |")
    lines.append("|---|---|---|")
    for key in ["chvar ∩ cam_eq", "chvar ∩ cam_D2", "chvar ∪ cam_eq",
                "chvar ∪ cam_D2", "max_rank(chvar, cam_eq)",
                "max_rank(chvar, cam_D2)"]:
        rs = [r for r in rows if r["teacher"] == key]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["iou"])
        lines.append(f"| {key} | {best['alpha']:.2f} | **{best['iou']:.4f}** |")
    lines.append("")

    lines.append("## Soft-blend of chvar + cam (threshold-sweep AUC)")
    lines.append("")
    lines.append("| blend | best IoU | AUC |")
    lines.append("|---|---|---|")
    for r in rows:
        if r["kind"] == "soft_blend_auc":
            lines.append(f"| {r['teacher']} | {r['iou']:.4f} | **{r['auc']:.4f}** |")
    lines.append("")

    lines.append("## EMA synthetic teacher")
    lines.append("")
    lines.append("| blend | best IoU | AUC |")
    lines.append("|---|---|---|")
    for r in rows:
        if r["kind"] == "ema_synthetic":
            lines.append(f"| {r['teacher']} | {r['iou']:.4f} | {r['auc']:.4f} |")
    lines.append("")

    summary.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", summary)


if __name__ == "__main__":
    main()
