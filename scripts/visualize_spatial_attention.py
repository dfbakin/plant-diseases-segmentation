"""Visualize SpatialCrossAttention behaviour in trained SPDNet (spatial fusion).

For each query/reference pair, generates a 6-panel figure:

  1. Query + GT contour (markers show sampled disease/healthy query pixels)
  2. Query: ||attended[q]|| heatmap (where does cross-attention update the query?)
  3. Reference + GT contour
  4. Reference: aggregate attention received (sum over query positions)
  5. Per-query attention map for the disease query pixel  (-> overlaid on ref)
  6. Per-query attention map for the healthy query pixel  (-> overlaid on ref)

Diagnostic questions:
  Q1 (query side):  Does the attended-magnitude map peak on disease in the query?
  Q2 (ref side):    Does aggregate attention land on disease in the reference?
  Q3 (pairwise):    Does a disease query pixel attend to disease reference pixels?

We also print summary statistics (energy concentration on GT disease) so the
visual inspection has a quantitative companion.

Usage:
    export PATH="/venv/main/bin:$PATH"
    python scripts/visualize_spatial_attention.py \
        --checkpoint outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/\
epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt \
        --output_dir outputs/visualizations/spatial_attention_n1_ps_pv \
        --num_images 25
"""
from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint
from src.wsss.spdnet.model import SPDNet, SpatialCrossAttention

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

GT_CONTOUR_RGB = (255, 50, 220)
DISEASE_MARKER_RGB = (255, 200, 0)
HEALTHY_MARKER_RGB = (0, 200, 255)
GT_FILL = np.array([0.85, 0.15, 0.85])


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def normalize_map(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def to_img_size(arr_2d: np.ndarray, w: int, h: int) -> np.ndarray:
    pil = Image.fromarray(arr_2d.astype(np.float32), mode="F")
    return np.array(pil.resize((w, h), Image.BILINEAR))


def overlay_heatmap(img_np, heatmap, alpha=0.55):
    hm = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]
    bl = (img_np.astype(np.float32) / 255 * (1 - alpha)
          + hm_c.astype(np.float32) / 255 * alpha)
    return (np.clip(bl, 0, 1) * 255).astype(np.uint8)


def overlay_mask(img_np, mask, color=GT_FILL, alpha=0.40):
    img_f = img_np.astype(np.float32) / 255
    out = img_f.copy()
    fg = mask > 0
    out[fg] = out[fg] * (1 - alpha) + color * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def overlay_contour(img_np, mask, color=GT_CONTOUR_RGB, thickness=2):
    if mask is None:
        return img_np
    out = img_np.copy()
    m8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_marker(img_np, xy, color, radius=10):
    out = img_np.copy()
    cv2.circle(out, xy, radius, color, thickness=2)
    cv2.circle(out, xy, 2, color, thickness=-1)
    return out


def build_val_ref_pool(
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
    image_ext: str,
) -> dict[int, list[str]]:
    """Class -> [val image names containing that class]."""
    pool: dict[int, list[str]] = defaultdict(list)
    for name, label in label_dict.items():
        if not (image_dir / f"{name}{image_ext}").exists():
            continue
        for cls in np.where(label > 0)[0]:
            pool[int(cls)].append(name)
    return pool


def load_class_names(class_names_path: Path) -> list[str]:
    return [l.strip() for l in open(class_names_path)]


def parse_class_from_filename(name: str, class_names: list[str]) -> int | None:
    """Recover class index from val image filename (e.g. apple_scab_google_0190 -> 'apple scab')."""
    candidates = []
    for idx, cn in enumerate(class_names):
        cn_us = cn.replace(" ", "_")
        if name.startswith(cn_us + "_") or name == cn_us:
            candidates.append((cn_us, idx))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -len(x[0]))
    return candidates[0][1]


def build_train_ref_pool(
    train_label_file: Path,
    ref_image_dir: Path,
    image_ext: str,
) -> dict[int, list[str]]:
    """Build class -> [train image names] pool from train label file.

    Filters to images that actually exist in *ref_image_dir* so we can load them.
    """
    train_labels = np.load(train_label_file, allow_pickle=True).item()
    pool: dict[int, list[str]] = defaultdict(list)
    for name, label in train_labels.items():
        if not (ref_image_dir / f"{name}{image_ext}").exists():
            continue
        for cls in np.where(label > 0)[0]:
            pool[int(cls)].append(name)
    return pool


@torch.no_grad()
def extract_spatial_attention(
    model: SPDNet,
    query_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run SPDNet (spatial fusion) and return attention weights + feature maps."""
    if model.fusion_mode != "spatial":
        raise ValueError(
            f"Model fusion_mode={model.fusion_mode!r} -- need 'spatial'"
        )

    q = query_tensor.unsqueeze(0).to(device)
    r = ref_tensor.unsqueeze(0).to(device)

    q_fpn = model._get_fpn_features(q)
    r_fpn = model._get_fpn_features(r)

    query_merged = model._merge_fpn(q_fpn)   # (1, C, H, W)
    ref_merged = model._merge_fpn(r_fpn)     # (1, C, Hr_full, Wr_full)

    sa: SpatialCrossAttention = model.spatial_attn
    B, C, H, W = query_merged.shape
    ref_pooled = sa.pool(ref_merged)         # (1, C, h, w)
    Hr, Wr = ref_pooled.shape[2:]

    q_flat = query_merged.flatten(2).permute(0, 2, 1)
    kv_flat = ref_pooled.flatten(2).permute(0, 2, 1)
    q_norm = sa.norm_q(q_flat)
    kv_norm = sa.norm_kv(kv_flat)

    attended, attn_weights = sa.cross_attn(
        q_norm, kv_norm, kv_norm,
        need_weights=True, average_attn_weights=True,
    )
    attended_2d = attended.permute(0, 2, 1).view(B, C, H, W)
    fused = query_merged + sa.gate * attended_2d

    attn_w = attn_weights[0].cpu().numpy()                       # (HW, Hr*Wr)
    attended_norm = attended_2d[0].norm(dim=0).cpu().numpy()     # (H, W)
    ref_received = attn_w.sum(axis=0).reshape(Hr, Wr)            # (Hr, Wr)

    return {
        "feat_size": (H, W),
        "ref_pool_size": (Hr, Wr),
        "attn_weights": attn_w,
        "attended_norm_query": attended_norm,
        "ref_received": ref_received,
        "gate": float(sa.gate.item()),
        "query_merged": query_merged[0].cpu().numpy(),
        "fused": fused[0].cpu().numpy(),
    }


def gt_to_feat_grid(gt_mask: np.ndarray, feat_size: tuple[int, int]) -> np.ndarray:
    """Downsample GT to feature-map resolution by max-pooling (preserves disease)."""
    H, W = feat_size
    img_h, img_w = gt_mask.shape
    pil = Image.fromarray((gt_mask > 0).astype(np.uint8) * 255, mode="L")
    down = np.array(pil.resize((W, H), Image.BILINEAR))
    return (down > 64).astype(np.uint8)


def pick_pixel(gt_feat: np.ndarray, target: str, rng: random.Random
               ) -> tuple[int, int] | None:
    """Pick a feature-grid pixel on disease (target='disease') or healthy regions.

    Prefers pixels in the largest connected component of the target region.
    Returns (h, w) at feature-grid resolution, or None if not available.
    """
    H, W = gt_feat.shape
    if target == "disease":
        coords = np.argwhere(gt_feat > 0)
    else:
        coords = np.argwhere(gt_feat == 0)

    if len(coords) == 0:
        return None

    if target == "disease":
        h = int(np.median(coords[:, 0]))
        w = int(np.median(coords[:, 1]))
        if gt_feat[h, w] > 0:
            return (h, w)
        return tuple(coords[len(coords) // 2])
    else:
        if len(coords) > 5:
            idx = rng.randrange(len(coords))
            return tuple(coords[idx])
        return tuple(coords[0])


def feat_pos_to_image_xy(pos: tuple[int, int],
                         feat_size: tuple[int, int],
                         img_size: tuple[int, int]
                         ) -> tuple[int, int]:
    """Map (h, w) at feature-grid -> (x, y) at image resolution (cell center)."""
    Hf, Wf = feat_size
    img_w, img_h = img_size
    cy = int((pos[0] + 0.5) * (img_h / Hf))
    cx = int((pos[1] + 0.5) * (img_w / Wf))
    return (cx, cy)


def energy_concentration(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of heatmap energy lying inside `mask`.

    Compare to the random baseline = mask area / total area.
    """
    h = np.asarray(heatmap, dtype=np.float64)
    h = h - h.min()
    total = h.sum()
    if total <= 0 or mask is None:
        return float("nan")
    return float(h[mask > 0].sum() / total)


def visualize_one(
    name: str,
    ref_name: str,
    img_q: np.ndarray, gt_q: np.ndarray | None,
    img_r: np.ndarray, gt_r: np.ndarray | None,
    attn_data: dict,
    output_path: Path,
    rng: random.Random,
) -> dict:
    img_h, img_w = img_q.shape[:2]
    rh, rw = img_r.shape[:2]
    feat_size = attn_data["feat_size"]
    ref_pool_size = attn_data["ref_pool_size"]

    attended_norm = normalize_map(attn_data["attended_norm_query"])
    attended_full = normalize_map(to_img_size(attn_data["attended_norm_query"],
                                              img_w, img_h))

    ref_received = normalize_map(attn_data["ref_received"])
    ref_received_full = normalize_map(to_img_size(attn_data["ref_received"],
                                                  rw, rh))

    gt_q_feat = (gt_to_feat_grid(gt_q, feat_size) if gt_q is not None
                 else np.zeros(feat_size, dtype=np.uint8))
    gt_r_feat = (gt_to_feat_grid(gt_r, ref_pool_size) if gt_r is not None
                 else np.zeros(ref_pool_size, dtype=np.uint8))

    pos_d = pick_pixel(gt_q_feat, "disease", rng) if gt_q is not None else None
    pos_h = pick_pixel(gt_q_feat, "healthy", rng) if gt_q is not None else None

    Hf, Wf = feat_size
    Hrp, Wrp = ref_pool_size

    def attn_for(pos):
        if pos is None:
            return np.zeros((rh, rw), dtype=np.float32)
        flat_idx = pos[0] * Wf + pos[1]
        attn_2d = attn_data["attn_weights"][flat_idx].reshape(Hrp, Wrp)
        return normalize_map(to_img_size(attn_2d, rw, rh))

    attn_disease = attn_for(pos_d)
    attn_healthy = attn_for(pos_h)

    p1 = overlay_contour(img_q, gt_q)
    if pos_d is not None:
        p1 = draw_marker(p1, feat_pos_to_image_xy(pos_d, feat_size, (img_w, img_h)),
                         DISEASE_MARKER_RGB)
    if pos_h is not None:
        p1 = draw_marker(p1, feat_pos_to_image_xy(pos_h, feat_size, (img_w, img_h)),
                         HEALTHY_MARKER_RGB)

    p2 = overlay_contour(overlay_heatmap(img_q, attended_full), gt_q)
    p3 = overlay_contour(img_r, gt_r)
    p4 = overlay_contour(overlay_heatmap(img_r, ref_received_full), gt_r)
    p5 = overlay_contour(overlay_heatmap(img_r, attn_disease), gt_r)
    p6 = overlay_contour(overlay_heatmap(img_r, attn_healthy), gt_r)

    metrics = {}
    if gt_q is not None:
        gt_q_full = (gt_q > 0).astype(np.uint8)
        rand_q = float(gt_q_full.sum() / gt_q_full.size) if gt_q_full.size else float("nan")
        metrics["q_attended_on_disease"] = energy_concentration(attended_full, gt_q_full)
        metrics["q_random_baseline"] = rand_q

    if gt_r is not None:
        gt_r_full = (gt_r > 0).astype(np.uint8)
        rand_r = float(gt_r_full.sum() / gt_r_full.size) if gt_r_full.size else float("nan")
        metrics["r_received_on_disease"] = energy_concentration(ref_received_full, gt_r_full)
        metrics["r_random_baseline"] = rand_r
        if pos_d is not None:
            metrics["r_attn_from_dis_q_on_disease"] = energy_concentration(attn_disease, gt_r_full)
        if pos_h is not None:
            metrics["r_attn_from_hlt_q_on_disease"] = energy_concentration(attn_healthy, gt_r_full)

    fig, axes = plt.subplots(1, 6, figsize=(28, 5), dpi=140)
    titles = [
        "Query + GT\n(yellow=disease pixel, cyan=healthy)",
        f"Query: ||attended[q]||\n(gate={attn_data['gate']:.2f})",
        f"Reference + GT\n(ref: {ref_name})",
        "Reference: attention received\n(sum over query positions)",
        "Disease-Q -> ref attention",
        "Healthy-Q -> ref attention",
    ]
    panels = [p1, p2, p3, p4, p5, p6]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")

    sub = []
    if "q_attended_on_disease" in metrics:
        sub.append(f"q-attended on dis: {metrics['q_attended_on_disease']*100:.1f}% "
                   f"(baseline {metrics['q_random_baseline']*100:.1f}%)")
    if "r_received_on_disease" in metrics:
        sub.append(f"r-received on dis: {metrics['r_received_on_disease']*100:.1f}% "
                   f"(baseline {metrics['r_random_baseline']*100:.1f}%)")
    fig.suptitle(f"{name}    |    {' | '.join(sub) if sub else ''}",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3)
    fig.savefig(str(output_path), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"name": name, "ref": ref_name, "panels": panels, "metrics": metrics}


def make_summary_grid(results: list[dict], output_path: Path, max_rows: int = 12):
    n_rows = min(len(results), max_rows)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.5 * n_rows),
                             dpi=140)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    headers = [
        "Query+GT\n(yellow=dis, cyan=healthy)",
        "Query ||attended||",
        "Reference+GT",
        "Ref attention received",
        "Dis-Q -> ref attn",
        "Healthy-Q -> ref attn",
    ]
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col].imshow(results[row]["panels"][col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(headers[col], fontsize=10, fontweight="bold")
        m = results[row]["metrics"]
        lab = results[row]["name"]
        if "q_attended_on_disease" in m:
            lab += (f"\nq-att/dis {m['q_attended_on_disease']*100:.0f}% "
                    f"(base {m['q_random_baseline']*100:.0f}%)")
        if "r_received_on_disease" in m:
            lab += (f"\nr-rcv/dis {m['r_received_on_disease']*100:.0f}% "
                    f"(base {m['r_random_baseline']*100:.0f}%)")
        axes[row, 0].set_ylabel(lab, fontsize=7, rotation=0, labelpad=140, va="center")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Summary grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Spatial cross-attention visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", default="data/plantsegv3/images/val")
    parser.add_argument("--gt_dir", default="outputs/plantseg_binary_mc115/gt_binary_val")
    parser.add_argument("--label_file",
                        default="outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy")
    parser.add_argument("--ref_image_dir", default="data/plantsegv3/images/train",
                        help="Directory containing reference images "
                             "(default: PlantSeg train -- same as training)")
    parser.add_argument("--ref_gt_dir", default="",
                        help="Optional GT directory for reference images")
    parser.add_argument("--class_names_file",
                        default="outputs/plantseg_binary_mc115/labels/class_names.txt")
    parser.add_argument("--ref_strategy",
                        choices=["same_class_train", "random_val"],
                        default="same_class_train",
                        help="'same_class_train' (correct, matches training) "
                             "or 'random_val' (reproduces previous-eval bug)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=25)
    parser.add_argument("--num_classes", type=int, default=115)
    parser.add_argument("--max_size", type=int, default=784)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_ext", default=".jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading SPDNet from {args.checkpoint}")
    model = load_spdnet_from_checkpoint(args.checkpoint, args.num_classes).to(device).eval()
    log.info(f"  fusion_mode = {model.fusion_mode}")
    if model.fusion_mode != "spatial":
        raise SystemExit(f"Need spatial fusion model; got {model.fusion_mode!r}")
    log.info(f"  spatial gate value = {model.spatial_attn.gate.item():.4f}")

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    ref_image_dir = Path(args.ref_image_dir)
    ref_gt_dir = Path(args.ref_gt_dir) if args.ref_gt_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_names_with_gt = sorted(f.stem for f in gt_dir.glob("*.png"))
    class_names = load_class_names(Path(args.class_names_file))

    if args.ref_strategy == "same_class_train":
        log.info("Reference strategy: same_class_train "
                 "(parsing class from val filename, picking from train)")
        train_pool = build_train_ref_pool(
            Path(args.label_file), ref_image_dir, args.image_ext,
        )
        log.info(f"  Train ref pool covers {len(train_pool)}/{args.num_classes} classes")
    else:
        log.info("Reference strategy: random_val (reproduces evaluation bug)")
        full_labels = np.load(args.label_file, allow_pickle=True).item()
        label_dict: dict[str, np.ndarray] = {}
        for name in val_names_with_gt:
            if name in full_labels:
                label_dict[name] = full_labels[name]
            else:
                gt = np.array(Image.open(gt_dir / f"{name}.png"))
                lbl = np.zeros(args.num_classes, dtype=np.float32)
                if (gt > 0).any():
                    lbl[0] = 1.0
                label_dict[name] = lbl
        ref_pool_val = build_val_ref_pool(label_dict, image_dir, args.image_ext)

    candidates = []
    for name in val_names_with_gt:
        gt_path = gt_dir / f"{name}.png"
        gt = np.array(Image.open(gt_path))
        if (gt > 0).sum() < 100:
            continue
        if not (image_dir / f"{name}{args.image_ext}").exists():
            continue
        candidates.append(name)

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.num_images, len(candidates)))
    selected.sort()
    log.info(f"Selected {len(selected)} query images "
             f"(filtered from {len(candidates)} val images with disease)")

    tfm = get_val_transform()
    results = []
    for name in tqdm(selected, desc="Visualizing"):
        q_pil = Image.open(image_dir / f"{name}{args.image_ext}").convert("RGB")
        if max(q_pil.size) > args.max_size:
            r = args.max_size / max(q_pil.size)
            q_pil = q_pil.resize((round(q_pil.width * r), round(q_pil.height * r)),
                                 resample=Image.BICUBIC)
        img_q = np.array(q_pil)
        img_h, img_w = img_q.shape[:2]
        gt_q_full = np.array(Image.open(gt_dir / f"{name}.png"))
        if gt_q_full.shape[:2] != (img_h, img_w):
            gt_q_full = np.array(
                Image.fromarray(gt_q_full).resize((img_w, img_h), Image.NEAREST)
            )

        ref_rng = random.Random(args.seed + abs(hash(name)) % (2**31))

        if args.ref_strategy == "same_class_train":
            ref_cls = parse_class_from_filename(name, class_names)
            if ref_cls is None or ref_cls not in train_pool:
                log.warning(f"  Could not parse class for {name}, skipping")
                continue
            ref_choices = [n for n in train_pool[ref_cls] if n != name]
            ref_name = ref_rng.choice(ref_choices) if ref_choices else name
            ref_dir_for_img = ref_image_dir
            ref_class_name = class_names[ref_cls]
        else:
            label_v = label_dict.get(name)
            active = np.where(label_v > 0)[0].tolist() if label_v is not None else []
            ref_cls = active[0] if active else 0
            ref_choices = [n for n in ref_pool_val.get(ref_cls, []) if n != name]
            if not ref_choices:
                ref_choices = [n for n in candidates if n != name]
            ref_name = ref_rng.choice(ref_choices) if ref_choices else name
            ref_dir_for_img = image_dir
            ref_class_name = class_names[ref_cls] if ref_cls < len(class_names) else "?"

        r_pil = Image.open(ref_dir_for_img / f"{ref_name}{args.image_ext}").convert("RGB")
        if max(r_pil.size) > args.max_size:
            rr = args.max_size / max(r_pil.size)
            r_pil = r_pil.resize((round(r_pil.width * rr), round(r_pil.height * rr)),
                                 resample=Image.BICUBIC)
        img_r = np.array(r_pil)
        gt_r_full = None
        for gtd in [ref_gt_dir, gt_dir]:
            if gtd is None:
                continue
            gt_r_path = gtd / f"{ref_name}.png"
            if gt_r_path.exists():
                gt_r_full = np.array(Image.open(gt_r_path))
                rh, rw = img_r.shape[:2]
                if gt_r_full.shape[:2] != (rh, rw):
                    gt_r_full = np.array(
                        Image.fromarray(gt_r_full).resize((rw, rh), Image.NEAREST)
                    )
                break

        q_t = tfm(q_pil)
        r_t = tfm(r_pil)
        attn_data = extract_spatial_attention(model, q_t, r_t, device)

        result = visualize_one(
            name=name, ref_name=f"{ref_name} [{ref_class_name}]",
            img_q=img_q, gt_q=gt_q_full,
            img_r=img_r, gt_r=gt_r_full,
            attn_data=attn_data,
            output_path=output_dir / f"{name}.png",
            rng=rng,
        )
        results.append(result)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    make_summary_grid(results, output_dir / "summary_grid.png",
                      max_rows=min(len(results), 12))

    print("\n" + "=" * 90)
    print(f"  SPATIAL ATTENTION DIAGNOSTIC SUMMARY  ({len(results)} images)")
    print(f"  Model: {Path(args.checkpoint).parts[-3]}")
    print(f"  Spatial gate gamma: {model.spatial_attn.gate.item():.4f}")
    print(f"  Reference strategy: {args.ref_strategy}")
    print("=" * 90)

    def collect(key: str) -> list[float]:
        return [r["metrics"][key] for r in results
                if key in r["metrics"] and not np.isnan(r["metrics"][key])]

    def summarise(label: str, vals: list[float], baseline: list[float] | None = None):
        if not vals:
            print(f"  {label:<55}  n/a")
            return
        s = (f"mean={np.mean(vals)*100:5.2f}%  "
             f"median={np.median(vals)*100:5.2f}%  "
             f"std={np.std(vals)*100:5.2f}%  "
             f"n={len(vals)}")
        if baseline:
            s += f"  | random baseline mean={np.mean(baseline)*100:5.2f}%"
        print(f"  {label:<55}  {s}")

    summarise(
        "Query: || attended || energy on Q-disease",
        collect("q_attended_on_disease"),
        collect("q_random_baseline"),
    )
    summarise(
        "Reference: attention received energy on R-disease",
        collect("r_received_on_disease"),
        collect("r_random_baseline"),
    )
    summarise(
        "Pairwise: dis-query -> ref attention on R-disease",
        collect("r_attn_from_dis_q_on_disease"),
        collect("r_random_baseline"),
    )
    summarise(
        "Pairwise: hlt-query -> ref attention on R-disease",
        collect("r_attn_from_hlt_q_on_disease"),
        collect("r_random_baseline"),
    )
    print("=" * 90)
    print(f"  Outputs in {output_dir}")
    print("  How to read: 'energy on disease' is the share of heatmap mass falling")
    print("  on GT disease pixels. The 'random baseline' is the disease coverage of")
    print("  the image: a value at the baseline means uniform/random attention; a")
    print("  value substantially above baseline means attention concentrates on disease.")


if __name__ == "__main__":
    main()
