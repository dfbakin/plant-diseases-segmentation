"""Comprehensive ADPL-CAM & feature visualization for SPDNet.

Generates high-resolution grid comparisons of different activation types:
  1. Original image + GT overlay
  2. Binary-aggregated ADPL-CAM (from pre-generated .npy)
  3. Per-class ADPL-CAM (top-1 active class by energy)
  4. Per-class ADPL-CAM (top-2 active class by energy)
  5. Query features only (max across channels, before reference fusion)
  6. Fused features (max across channels, after reference token injection)
  7. Reference contribution (fused - query, shows what reference adds)
  8. GradCAM from classification head

Uses the same 25 validation images as MCTformer visualization for direct comparison.

Usage:
    export PATH="/venv/main/bin:$PATH"
    python scripts/visualize_spdnet_activations.py \
        --checkpoint outputs/spdnet_plantseg/2026-04-13_10-52-29/checkpoints/last.ckpt \
        --image_dir data/plantsegv3/images/val \
        --gt_dir outputs/plantseg_binary_mc115/gt_binary_val \
        --cam_dir outputs/spdnet_plantseg/cams/cam_npy_val \
        --label_file outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy \
        --output_dir outputs/visualizations/spdnet_val_cam_exploration \
        --num_images 25 \
        --seed 42
"""

import argparse
import gc
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantseg import DISEASE_CLASSES
from src.wsss.spdnet.model import SPDNet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

GT_COLOR = np.array([0.15, 0.85, 0.30])


def load_model(checkpoint: str, num_classes: int, device: torch.device) -> SPDNet:
    model = SPDNet(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def build_reference_pool(
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
) -> dict[int, list[str]]:
    pool: dict[int, list[str]] = defaultdict(list)
    for name, label in label_dict.items():
        if not (image_dir / f"{name}.jpg").exists():
            continue
        for cls in np.where(label > 0)[0]:
            pool[int(cls)].append(name)
    return pool


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    img = tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(img_np, heatmap, alpha=0.5, cmap=cv2.COLORMAP_JET):
    hm_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_color = cv2.applyColorMap(hm_uint8, cmap)[:, :, ::-1]
    blended = (
        img_np.astype(np.float32) / 255 * (1 - alpha)
        + hm_color.astype(np.float32) / 255 * alpha
    )
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def overlay_mask(img_np, mask, color=GT_COLOR, alpha=0.45):
    img_f = img_np.astype(np.float32) / 255
    result = img_f.copy()
    fg = mask > 0
    result[fg] = result[fg] * (1 - alpha) + color * alpha
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def normalize_map(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def to_img_size(arr_2d, img_w, img_h):
    pil = Image.fromarray(arr_2d.astype(np.float32), mode="F")
    return np.array(pil.resize((img_w, img_h), Image.BILINEAR))


@torch.no_grad()
def extract_spdnet_activations(
    model: SPDNet,
    query_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run SPDNet forward and extract intermediate activations.

    Returns dict with:
        per_class_cam: (num_classes, Hf, Wf)
        query_feat: (C, Hf, Wf) merged query FPN features before reference fusion
        fused_feat: (C, Hf, Wf) features after reference token injection
        logits: (num_classes,)
    """
    q = query_tensor.unsqueeze(0).to(device)
    r = ref_tensor.unsqueeze(0).to(device)

    q_feats = model.extract_features(q)
    r_feats = model.extract_features(r)

    q_fpn = model.fpn(q_feats)
    r_fpn = model.fpn(r_feats)

    q_fpn = [model.mse(p) for p in q_fpn]
    r_fpn = [model.mse(p) for p in r_fpn]

    target_size = q_fpn[0].shape[2:]
    query_merged = torch.zeros_like(q_fpn[0])
    for level in q_fpn:
        query_merged = query_merged + F.interpolate(
            level, size=target_size, mode="bilinear", align_corners=False
        )
    query_merged = query_merged / len(q_fpn)

    ref_tokens = model.adpl_cam.tokenize(r_fpn)
    fused = model.adpl_cam.fuse(query_merged, ref_tokens)

    pooled = fused.mean(dim=[2, 3])
    logits = model.classifier(pooled)[0]

    w = model.classifier.weight  # (num_classes, C)
    cam = F.relu(torch.einsum("nc,bchw->bnhw", w, fused))

    return {
        "per_class_cam": cam[0].cpu().numpy(),
        "query_feat": query_merged[0].cpu().numpy(),
        "fused_feat": fused[0].cpu().numpy(),
        "logits": logits.cpu().numpy(),
    }


def compute_gradcam(
    model: SPDNet,
    query_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
) -> tuple[np.ndarray, int]:
    """GradCAM via gradients of the classifier w.r.t. fused features."""
    model.eval()
    q = query_tensor.unsqueeze(0).to(device).requires_grad_(False)
    r = ref_tensor.unsqueeze(0).to(device).requires_grad_(False)

    activations = {}
    gradients = {}

    def fwd_hook(_mod, _inp, out):
        activations["fused"] = out

    def bwd_hook(_mod, _grad_in, grad_out):
        gradients["fused"] = grad_out[0]

    q_feats = model.extract_features(q)
    r_feats = model.extract_features(r)

    q_fpn = model.fpn(q_feats)
    r_fpn = model.fpn(r_feats)
    q_fpn = [model.mse(p) for p in q_fpn]
    r_fpn = [model.mse(p) for p in r_fpn]

    target_size = q_fpn[0].shape[2:]
    query_merged = torch.zeros_like(q_fpn[0])
    for level in q_fpn:
        query_merged = query_merged + F.interpolate(
            level, size=target_size, mode="bilinear", align_corners=False
        )
    query_merged = (query_merged / len(q_fpn)).detach().requires_grad_(True)

    ref_tokens = model.adpl_cam.tokenize(r_fpn)
    fused = model.adpl_cam.fuse(query_merged, ref_tokens)

    w = model.classifier.weight
    cam_all = torch.einsum("nc,bchw->bnhw", w, fused)
    cls_scores = cam_all.mean(dim=[2, 3])[0]

    if target_class is None:
        target_class = cls_scores.argmax().item()

    model.zero_grad()
    cls_scores[target_class].backward(retain_graph=False)

    if query_merged.grad is not None:
        grads = query_merged.grad[0]
        feats = fused[0].detach()
        weights = grads.mean(dim=[1, 2])
        gradcam = F.relu((weights[:, None, None] * feats).sum(0))
        return gradcam.cpu().numpy(), target_class

    return np.zeros(target_size, dtype=np.float32), target_class


def make_image_grid(
    img_np, gt_mask, binary_cam_full, activations, gradcam_map, gradcam_cls,
    name, output_path, class_names, ref_name,
):
    per_class_cam = activations["per_class_cam"]
    query_feat = activations["query_feat"]
    fused_feat = activations["fused_feat"]
    logits = activations["logits"]
    img_h, img_w = img_np.shape[:2]

    def to_img(a):
        return to_img_size(a, img_w, img_h)

    energy = per_class_cam.sum(axis=(1, 2))
    top_k = np.argsort(energy)[::-1]
    top1_cls, top2_cls = top_k[0], top_k[1]

    top1_name = class_names[top1_cls] if top1_cls < len(class_names) else f"cls_{top1_cls}"
    top2_name = class_names[top2_cls] if top2_cls < len(class_names) else f"cls_{top2_cls}"

    cam_top1 = normalize_map(to_img(per_class_cam[top1_cls]))
    cam_top2 = normalize_map(to_img(per_class_cam[top2_cls]))

    query_max = normalize_map(to_img(query_feat.max(axis=0)))
    fused_max = normalize_map(to_img(fused_feat.max(axis=0)))

    delta = fused_feat - query_feat
    delta_pos = np.maximum(delta, 0).max(axis=0)
    delta_map = normalize_map(to_img(delta_pos))

    gradcam_full = normalize_map(to_img(gradcam_map))

    panels = [
        ("Original + GT",
         overlay_mask(img_np, gt_mask) if gt_mask is not None else img_np),
        ("Binary-agg CAM",
         overlay_heatmap(img_np, binary_cam_full, alpha=0.55)),
        (f"ADPL-CAM top-1\n({top1_name})",
         overlay_heatmap(img_np, cam_top1, alpha=0.55)),
        (f"ADPL-CAM top-2\n({top2_name})",
         overlay_heatmap(img_np, cam_top2, alpha=0.55)),
        ("Query features\n(before fusion)",
         overlay_heatmap(img_np, query_max, alpha=0.55)),
        ("Fused features\n(after ref inject)",
         overlay_heatmap(img_np, fused_max, alpha=0.55)),
        ("Ref contribution\n(fused - query)",
         overlay_heatmap(img_np, delta_map, alpha=0.55)),
        (f"GradCAM\n(cls {gradcam_cls}: {class_names[gradcam_cls] if gradcam_cls < len(class_names) else '?'})",
         overlay_heatmap(img_np, gradcam_full, alpha=0.55)),
    ]

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), dpi=200)

    for i, (title, panel_img) in enumerate(panels):
        axes[i].imshow(panel_img)
        axes[i].set_title(title, fontsize=9, pad=4)
        axes[i].axis("off")

    fig.suptitle(f"{name}  (ref: {ref_name})", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_summary_grid(all_results, output_path, max_rows=8):
    cols = [
        "Original + GT", "Binary-agg CAM", "ADPL-CAM top-1", "ADPL-CAM top-2",
        "Query features", "Fused features", "GradCAM",
    ]
    n_rows = min(len(all_results), max_rows)
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), dpi=200)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, result in enumerate(all_results[:n_rows]):
        for col_idx, col_name in enumerate(cols):
            panel = result["panels"].get(col_name)
            if panel is not None:
                axes[row, col_idx].imshow(panel)
            axes[row, col_idx].axis("off")
            if row == 0:
                axes[row, col_idx].set_title(col_name, fontsize=9, pad=4)
        axes[row, 0].set_ylabel(
            result["name"], fontsize=7, rotation=0, labelpad=80, va="center",
        )

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Summary grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SPDNet ADPL-CAM visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_dir", default="")
    parser.add_argument("--cam_dir", default="")
    parser.add_argument("--label_file", required=True,
                        help="Path to .npy label dict for reference pairing")
    parser.add_argument("--output_dir",
                        default="outputs/visualizations/spdnet_val_cam_exploration")
    parser.add_argument("--num_images", type=int, default=25)
    parser.add_argument("--num_classes", type=int, default=115)
    parser.add_argument("--max_size", type=int, default=784)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading SPDNet from {args.checkpoint}")
    model = load_model(args.checkpoint, args.num_classes, device)

    label_dict = np.load(args.label_file, allow_pickle=True).item()
    ref_pool = build_reference_pool(label_dict, Path(args.image_dir))

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    cam_dir = Path(args.cam_dir) if args.cam_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(f.stem for f in image_dir.glob("*.jpg"))
    if gt_dir:
        gt_stems = {f.stem for f in gt_dir.glob("*.png")}
        all_images = [n for n in all_images if n in gt_stems]

    random.seed(args.seed)
    selected = random.sample(all_images, min(args.num_images, len(all_images)))
    selected.sort()
    log.info(f"Selected {len(selected)} images for visualization")

    tfm = get_val_transform()
    class_names = DISEASE_CLASSES[1:]

    all_results = []

    for name in tqdm(selected, desc="Visualizing"):
        img_pil = Image.open(image_dir / f"{name}.jpg").convert("RGB")
        long_side = max(img_pil.size)
        if long_side > args.max_size:
            ratio = args.max_size / long_side
            img_pil = img_pil.resize(
                (round(img_pil.width * ratio), round(img_pil.height * ratio)),
                resample=Image.BICUBIC,
            )
        img_np = np.array(img_pil)
        img_h, img_w = img_np.shape[:2]

        gt_mask = None
        if gt_dir and (gt_dir / f"{name}.png").exists():
            gt_pil = Image.open(gt_dir / f"{name}.png")
            if gt_pil.size != img_pil.size:
                gt_pil = gt_pil.resize(img_pil.size, resample=Image.NEAREST)
            gt_mask = np.array(gt_pil)

        binary_cam_full = np.zeros((img_h, img_w), dtype=np.float32)
        if cam_dir and (cam_dir / f"{name}.npy").exists():
            d = np.load(str(cam_dir / f"{name}.npy"), allow_pickle=True).item()
            raw_cam = d.get(0)
            if raw_cam is not None:
                binary_cam_full = normalize_map(
                    to_img_size(raw_cam, img_w, img_h)
                    if raw_cam.shape != (img_h, img_w) else raw_cam
                )

        label = label_dict.get(name)
        active_classes = np.where(label > 0)[0].tolist() if label is not None else []
        ref_cls = active_classes[0] if active_classes else 0
        candidates = [n for n in ref_pool.get(ref_cls, []) if n != name]
        rng = random.Random(args.seed + hash(name))
        ref_name = rng.choice(candidates) if candidates else name

        ref_pil = Image.open(image_dir / f"{ref_name}.jpg").convert("RGB")
        ref_long = max(ref_pil.size)
        if ref_long > args.max_size:
            r = args.max_size / ref_long
            ref_pil = ref_pil.resize(
                (round(ref_pil.width * r), round(ref_pil.height * r)),
                resample=Image.BICUBIC,
            )

        q_tensor = tfm(img_pil)
        r_tensor = tfm(ref_pil)

        activations = extract_spdnet_activations(model, q_tensor, r_tensor, device)

        try:
            gradcam_map, gradcam_cls = compute_gradcam(
                model, q_tensor, r_tensor, device, target_class=None,
            )
        except RuntimeError:
            fh, fw = activations["per_class_cam"].shape[1:]
            gradcam_map = np.zeros((fh, fw), dtype=np.float32)
            gradcam_cls = 0
            log.warning(f"GradCAM failed for {name}, using zeros")

        make_image_grid(
            img_np, gt_mask, binary_cam_full, activations,
            gradcam_map, gradcam_cls, name,
            output_dir / f"{name}_full.png",
            class_names, ref_name,
        )

        per_class_cam = activations["per_class_cam"]
        query_feat = activations["query_feat"]
        fused_feat = activations["fused_feat"]
        energy = per_class_cam.sum(axis=(1, 2))
        top_k = np.argsort(energy)[::-1]

        def to_img(a):
            return to_img_size(a, img_w, img_h)

        panels = {
            "Original + GT": (
                overlay_mask(img_np, gt_mask) if gt_mask is not None else img_np
            ),
            "Binary-agg CAM": overlay_heatmap(img_np, binary_cam_full, alpha=0.55),
            "ADPL-CAM top-1": overlay_heatmap(
                img_np, normalize_map(to_img(per_class_cam[top_k[0]])), alpha=0.55,
            ),
            "ADPL-CAM top-2": overlay_heatmap(
                img_np, normalize_map(to_img(per_class_cam[top_k[1]])), alpha=0.55,
            ),
            "Query features": overlay_heatmap(
                img_np, normalize_map(to_img(query_feat.max(axis=0))), alpha=0.55,
            ),
            "Fused features": overlay_heatmap(
                img_np, normalize_map(to_img(fused_feat.max(axis=0))), alpha=0.55,
            ),
            "GradCAM": overlay_heatmap(
                img_np, normalize_map(to_img(gradcam_map)), alpha=0.55,
            ),
        }
        all_results.append({"name": name, "panels": panels})

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    make_summary_grid(all_results, output_dir / "summary_grid.png", max_rows=8)

    log.info(f"Done. {len(selected)} individual figures + summary grid -> {output_dir}")


if __name__ == "__main__":
    main()
