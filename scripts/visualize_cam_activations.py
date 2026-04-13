"""Comprehensive CAM & attention visualization for MCTformer MC115.

Generates high-resolution grid comparisons of different activation types:
  1. Original image + GT overlay
  2. Binary-aggregated CAM (max of 115 classes)
  3. MCT class-token attention (top-1 class token by energy)
  4. MCT class-token attention (top-2 class token)
  5. Feature-map response (patchcam, max across classes)
  6. Fused CAM for top-1 class (before aggregation)
  7. Averaged patch self-attention
  8. GradCAM from classification head

Usage:
    export PATH="/venv/main/bin:$PATH"
    python scripts/visualize_cam_activations.py \
        --checkpoint outputs/mctformer_plantseg_multiclass/2026-03-08_11-32-35/checkpoints/last.ckpt \
        --image_dir data/plantsegv3/images/val \
        --gt_dir outputs/plantseg_binary_mc115/gt_binary_val \
        --cam_dir outputs/plantseg_binary_mc115/cams/cam_npy_val \
        --output_dir outputs/visualizations/val_cam_exploration \
        --num_images 25 \
        --seed 42
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantseg import DISEASE_CLASSES
from src.wsss.mctformer.model import create_mctformer_v2

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

GT_COLOR = np.array([0.15, 0.85, 0.30])
DISEASE_CMAP = LinearSegmentedColormap.from_list(
    "disease", [(0, 0, 0.15), (0, 0.1, 0.8), (0.9, 0.9, 0), (1, 0.2, 0)], N=256
)


def load_model(checkpoint: str, num_classes: int, input_size: int, device: torch.device):
    model = create_mctformer_v2(num_classes=num_classes, pretrained=False, input_size=input_size)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Convert normalized tensor back to uint8 image."""
    img = tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(img_np, heatmap, alpha=0.5, cmap=cv2.COLORMAP_JET):
    """Overlay a [0,1] heatmap on an RGB image."""
    hm_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    hm_color = cv2.applyColorMap(hm_uint8, cmap)[:, :, ::-1]  # BGR->RGB
    blended = img_np.astype(np.float32) / 255 * (1 - alpha) + hm_color.astype(np.float32) / 255 * alpha
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def overlay_mask(img_np, mask, color=GT_COLOR, alpha=0.45):
    """Overlay a binary mask with a solid color."""
    img_f = img_np.astype(np.float32) / 255
    result = img_f.copy()
    fg = mask > 0
    result[fg] = result[fg] * (1 - alpha) + color * alpha
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def normalize_map(x):
    """Min-max normalize to [0, 1]."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


@torch.no_grad()
def extract_all_activations(model, img_tensor, device, num_classes=115, n_layers=3):
    """Run MCTformer forward pass and extract all activation components.

    Returns dict with:
        mtatt: (num_classes, fh, fw) class-token-to-patch attention
        feature_map: (num_classes, fh, fw) ReLU'd patch feature map from conv head
        fused_cam: (num_classes, fh, fw) sqrt(mtatt * feature_map)
        patch_attn_avg: (fh*fw, fh*fw) averaged patch self-attention
        cls_logits: (num_classes,) per-class logits
        attn_weights_raw: list of per-layer attention weights
    """
    x = img_tensor.unsqueeze(0).to(device)
    w, h = x.shape[2:]

    x_cls, x_patch, attn_weights, _all_x_cls = model.forward_features(x)
    n, p, c = x_patch.shape
    w0 = w // model.patch_embed.patch_size[0]
    h0 = h // model.patch_embed.patch_size[0]

    x_patch_2d = x_patch.reshape(n, w0, h0, c).permute(0, 3, 1, 2).contiguous()
    feature_map = F.relu(model.head(x_patch_2d))[0].cpu().numpy()  # (num_classes, fh, fw)

    attn_stack = torch.stack(attn_weights)  # (depth, B, heads, N, N)
    attn_stack = attn_stack.mean(dim=2)  # average over heads: (depth, B, N, N)

    mtatt = (
        attn_stack[-n_layers:]
        .mean(0)[0, :num_classes, num_classes:]
        .reshape(num_classes, w0, h0)
        .cpu().numpy()
    )

    patch_attn = attn_stack[:, 0, num_classes:, num_classes:]  # (depth, npatches, npatches)
    patch_attn_avg = patch_attn.mean(0).cpu().numpy()  # (npatches, npatches)

    fused_cam = np.sqrt(np.maximum(mtatt * feature_map, 0))

    cls_logits = x_cls.mean(-1)[0].cpu().numpy()  # (num_classes,)

    return {
        "mtatt": mtatt,
        "feature_map": feature_map,
        "fused_cam": fused_cam,
        "patch_attn_avg": patch_attn_avg,
        "cls_logits": cls_logits,
        "spatial_shape": (w0, h0),
    }


def compute_gradcam(model, img_tensor, device, target_class=None, num_classes=115):
    """Compute GradCAM using the Conv2d classification head.

    Hooks into the patch features before the Conv2d head to get
    gradients of the target class w.r.t. the spatial feature map.
    """
    x = img_tensor.unsqueeze(0).to(device).requires_grad_(False)
    model.eval()

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["feat"] = inp[0].detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_in[0].detach()

    handle_fwd = model.head.register_forward_hook(fwd_hook)
    handle_bwd = model.head.register_full_backward_hook(bwd_hook)

    try:
        x_req = img_tensor.unsqueeze(0).to(device)
        x_req.requires_grad_(True)

        w, h = x_req.shape[2:]
        x_cls, x_patch, attn_weights, _all_x_cls = model.forward_features(x_req)
        n, p, c = x_patch.shape
        w0 = w // model.patch_embed.patch_size[0]
        h0 = h // model.patch_embed.patch_size[0]

        x_patch_2d = x_patch.reshape(n, w0, h0, c).permute(0, 3, 1, 2).contiguous()
        out = model.head(x_patch_2d)  # (1, num_classes, fh, fw)

        cls_scores = out.mean(dim=[2, 3])[0]  # (num_classes,)
        if target_class is None:
            target_class = cls_scores.argmax().item()

        model.zero_grad()
        cls_scores[target_class].backward()

        feats = activations["feat"][0]  # (C, fh, fw)
        grads = gradients["feat"][0]    # (C, fh, fw)

        weights = grads.mean(dim=[1, 2])  # (C,)
        gradcam = (weights[:, None, None] * feats).sum(0)
        gradcam = F.relu(gradcam).cpu().numpy()

        return normalize_map(gradcam), target_class
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def make_image_grid(img_np, gt_mask, activations, gradcam_map, gradcam_cls,
                    cam_npy, name, output_path, class_names):
    """Create a high-resolution multi-panel figure for one image."""
    mtatt = activations["mtatt"]
    feature_map = activations["feature_map"]
    fused_cam = activations["fused_cam"]
    patch_attn_avg = activations["patch_attn_avg"]
    cls_logits = activations["cls_logits"]
    w0, h0 = activations["spatial_shape"]
    img_h, img_w = img_np.shape[:2]

    def to_img_size(arr_2d):
        """Resize feature-map-resolution array to image size."""
        pil = Image.fromarray(arr_2d.astype(np.float32), mode="F")
        return np.array(pil.resize((img_w, img_h), Image.BILINEAR))

    energy_per_class = fused_cam.sum(axis=(1, 2))
    top_k_idx = np.argsort(energy_per_class)[::-1]
    top1_cls, top2_cls = top_k_idx[0], top_k_idx[1]

    top1_name = class_names[top1_cls] if top1_cls < len(class_names) else f"cls_{top1_cls}"
    top2_name = class_names[top2_cls] if top2_cls < len(class_names) else f"cls_{top2_cls}"

    binary_cam = normalize_map(cam_npy) if cam_npy is not None else normalize_map(fused_cam.max(axis=0))
    binary_cam_full = to_img_size(binary_cam) if binary_cam.shape != (img_h, img_w) else binary_cam

    mtatt_top1 = normalize_map(to_img_size(mtatt[top1_cls]))
    mtatt_top2 = normalize_map(to_img_size(mtatt[top2_cls]))

    featmap_max = normalize_map(to_img_size(feature_map.max(axis=0)))

    fused_top1 = normalize_map(to_img_size(fused_cam[top1_cls]))

    patch_attn_spatial = patch_attn_avg.mean(axis=0).reshape(w0, h0)
    patch_attn_full = normalize_map(to_img_size(patch_attn_spatial))

    gradcam_full = normalize_map(to_img_size(gradcam_map)) if gradcam_map.shape != (img_h, img_w) else normalize_map(gradcam_map)

    panels = [
        ("Original + GT", overlay_mask(img_np, gt_mask, GT_COLOR, 0.45) if gt_mask is not None else img_np),
        ("Binary-agg CAM", overlay_heatmap(img_np, binary_cam_full, alpha=0.55)),
        (f"MCT attn top-1\n({top1_name})", overlay_heatmap(img_np, mtatt_top1, alpha=0.55)),
        (f"MCT attn top-2\n({top2_name})", overlay_heatmap(img_np, mtatt_top2, alpha=0.55)),
        ("Feature map (max)", overlay_heatmap(img_np, featmap_max, alpha=0.55)),
        (f"Fused CAM top-1\n({top1_name})", overlay_heatmap(img_np, fused_top1, alpha=0.55)),
        ("Patch self-attn\n(averaged)", overlay_heatmap(img_np, patch_attn_full, alpha=0.55)),
        (f"GradCAM\n(cls {gradcam_cls}: {class_names[gradcam_cls] if gradcam_cls < len(class_names) else '?'})",
         overlay_heatmap(img_np, gradcam_full, alpha=0.55)),
    ]

    n_cols = len(panels)
    fig_w = 4.5 * n_cols
    fig_h = 4.5
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), dpi=200)

    for i, (title, panel_img) in enumerate(panels):
        axes[i].imshow(panel_img)
        axes[i].set_title(title, fontsize=9, pad=4)
        axes[i].axis("off")

    fig.suptitle(name, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_summary_grid(all_results, output_path, max_rows=8):
    """Create a summary grid: rows = images, columns = key visualization types."""
    cols = ["Original + GT", "Binary-agg CAM", "MCT attn top-1", "MCT attn top-2",
            "Fused CAM top-1", "GradCAM"]
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
        axes[row, 0].set_ylabel(result["name"], fontsize=7, rotation=0, labelpad=80, va="center")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Summary grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive MCTformer CAM visualization")
    parser.add_argument("--checkpoint", required=True, help="MCTformer checkpoint path")
    parser.add_argument("--image_dir", required=True, help="Validation images directory")
    parser.add_argument("--gt_dir", default="", help="Binary GT mask directory")
    parser.add_argument("--cam_dir", default="", help="Pre-generated binary CAM .npy directory")
    parser.add_argument("--output_dir", default="outputs/visualizations/val_cam_exploration")
    parser.add_argument("--num_images", type=int, default=25)
    parser.add_argument("--num_classes", type=int, default=115)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--max_size", type=int, default=0, help="Max long side (0=auto from input_size*1.75)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_size <= 0:
        args.max_size = int(args.input_size * 1.75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.num_classes, args.input_size, device)

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
    class_names = DISEASE_CLASSES[1:]  # 115 foreground classes

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

        gt_mask = None
        if gt_dir and (gt_dir / f"{name}.png").exists():
            gt_pil = Image.open(gt_dir / f"{name}.png")
            if gt_pil.size != img_pil.size:
                gt_pil = gt_pil.resize(img_pil.size, resample=Image.NEAREST)
            gt_mask = np.array(gt_pil)

        cam_npy = None
        if cam_dir and (cam_dir / f"{name}.npy").exists():
            d = np.load(str(cam_dir / f"{name}.npy"), allow_pickle=True).item()
            cam_npy = d.get(0)

        img_tensor = tfm(img_pil)
        activations = extract_all_activations(
            model, img_tensor, device,
            num_classes=args.num_classes, n_layers=args.n_layers,
        )

        try:
            gradcam_map, gradcam_cls = compute_gradcam(
                model, img_tensor, device,
                target_class=None, num_classes=args.num_classes,
            )
        except RuntimeError:
            w0, h0 = activations["spatial_shape"]
            gradcam_map = np.zeros((w0, h0), dtype=np.float32)
            gradcam_cls = 0
            log.warning(f"GradCAM OOM for {name}, using zeros")

        per_image_path = output_dir / f"{name}_full.png"
        make_image_grid(
            img_np, gt_mask, activations, gradcam_map, gradcam_cls,
            cam_npy, name, per_image_path, class_names,
        )

        img_h, img_w = img_np.shape[:2]
        mtatt = activations["mtatt"]
        fused_cam = activations["fused_cam"]
        w0, h0 = activations["spatial_shape"]
        patch_attn_avg = activations["patch_attn_avg"]

        def to_img(a):
            return np.array(Image.fromarray(a.astype(np.float32), mode="F").resize((img_w, img_h), Image.BILINEAR))

        energy = fused_cam.sum(axis=(1, 2))
        top_k = np.argsort(energy)[::-1]

        binary_cam_full = to_img(normalize_map(cam_npy)) if cam_npy is not None else to_img(normalize_map(fused_cam.max(axis=0)))

        panels = {
            "Original + GT": overlay_mask(img_np, gt_mask, GT_COLOR, 0.45) if gt_mask is not None else img_np,
            "Binary-agg CAM": overlay_heatmap(img_np, binary_cam_full, alpha=0.55),
            "MCT attn top-1": overlay_heatmap(img_np, normalize_map(to_img(mtatt[top_k[0]])), alpha=0.55),
            "MCT attn top-2": overlay_heatmap(img_np, normalize_map(to_img(mtatt[top_k[1]])), alpha=0.55),
            "Fused CAM top-1": overlay_heatmap(img_np, normalize_map(to_img(fused_cam[top_k[0]])), alpha=0.55),
            "GradCAM": overlay_heatmap(img_np, normalize_map(to_img(gradcam_map)) if gradcam_map.shape != (img_h, img_w) else normalize_map(gradcam_map), alpha=0.55),
        }
        all_results.append({"name": name, "panels": panels})

        if device.type == "cuda":
            torch.cuda.empty_cache()
            import gc; gc.collect()

    make_summary_grid(all_results, output_dir / "summary_grid.png", max_rows=8)

    log.info(f"Done. {len(selected)} individual figures + summary grid saved to {output_dir}")


if __name__ == "__main__":
    main()
