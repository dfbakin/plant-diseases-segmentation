"""ADPL-CAM generation from a trained SPDNet checkpoint.

Generates per-image CAMs using reference-guided token fusion.
Supports multi-scale + flip augmentation, multiple references, and binary
aggregation.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

from src.wsss.spdnet.model import SPDNet

log = logging.getLogger(__name__)

SEED = 42


def load_spdnet_from_checkpoint(
    checkpoint: str,
    num_classes: int,
    fpn_channels: int = 256,
    mse_reduction: int = 4,
) -> SPDNet:
    """Load SPDNet from a Lightning checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        sd = ckpt.get("model", ckpt)

    model = SPDNet(
        num_classes=num_classes,
        fpn_channels=fpn_channels,
        mse_reduction=mse_reduction,
        pretrained=False,
    )
    model.load_state_dict(sd, strict=False)
    return model


def build_reference_pool(
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
    image_ext: str,
) -> dict[int, list[str]]:
    """Build class -> [image_name] mapping for reference selection."""
    pool: dict[int, list[str]] = defaultdict(list)
    for name, label in label_dict.items():
        if not (image_dir / f"{name}{image_ext}").exists():
            continue
        active = np.where(label > 0)[0]
        for cls in active:
            pool[int(cls)].append(name)
    return pool


@torch.no_grad()
def generate_spdnet_cam(
    model: SPDNet,
    query_images: list[torch.Tensor],
    ref_image_lists: list[list[torch.Tensor]],
    num_classes: int,
    device: torch.device,
    binary_aggregate: str = "max",
) -> dict[int, np.ndarray]:
    """Generate ADPL-CAM for one query with multi-scale + flip + multi-ref.

    Args:
        query_images: list of (1, 3, H_s, W_s) tensors at different scales/flips
        ref_image_lists: for each scale/flip, a list of N reference tensors
                         (each (1, 3, H_s, W_s))
        num_classes:   number of foreground classes
        device:        computation device
        binary_aggregate: "max", "mean", or "top_energy" to merge per-class CAMs

    Returns:
        cam_dict: {0: 2D_array} when binary_aggregate, else {cls_idx: 2D_array}
    """
    h_orig = query_images[0].shape[2]
    w_orig = query_images[0].shape[3]

    cam_list = []
    for s, (q_img, r_imgs) in enumerate(zip(query_images, ref_image_lists)):
        q_img = q_img.to(device)
        refs = [r.to(device) for r in r_imgs]

        if len(refs) == 1:
            _, cam = model(q_img, refs[0], return_cam=True)
        else:
            _, cam = model(q_img, refs, return_cam=True)

        cam = F.interpolate(cam, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        cam = cam[0].cpu().numpy()  # (num_classes, H, W)

        if s % 2 == 1:
            cam = np.flip(cam, axis=-1)
        cam_list.append(cam)

    sum_cam = np.sum(cam_list, axis=0)  # (num_classes, H, W)

    if binary_aggregate:
        if binary_aggregate == "max":
            merged = np.max(sum_cam, axis=0)
        elif binary_aggregate == "mean":
            merged = np.mean(sum_cam, axis=0)
        elif binary_aggregate == "top_energy":
            energy = sum_cam.sum(axis=(1, 2))
            top_cls = int(np.argmax(energy))
            merged = sum_cam[top_cls]
        else:
            raise ValueError(f"Unknown binary_aggregate: {binary_aggregate!r}")
        merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-8)
        return {0: merged.astype(np.float32)}

    cam_dict: dict[int, np.ndarray] = {}
    for c in range(num_classes):
        cls_cam = sum_cam[c]
        if cls_cam.max() - cls_cam.min() > 1e-8:
            cls_cam = (cls_cam - cls_cam.min()) / (cls_cam.max() - cls_cam.min() + 1e-8)
            cam_dict[c] = cls_cam.astype(np.float32)
    return cam_dict


@torch.no_grad()
def generate_spdnet_seed(
    model: SPDNet,
    query_images: list[torch.Tensor],
    ref_image_lists: list[list[torch.Tensor]],
    device: torch.device,
    seed_mode: str = "feat_chmean",
) -> dict[int, np.ndarray]:
    """Generate a feature-based seed map (no classifier projection).

    Args:
        query_images: list of (1, 3, H_s, W_s) tensors at different scales/flips
        ref_image_lists: for each scale/flip, list of N reference tensors
        device: computation device
        seed_mode: one of "feat_chmean", "feat_chmax", "spatial_proto"

    Returns:
        {0: float32_2d_array} min-max normalized to [0, 1]
    """
    h_orig = query_images[0].shape[2]
    w_orig = query_images[0].shape[3]
    seed_2d_list = []

    for s, (q_img, r_imgs) in enumerate(zip(query_images, ref_image_lists)):
        q_img = q_img.to(device)
        refs = [r.to(device) for r in r_imgs]

        feats = model.extract_merged_features(q_img, refs if len(refs) > 1 else refs[0])

        if seed_mode == "feat_chmean":
            feat_map = feats["query_merged"].mean(dim=1, keepdim=True)
        elif seed_mode == "feat_chmax":
            feat_map = feats["query_merged"].amax(dim=1, keepdim=True)
        elif seed_mode == "spatial_proto":
            ref_merged = feats["ref_merged"]
            proto = F.normalize(ref_merged.mean(dim=[2, 3]), dim=1)  # (B, C)
            query_norm = F.normalize(feats["query_merged"], dim=1)   # (B, C, H, W)
            sim = (query_norm * proto[:, :, None, None]).sum(dim=1, keepdim=True)
            feat_map = sim
        else:
            raise ValueError(f"Unknown seed_mode: {seed_mode!r}")

        feat_2d = F.interpolate(
            feat_map, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )
        seed = feat_2d[0, 0].cpu().numpy()  # (H, W)

        if s % 2 == 1:
            seed = np.flip(seed, axis=-1)
        seed_2d_list.append(seed)

    merged = np.mean(seed_2d_list, axis=0)
    vmin, vmax = merged.min(), merged.max()
    if vmax - vmin > 1e-8:
        merged = (merged - vmin) / (vmax - vmin)
    else:
        merged = np.zeros_like(merged)
    return {0: merged.astype(np.float32).copy()}


def generate_all_seeds(
    model: SPDNet,
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
    output_dir: Path,
    image_ext: str = ".jpg",
    scales: list[float] = [1.0, 0.75, 1.25],
    max_size: int = 0,
    input_size: int = 448,
    num_ref_images: int = 1,
    seed_mode: str = "feat_chmean",
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Generate feature-based seed maps for all images.

    Same image preparation as generate_all_cams but dispatches to
    generate_spdnet_seed instead of generate_spdnet_cam.
    """
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_pool = build_reference_pool(label_dict, image_dir, image_ext)
    max_long = max_size if max_size > 0 else int(input_size * 1.75)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    processed = []

    for name in tqdm(list(label_dict.keys()), desc=f"Generating seeds ({seed_mode})"):
        img_path = image_dir / f"{name}{image_ext}"
        if not img_path.exists():
            continue

        label = label_dict[name]
        active_classes = np.where(label > 0)[0].tolist()
        if not active_classes:
            continue

        query_pil = PIL.Image.open(img_path).convert("RGB")
        long_side = max(query_pil.size)
        if long_side > max_long:
            ratio = max_long / long_side
            query_pil = query_pil.resize(
                (round(query_pil.width * ratio), round(query_pil.height * ratio)),
                resample=PIL.Image.BICUBIC,
            )

        ref_cls = active_classes[0]
        ref_names = ref_pool.get(ref_cls, [])
        ref_names = [n for n in ref_names if n != name]
        if not ref_names:
            ref_names = [name]
        ref_picks = random.choices(ref_names, k=num_ref_images)

        ref_pils = []
        for rn in ref_picks:
            rpil = PIL.Image.open(image_dir / f"{rn}{image_ext}").convert("RGB")
            ls = max(rpil.size)
            if ls > max_long:
                r = max_long / ls
                rpil = rpil.resize(
                    (round(rpil.width * r), round(rpil.height * r)),
                    resample=PIL.Image.BICUBIC,
                )
            ref_pils.append(rpil)

        query_imgs: list[torch.Tensor] = []
        ref_img_lists: list[list[torch.Tensor]] = []

        for sc in scales:
            tw = round(query_pil.size[0] * sc)
            th = round(query_pil.size[1] * sc)
            scaled_long = max(tw, th)
            if scaled_long > max_long:
                r = max_long / scaled_long
                tw, th = round(tw * r), round(th * r)

            q_scaled = query_pil.resize((tw, th), resample=PIL.Image.BICUBIC)
            q_t = tfm(q_scaled).unsqueeze(0)

            r_tensors = []
            for rpil in ref_pils:
                r_sc = rpil.resize((tw, th), resample=PIL.Image.BICUBIC)
                r_tensors.append(tfm(r_sc).unsqueeze(0))

            query_imgs.append(q_t)
            ref_img_lists.append(r_tensors)

            query_imgs.append(torch.flip(q_t, [-1]))
            ref_img_lists.append([torch.flip(rt, [-1]) for rt in r_tensors])

        seed_dict = generate_spdnet_seed(
            model, query_imgs, ref_img_lists, device, seed_mode
        )

        if seed_dict:
            np.save(str(output_dir / f"{name}.npy"), seed_dict)
            processed.append(name)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info(f"Saved {len(processed)} seeds ({seed_mode}) to {output_dir}")
    return processed


def generate_all_cams(
    model: SPDNet,
    label_dict: dict[str, np.ndarray],
    image_dir: Path,
    output_dir: Path,
    image_ext: str = ".jpg",
    scales: list[float] = [1.0, 0.75, 1.25],
    max_size: int = 0,
    input_size: int = 448,
    num_ref_images: int = 1,
    binary_aggregate: str = "max",
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Generate CAMs for all images and save as .npy files.

    All references are passed to the model simultaneously so that token
    averaging happens inside the model, consistent with training.

    Returns the list of processed image names.
    """
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_pool = build_reference_pool(label_dict, image_dir, image_ext)
    max_long = max_size if max_size > 0 else int(input_size * 1.75)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    num_classes = model.num_classes
    processed = []

    for name in tqdm(list(label_dict.keys()), desc="Generating SPDNet CAMs"):
        img_path = image_dir / f"{name}{image_ext}"
        if not img_path.exists():
            continue

        label = label_dict[name]
        active_classes = np.where(label > 0)[0].tolist()
        if not active_classes:
            continue

        query_pil = PIL.Image.open(img_path).convert("RGB")
        long_side = max(query_pil.size)
        if long_side > max_long:
            ratio = max_long / long_side
            query_pil = query_pil.resize(
                (round(query_pil.width * ratio), round(query_pil.height * ratio)),
                resample=PIL.Image.BICUBIC,
            )

        ref_cls = active_classes[0]
        ref_names = ref_pool.get(ref_cls, [])
        ref_names = [n for n in ref_names if n != name]
        if not ref_names:
            ref_names = [name]

        ref_picks = random.choices(ref_names, k=num_ref_images)

        ref_pils = []
        for rn in ref_picks:
            rpil = PIL.Image.open(image_dir / f"{rn}{image_ext}").convert("RGB")
            ls = max(rpil.size)
            if ls > max_long:
                r = max_long / ls
                rpil = rpil.resize(
                    (round(rpil.width * r), round(rpil.height * r)),
                    resample=PIL.Image.BICUBIC,
                )
            ref_pils.append(rpil)

        query_imgs: list[torch.Tensor] = []
        ref_img_lists: list[list[torch.Tensor]] = []

        for sc in scales:
            tw = round(query_pil.size[0] * sc)
            th = round(query_pil.size[1] * sc)
            scaled_long = max(tw, th)
            if scaled_long > max_long:
                r = max_long / scaled_long
                tw, th = round(tw * r), round(th * r)

            q_scaled = query_pil.resize((tw, th), resample=PIL.Image.BICUBIC)
            q_t = tfm(q_scaled).unsqueeze(0)

            r_tensors = []
            for rpil in ref_pils:
                r_sc = rpil.resize((tw, th), resample=PIL.Image.BICUBIC)
                r_tensors.append(tfm(r_sc).unsqueeze(0))

            query_imgs.append(q_t)
            ref_img_lists.append(r_tensors)

            query_imgs.append(torch.flip(q_t, [-1]))
            ref_img_lists.append([torch.flip(rt, [-1]) for rt in r_tensors])

        cam_dict = generate_spdnet_cam(
            model, query_imgs, ref_img_lists, num_classes, device, binary_aggregate
        )

        if cam_dict:
            np.save(str(output_dir / f"{name}.npy"), cam_dict)
            processed.append(name)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info(f"Saved {len(processed)} CAMs to {output_dir}")
    return processed
