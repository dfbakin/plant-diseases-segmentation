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

from src.wsss.spdnet.gradient_cam_methods import (
    MAX_CLASSES_PER_IMAGE,
    generate_gradient_spdnet_seed,
    is_gradient_cam_mode,
    list_methods as list_gradient_cam_methods,
)
from src.wsss.spdnet.model import SPDNet

log = logging.getLogger(__name__)

SEED = 42


def load_spdnet_from_checkpoint(
    checkpoint: str,
    num_classes: int,
    fpn_channels: int = 256,
    mse_reduction: int = 4,
    fusion_mode: str | None = None,
    ref_pool_size: int | None = None,
) -> SPDNet:
    """Load SPDNet from a Lightning checkpoint.

    If *fusion_mode* is ``None``, it is auto-detected from saved
    hyperparameters (falls back to ``"token"`` for old checkpoints).

    If *ref_pool_size* is ``None``, it is auto-detected from saved
    hyperparameters (falls back to the SPDNet default of 14 for old
    checkpoints that pre-date the rps logging in ``SPDNetModule.__init__``).
    Auto-detection matters: ``ref_pool_size`` controls the spatial K side
    of the SCA attention buffer ``(B, heads, Q, K^2)`` -- loading a rps=56
    checkpoint into a rps=14 model will silently produce incorrect CAMs
    because the AdaptiveAvgPool2d on the reference path collapses keys to
    a different grid size.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        sd = ckpt.get("model", ckpt)

    hp = ckpt.get("hyper_parameters", {})
    if fusion_mode is None:
        fusion_mode = hp.get("fusion_mode", "token")
    if ref_pool_size is None:
        ref_pool_size = int(hp.get("ref_pool_size", 14))

    model = SPDNet(
        num_classes=num_classes,
        fpn_channels=fpn_channels,
        mse_reduction=mse_reduction,
        pretrained=False,
        fusion_mode=fusion_mode,
        ref_pool_size=ref_pool_size,
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


def generate_spdnet_seed(
    model: SPDNet,
    query_images: list[torch.Tensor],
    ref_image_lists: list[list[torch.Tensor]],
    device: torch.device,
    seed_mode: str = "feat_chmean",
    active_classes: list[int] | None = None,
    target_layer: str = "query_merged",
    max_classes_per_image: int = MAX_CLASSES_PER_IMAGE,
) -> dict[int, np.ndarray]:
    """Generate a seed map from one image with multi-scale / flip TTA.

    Dispatches between three code paths:

      * **Feature / attention readouts** (``feat_*``, ``fused_*``,
        ``spatial_proto``, ``attn_map``, ``attn_max``): pure no-grad
        channel-wise aggregations of intermediate tensors. This is the
        original implementation and is unchanged.
      * **Classifier CAM** (``cam_max``): produced by
        ``generate_spdnet_cam`` / ``generate_all_cams`` elsewhere; not
        routed through this function.
      * **Gradient-based CAMs** (``layercam``, ``gradcam_pp``,
        ``xgradcam``): require gradients w.r.t. a captured intermediate
        activation and an explicit active-class list. Delegated to
        ``generate_gradient_spdnet_seed``.

    Args:
        query_images: list of (1, 3, H_s, W_s) tensors at different scales/flips.
        ref_image_lists: for each scale/flip, list of N reference tensors.
        device: computation device.
        seed_mode: one of:
            * ``feat_chmean``, ``feat_neg_chmean``, ``feat_chvar``,
              ``feat_chmax``, ``feat_l2norm``   (pre-fusion ``query_merged``)
            * ``fused_<any of above>``         (post-fusion ``fused``)
            * ``spatial_proto``                (cosine to ref prototype)
            * ``attn_map``, ``attn_max``       (spatial attention read-outs;
              the direct targets of ``L_ac`` / ``L_marg_H``). Only valid when
              the model runs in spatial-attention fusion mode.
            * ``layercam``, ``gradcam_pp``, ``xgradcam``  (gradient CAMs;
              require ``active_classes`` to be set).
        active_classes: foreground class indices to aggregate for
            gradient-CAM modes. Ignored by all other modes. Required
            (non-empty) when ``seed_mode`` is a gradient-CAM mode.
        target_layer: target layer for gradient-CAM modes; one of
            ``"query_merged"`` (default, probe P3), ``"fused"`` (P4),
            ``"layer4"`` (P1).
        max_classes_per_image: cap on gradient-CAM active-class count
            (one backward per class).

    Returns:
        {0: float32_2d_array} min-max normalized to [0, 1].
    """
    # Normalise aliases: users often type "fused_chvar" meaning the post-fusion
    # channel variance. Internally we keep the "fused_feat_*" canonical form.
    _FUSED_ALIASES = {
        "fused_chmean": "fused_feat_chmean",
        "fused_neg_chmean": "fused_feat_neg_chmean",
        "fused_chvar": "fused_feat_chvar",
        "fused_chmax": "fused_feat_chmax",
        "fused_l2norm": "fused_feat_l2norm",
    }
    seed_mode = _FUSED_ALIASES.get(seed_mode, seed_mode)

    # Gradient-CAM dispatch. Uses ``torch.enable_grad()`` internally
    # (the @torch.no_grad() decorator is gone from this function so
    # the dispatch branch below is free to set its own grad context).
    if is_gradient_cam_mode(seed_mode):
        if not active_classes:
            raise ValueError(
                f"seed_mode={seed_mode!r} is a gradient-CAM method "
                f"({list_gradient_cam_methods()}) and requires a non-empty "
                f"active_classes list (got {active_classes!r}). "
                f"generate_all_seeds passes it automatically from the "
                f"per-image label_dict."
            )
        return generate_gradient_spdnet_seed(
            model=model,
            query_images=query_images,
            ref_image_lists=ref_image_lists,
            active_classes=active_classes,
            device=device,
            method=seed_mode,  # type: ignore[arg-type]
            target_layer=target_layer,  # type: ignore[arg-type]
            max_classes_per_image=max_classes_per_image,
        )

    # Non-gradient path: unchanged, wrap in an explicit no_grad block.
    with torch.no_grad():
        return _no_grad_seed(
            model=model,
            query_images=query_images,
            ref_image_lists=ref_image_lists,
            device=device,
            seed_mode=seed_mode,
        )


def _no_grad_seed(
    model: SPDNet,
    query_images: list[torch.Tensor],
    ref_image_lists: list[list[torch.Tensor]],
    device: torch.device,
    seed_mode: str,
) -> dict[int, np.ndarray]:
    """Original no-grad seed generation (feat_* / fused_* / attn_* / spatial_proto).

    Separated out from ``generate_spdnet_seed`` so the dispatcher can
    route gradient-CAM modes through an ``enable_grad`` path without
    duplicating logic. Callers should prefer ``generate_spdnet_seed``.
    """
    h_orig = query_images[0].shape[2]
    w_orig = query_images[0].shape[3]
    seed_2d_list = []
    needs_attn = seed_mode in {"attn_map", "attn_max"}

    for s, (q_img, r_imgs) in enumerate(zip(query_images, ref_image_lists)):
        q_img = q_img.to(device)
        refs = [r.to(device) for r in r_imgs]

        feats = model.extract_merged_features(
            q_img, refs if len(refs) > 1 else refs[0],
            return_attn=needs_attn,
        )

        if needs_attn:
            if "attn_map" not in feats:
                raise ValueError(
                    f"seed_mode={seed_mode!r} requires spatial-attention fusion "
                    "but the model did not return attn_map (is fusion_mode='spatial'?)"
                )
            if seed_mode == "attn_map":
                # (B, H_q, W_q) in [0, 1], already head-averaged normalized
                # negative entropy (higher = more concentrated).
                feat_map = feats["attn_map"].unsqueeze(1)  # (B, 1, H_q, W_q)
            else:  # attn_max
                # attn_w: (B, P, N_k). Max attention weight per query position.
                attn_w = feats["attn_w"]
                q_hw = feats["query_merged"].shape[-2:]
                peak = attn_w.max(dim=-1).values                  # (B, P)
                feat_map = peak.view(-1, 1, *q_hw)                # (B, 1, H_q, W_q)
        else:
            feat_src = (
                feats.get("fused", feats["query_merged"])
                if seed_mode.startswith("fused_")
                else feats["query_merged"]
            )
            mode_key = seed_mode.removeprefix("fused_")

            if mode_key == "feat_chmean":
                feat_map = feat_src.mean(dim=1, keepdim=True)
            elif mode_key == "feat_neg_chmean":
                feat_map = -feat_src.mean(dim=1, keepdim=True)
            elif mode_key == "feat_chvar":
                feat_map = feat_src.var(dim=1, keepdim=True)
            elif mode_key == "feat_chmax":
                feat_map = feat_src.amax(dim=1, keepdim=True)
            elif mode_key == "feat_l2norm":
                feat_map = feat_src.norm(dim=1, keepdim=True)
            elif mode_key == "spatial_proto":
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
    ref_pool: dict[int, list[str]] | None = None,
    ref_image_dir: Path | None = None,
    query_class_resolver=None,
    target_layer: str = "query_merged",
    max_classes_per_image: int = MAX_CLASSES_PER_IMAGE,
) -> list[str]:
    """Generate feature-based seed maps for all images.

    Same image preparation as generate_all_cams but dispatches to
    generate_spdnet_seed instead of generate_spdnet_cam.

    Args:
        ref_pool: Pre-built {class_idx: [ref_image_names]} mapping. If
            ``None``, built from ``label_dict`` + ``image_dir`` (legacy
            behaviour, only correct when ``label_dict`` truly contains
            multi-class labels for the query images).
        ref_image_dir: Directory holding reference images. Defaults to
            ``image_dir`` (queries and refs share a directory). Pass a
            different directory to draw refs from a separate set
            (e.g. queries from val, refs from train).
        query_class_resolver: Optional callable ``(name) -> int`` that
            returns the class index used for reference selection. Useful
            when the query labels are a binary fallback (label[0]=1) but
            the true class can be parsed from the filename. If ``None``,
            falls back to ``label_dict[name].argmax()``.
        target_layer: (gradient-CAM modes only) ``"query_merged"``
            (default, probe P3), ``"fused"`` (P4), or ``"layer4"`` (P1).
        max_classes_per_image: (gradient-CAM modes only) hard cap on the
            number of active classes per image (one backward per class).
    """
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ref_pool is None:
        ref_pool = build_reference_pool(label_dict, image_dir, image_ext)
    if ref_image_dir is None:
        ref_image_dir = image_dir
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

        if query_class_resolver is not None:
            resolved = query_class_resolver(name)
            ref_cls = resolved if resolved is not None else active_classes[0]
        else:
            ref_cls = active_classes[0]
        ref_names = ref_pool.get(ref_cls, [])
        ref_names = [n for n in ref_names if n != name]
        if not ref_names:
            ref_names = [name]
        ref_picks = random.choices(ref_names, k=num_ref_images)

        ref_pils = []
        for rn in ref_picks:
            rpil = PIL.Image.open(ref_image_dir / f"{rn}{image_ext}").convert("RGB")
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
            model, query_imgs, ref_img_lists, device, seed_mode,
            active_classes=active_classes,
            target_layer=target_layer,
            max_classes_per_image=max_classes_per_image,
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
    ref_pool: dict[int, list[str]] | None = None,
    ref_image_dir: Path | None = None,
    query_class_resolver=None,
) -> list[str]:
    """Generate CAMs for all images and save as .npy files.

    All references are passed to the model simultaneously so that token
    averaging happens inside the model, consistent with training.

    Args:
        ref_pool, ref_image_dir, query_class_resolver: see
            ``generate_all_seeds``. Pass these to draw references from a
            different set than the queries (e.g. queries from val,
            references from train).

    Returns the list of processed image names.
    """
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ref_pool is None:
        ref_pool = build_reference_pool(label_dict, image_dir, image_ext)
    if ref_image_dir is None:
        ref_image_dir = image_dir
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

        if query_class_resolver is not None:
            resolved = query_class_resolver(name)
            ref_cls = resolved if resolved is not None else active_classes[0]
        else:
            ref_cls = active_classes[0]
        ref_names = ref_pool.get(ref_cls, [])
        ref_names = [n for n in ref_names if n != name]
        if not ref_names:
            ref_names = [name]

        ref_picks = random.choices(ref_names, k=num_ref_images)

        ref_pils = []
        for rn in ref_picks:
            rpil = PIL.Image.open(ref_image_dir / f"{rn}{image_ext}").convert("RGB")
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
