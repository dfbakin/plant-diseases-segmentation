"""Generate pseudo masks from a trained WeakCLIP checkpoint.

Replicates the original WeakCLIP evaluation pipeline:
  dist_test.sh --aug-test  ->  multi-scale + flip slide-window inference
  make_crf.py              ->  done separately in refine_weakclip_masks.py

Dataset-agnostic: reads image directory + optional names file.

Example:
    python src/generate_weakclip_masks.py \
        checkpoint=outputs/weakclip/weakclip-voc/checkpoints/last.ckpt \
        image_dir=data/VOC2012/JPEGImages \
        names_file=data/VOC2012/ImageSets/Segmentation/trainaug.txt \
        output_dir=outputs/weakclip_probs
"""

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.train_weakclip import build_weakclip_model, load_class_names, WeakCLIPTrainConfig
from src.wsss.weakclip.lightning import WeakCLIPModule

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class GenerateMasksConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    checkpoint: str = ""
    class_names_file: str = "outputs/labels/class_names.txt"
    image_dir: str = "data/VOC2012/JPEGImages"
    image_ext: str = ".jpg"
    names_file: str = ""
    output_dir: str = "outputs/weakclip_probs"

    scales: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )
    flip: bool = True
    crop_size: int = 512
    stride: int = 341

    clip_pretrained: str = "pretrained/ViT-B-16.pt"
    image_size: int = 512
    context_length: int = 5
    tau: float = 0.07
    num_classes: int = 0
    batch_size: int = 1
    checkpoint_format: str = "lightning"


cs = ConfigStore.instance()
cs.store(name="generate_masks_config", node=GenerateMasksConfig)


def _get_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _load_image(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Load image, return (H,W,3) uint8 numpy and (1,3,H,W) normalized tensor."""
    img_np = np.array(Image.open(path).convert("RGB"))
    normalize = _get_normalize()
    tensor = transforms.ToTensor()(img_np)
    tensor = normalize(tensor).unsqueeze(0)
    return img_np, tensor


def _pad_to_multiple(img: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad (1,C,H,W) so H and W are at least crop_size."""
    _, _, h, w = img.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return img, (h, w)


@torch.no_grad()
def slide_inference(
    model: torch.nn.Module,
    img: torch.Tensor,
    num_classes: int,
    crop_size: int = 512,
    stride: int = 341,
) -> torch.Tensor:
    """Slide-window inference matching mmseg test_cfg(mode='slide').

    Args:
        model: WeakCLIP model returning (seg_logits, score_map).
        img: (1, 3, H, W) tensor, already normalized.
        num_classes: Number of output classes.
        crop_size: Window size for each crop.
        stride: Step size between windows.

    Returns:
        (1, C, H, W) softmax probability map at the input resolution.
    """
    device = next(model.parameters()).device
    img = img.to(device)
    img, (orig_h, orig_w) = _pad_to_multiple(img, crop_size)
    _, _, h, w = img.shape

    count = torch.zeros(1, 1, h, w, device=device)
    pred = torch.zeros(1, num_classes, h, w, device=device)

    h_steps = max(1, math.ceil((h - crop_size) / stride) + 1)
    w_steps = max(1, math.ceil((w - crop_size) / stride) + 1)

    for i in range(h_steps):
        for j in range(w_steps):
            y1 = min(i * stride, h - crop_size)
            x1 = min(j * stride, w - crop_size)
            y2 = y1 + crop_size
            x2 = x1 + crop_size

            crop = img[:, :, y1:y2, x1:x2]
            seg_logits, _ = model(crop)
            seg_logits = F.interpolate(
                seg_logits, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False,
            )
            pred[:, :, y1:y2, x1:x2] += seg_logits
            count[:, :, y1:y2, x1:x2] += 1

    pred = pred / count
    pred = pred[:, :, :orig_h, :orig_w]
    return F.softmax(pred, dim=1)


@torch.no_grad()
def multiscale_flip_inference(
    model: torch.nn.Module,
    img: torch.Tensor,
    num_classes: int,
    scales: list[float],
    flip: bool = True,
    crop_size: int = 512,
    stride: int = 341,
) -> torch.Tensor:
    """Run slide inference at multiple scales with optional flip.

    Args:
        model: WeakCLIP model.
        img: (1, 3, H, W) normalized tensor.
        num_classes: Number of output classes.
        scales: List of scale factors (e.g. [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).
        flip: Whether to also run horizontally-flipped inference.
        crop_size: Slide window crop size.
        stride: Slide window stride.

    Returns:
        (C, H, W) averaged softmax probability map.
    """
    _, _, orig_h, orig_w = img.shape
    accum = torch.zeros(num_classes, orig_h, orig_w, device=img.device)
    n = 0

    for scale in scales:
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        scaled = F.interpolate(
            img, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        prob = slide_inference(model, scaled, num_classes, crop_size, stride)
        prob = F.interpolate(
            prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        accum += prob.squeeze(0)
        n += 1

        if flip:
            flipped = torch.flip(scaled, dims=[3])
            prob_f = slide_inference(model, flipped, num_classes, crop_size, stride)
            prob_f = torch.flip(prob_f, dims=[3])
            prob_f = F.interpolate(
                prob_f, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            accum += prob_f.squeeze(0)
            n += 1

    return accum / n


def _map_author_key(k: str) -> str:
    """Map author checkpoint keys to our model's state_dict key format."""
    if k.startswith("backbone.transformer."):
        return k.replace("backbone.transformer.", "backbone.")
    m = re.match(r"text_encoder\.transformer\.resblocks\.(\d+)\.(.*)", k)
    if m:
        return f"text_encoder.transformer.{m.group(1)}.{m.group(2)}"
    m = re.match(r"neck\.(lateral_convs|fpn_convs)\.(\d+)\.conv\.(.*)", k)
    if m:
        return f"neck.{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m = re.match(r"decode_head\.scale_heads\.(\d+)\.(\d+)\.conv\.(.*)", k)
    if m:
        return f"decode_head.scale_heads.{m.group(1)}.{int(m.group(2))*2}.{m.group(3)}"
    m = re.match(r"decode_head\.scale_heads\.(\d+)\.(\d+)\.bn\.(.*)", k)
    if m:
        return f"decode_head.scale_heads.{m.group(1)}.{int(m.group(2))*2+1}.{m.group(3)}"
    return k


def _load_author_checkpoint(
    ckpt_path: str, cfg: GenerateMasksConfig, class_names: tuple[str, ...]
) -> torch.nn.Module:
    """Load author's raw PyTorch checkpoint with key mapping."""
    from src.wsss.weakclip.model import WeakCLIP

    train_cfg = WeakCLIPTrainConfig(
        class_names_file=cfg.class_names_file,
        clip_pretrained=cfg.clip_pretrained,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        context_length=cfg.context_length,
        tau=cfg.tau,
    )
    model = build_weakclip_model(train_cfg, class_names)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    author_sd = ckpt["state_dict"]
    mapped_sd = {_map_author_key(k): v for k, v in author_sd.items()}
    mapped_sd["texts"] = model.texts
    model.load_state_dict(mapped_sd, strict=True)
    log.info("Loaded author checkpoint with strict key mapping")
    return model


def generate_weakclip_masks(cfg: GenerateMasksConfig) -> None:
    if not cfg.checkpoint:
        raise ValueError("checkpoint is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(cfg.class_names_file)
    if cfg.num_classes == 0:
        cfg.num_classes = len(class_names)

    log.info(f"Loading checkpoint from {cfg.checkpoint} (format={cfg.checkpoint_format})")
    if cfg.checkpoint_format == "author":
        model = _load_author_checkpoint(cfg.checkpoint, cfg, class_names).to(device).eval()
    else:
        lit_module = WeakCLIPModule.load_from_checkpoint(
            cfg.checkpoint, map_location="cpu",
            model=build_weakclip_model(
                WeakCLIPTrainConfig(
                    class_names_file=cfg.class_names_file,
                    clip_pretrained=cfg.clip_pretrained,
                    num_classes=cfg.num_classes,
                    image_size=cfg.image_size,
                    context_length=cfg.context_length,
                    tau=cfg.tau,
                ),
                class_names,
            ),
        )
        model = lit_module.model.to(device).eval()

    image_dir = Path(cfg.image_dir)
    if cfg.names_file:
        names = [
            l.strip()
            for l in Path(cfg.names_file).read_text().splitlines()
            if l.strip()
        ]
    else:
        names = sorted(f.stem for f in image_dir.glob(f"*{cfg.image_ext}"))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Generating masks for {len(names)} images, "
        f"scales={cfg.scales}, flip={cfg.flip}, "
        f"crop={cfg.crop_size}, stride={cfg.stride}"
    )

    for name in tqdm(names, desc="Generating masks"):
        img_path = image_dir / f"{name}{cfg.image_ext}"
        if not img_path.exists():
            log.warning(f"Image not found: {img_path}")
            continue

        _, img_tensor = _load_image(img_path)
        img_tensor = img_tensor.to(device)

        prob = multiscale_flip_inference(
            model, img_tensor, cfg.num_classes,
            scales=cfg.scales, flip=cfg.flip,
            crop_size=cfg.crop_size, stride=cfg.stride,
        )

        prob_np = prob.cpu().numpy()
        np.save(str(output_dir / f"{name}.npy"), prob_np)

    log.info(f"Saved {len(names)} probability maps to {output_dir}")


@hydra.main(version_base=None, config_name="generate_masks_config")
def main(cfg: DictConfig) -> None:
    generate_weakclip_masks(cfg)


if __name__ == "__main__":
    main()
