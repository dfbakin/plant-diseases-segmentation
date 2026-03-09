"""Generate and refine WeakCLIP pseudo masks in a single streaming pass.

Merges generate_weakclip_masks.py + refine_weakclip_masks.py to avoid
writing intermediate probability maps to disk (which would be ~1 TB for
100+ class datasets).

Pipeline per image:
    load image -> multi-scale/flip slide inference -> (C,H,W) probs
    -> DenseCRF refinement -> argmax -> image-level label filtering
    -> save uint8 PNG mask

Example:
    python src/generate_refine_weakclip_masks.py \
        checkpoint=outputs/weakclip/checkpoints/last.ckpt \
        class_names_file=outputs/labels/class_names.txt \
        image_dir=data/plantsegv3/images/train \
        labels_file=outputs/labels/plantseg_wsss_train.npy \
        output_dir=outputs/plantseg_wsss/weakclip_masks
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.generate_weakclip_masks import (
    _load_image,
    multiscale_flip_inference,
)
from src.refine_weakclip_masks import (
    crf_inference,
    filter_by_image_labels,
    get_valid_seg_classes,
    load_labels_dict,
)
from src.train_weakclip import WeakCLIPTrainConfig, build_weakclip_model, load_class_names
from src.wsss.weakclip.lightning import WeakCLIPModule

log = logging.getLogger(__name__)


@dataclass
class StreamMasksConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    checkpoint: str = ""
    checkpoint_format: str = "lightning"
    class_names_file: str = "outputs/labels/class_names.txt"
    image_dir: str = ""
    image_ext: str = ".jpg"
    names_file: str = ""
    labels_file: str = ""
    output_dir: str = "outputs/weakclip_masks"

    # Inference params
    scales: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )
    flip: bool = True
    crop_size: int = 512
    stride: int = 341

    # Model params
    clip_pretrained: str = "pretrained/ViT-B-16.pt"
    image_size: int = 512
    context_length: int = 5
    tau: float = 0.07
    num_classes: int = 0

    # CRF params
    crf_t: int = 10
    crf_sxy_gauss: float = 3.0
    crf_compat_gauss: float = 3.0
    crf_sxy_bilat: float = 83.0
    crf_srgb_bilat: float = 5.0
    crf_compat_bilat: float = 3.0


cs = ConfigStore.instance()
cs.store(name="stream_masks_config", node=StreamMasksConfig)


def _process_single(
    name: str,
    model: torch.nn.Module,
    device: torch.device,
    image_dir: Path,
    image_ext: str,
    labels_dict: dict[str, np.ndarray] | None,
    output_dir: Path,
    num_classes: int,
    scales: list[float],
    flip: bool,
    crop_size: int,
    stride: int,
    crf_t: int,
    crf_sxy_gauss: float,
    crf_compat_gauss: float,
    crf_sxy_bilat: float,
    crf_srgb_bilat: float,
    crf_compat_bilat: float,
) -> bool:
    """Generate probs, apply CRF, filter, save PNG for one image."""
    img_path = image_dir / f"{name}{image_ext}"
    if not img_path.exists():
        log.warning(f"Image not found: {img_path}")
        return False

    img_np, img_tensor = _load_image(img_path)
    img_tensor = img_tensor.to(device)

    prob = multiscale_flip_inference(
        model, img_tensor, num_classes,
        scales=scales, flip=flip,
        crop_size=crop_size, stride=stride,
    )
    probs = prob.cpu().numpy()  # (C, H, W)

    orig_h, orig_w = img_np.shape[:2]
    if probs.shape[1] != orig_h or probs.shape[2] != orig_w:
        import torch.nn.functional as F_t

        probs_t = torch.from_numpy(probs).unsqueeze(0).float()
        probs_t = F_t.interpolate(
            probs_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False,
        )
        probs = probs_t.squeeze(0).numpy()
        probs = np.clip(probs, 1e-7, None)
        probs = probs / probs.sum(axis=0, keepdims=True)

    crf_probs = crf_inference(
        img_np, probs, t=crf_t,
        sxy_gauss=crf_sxy_gauss, compat_gauss=crf_compat_gauss,
        sxy_bilat=crf_sxy_bilat, srgb_bilat=crf_srgb_bilat,
        compat_bilat=crf_compat_bilat,
    )

    pred = np.argmax(crf_probs, axis=0)

    valid_classes = get_valid_seg_classes(labels_dict, name)
    pred = filter_by_image_labels(pred, valid_classes)

    out_path = output_dir / f"{name}.png"
    Image.fromarray(pred.astype(np.uint8)).save(str(out_path))
    return True


@torch.no_grad()
def generate_refine_masks(cfg: StreamMasksConfig) -> None:
    if not cfg.checkpoint:
        raise ValueError("checkpoint is required")
    if not cfg.image_dir:
        raise ValueError("image_dir is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(cfg.class_names_file)
    if cfg.num_classes == 0:
        cfg.num_classes = len(class_names) + 1

    log.info(f"Loading checkpoint from {cfg.checkpoint} (format={cfg.checkpoint_format})")
    if cfg.checkpoint_format == "author":
        from src.generate_weakclip_masks import _load_author_checkpoint

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

    labels_dict = load_labels_dict(cfg.labels_file) if cfg.labels_file else None
    if labels_dict is not None:
        log.info(f"Loaded image-level labels for {len(labels_dict)} images")
    else:
        log.info("No labels file; label filtering disabled")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Streaming generate+refine for {len(names)} images, "
        f"num_classes={cfg.num_classes}, scales={cfg.scales}, flip={cfg.flip}"
    )

    success = 0
    for name in tqdm(names, desc="Generate+Refine"):
        ok = _process_single(
            name, model, device, image_dir, cfg.image_ext,
            labels_dict, output_dir, cfg.num_classes,
            cfg.scales, cfg.flip, cfg.crop_size, cfg.stride,
            cfg.crf_t, cfg.crf_sxy_gauss, cfg.crf_compat_gauss,
            cfg.crf_sxy_bilat, cfg.crf_srgb_bilat, cfg.crf_compat_bilat,
        )
        if ok:
            success += 1

    log.info(f"Saved {success}/{len(names)} refined masks to {output_dir}")


@hydra.main(version_base=None, config_name="stream_masks_config")
def main(cfg: DictConfig) -> None:
    generate_refine_masks(cfg)


if __name__ == "__main__":
    main()
