"""Apply random walk refinement to produce final pseudo masks.

Reads raw CAMs + trained AffinityNet, outputs .png pseudo masks.
Dataset-agnostic: derives image list from cam_dir glob.

Example:
    python src/run_random_walk.py cam_dir=outputs/cams/cam_npy
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import PIL.Image
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

from src.wsss.refinement.affinity_net import AffinityNet
from src.wsss.refinement.random_walk import random_walk_refine

log = logging.getLogger(__name__)


@dataclass
class RWConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    cam_dir: str = "outputs/cams/cam_npy"
    image_dir: str = "data/VOC2012/JPEGImages"
    image_ext: str = ".jpg"
    aff_checkpoint: str = "outputs/psa/psa_aff.pth"
    output_dir: str = "outputs/pseudo_masks"

    bg_threshold: float = 0.3
    beta: int = 8
    logt: int = 6
    num_cls: int = 21
    cropsize: int = 448
    max_size: int = 0


cs = ConfigStore.instance()
cs.store(name="rw_config", node=RWConfig)


def build_rw_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def run_random_walk(cfg: RWConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_size = cfg.cropsize // 8
    model = AffinityNet(predefined_featuresize=feature_size)
    sd = torch.load(cfg.aff_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    log.info(f"Loaded AffinityNet from {cfg.aff_checkpoint}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tfm = build_rw_transform()

    cam_dir = Path(cfg.cam_dir)
    image_dir = Path(cfg.image_dir)
    names = sorted(f.stem for f in cam_dir.glob("*.npy"))
    log.info(f"Running random walk for {len(names)} images (beta={cfg.beta}, logt={cfg.logt})")

    max_long = cfg.max_size if cfg.max_size > 0 else cfg.cropsize
    log.info(f"Max long side for random walk: {max_long}")

    for name in tqdm(names, desc="Random Walk"):
        cam_dict = np.load(str(cam_dir / f"{name}.npy"), allow_pickle=True).item()
        img_pil = PIL.Image.open(image_dir / f"{name}{cfg.image_ext}").convert("RGB")
        orig_w, orig_h = img_pil.size

        sample_cam = next(iter(cam_dict.values()))
        cam_h, cam_w = sample_cam.shape

        proc_w, proc_h = cam_w, cam_h
        long_side = max(proc_w, proc_h)
        if long_side > max_long:
            ratio = max_long / long_side
            proc_w = round(proc_w * ratio)
            proc_h = round(proc_h * ratio)

        proc_w = max((proc_w // 8) * 8, 80)
        proc_h = max((proc_h // 8) * 8, 80)

        img_resized = img_pil.resize((proc_w, proc_h), resample=PIL.Image.BICUBIC)
        img_t = tfm(img_resized).unsqueeze(0)

        resized_cam_dict = {}
        for cls_idx, cam in cam_dict.items():
            cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
            cam_resized = torch.nn.functional.interpolate(
                cam_t, size=(proc_h, proc_w), mode="bilinear", align_corners=False
            )
            resized_cam_dict[cls_idx] = cam_resized.squeeze().numpy()

        label = random_walk_refine(
            model,
            img_t,
            resized_cam_dict,
            bg_threshold=cfg.bg_threshold,
            beta=cfg.beta,
            logt=cfg.logt,
            num_cls=cfg.num_cls,
            device=device,
        )

        label_pil = PIL.Image.fromarray(label).resize(
            (orig_w, orig_h), resample=PIL.Image.NEAREST
        )
        label_pil.save(str(output_dir / f"{name}.png"))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info(f"Saved pseudo masks to {output_dir}")


@hydra.main(version_base=None, config_name="rw_config")
def main(cfg: DictConfig) -> None:
    run_random_walk(cfg)


if __name__ == "__main__":
    main()
