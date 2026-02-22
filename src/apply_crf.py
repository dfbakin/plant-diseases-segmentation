"""Apply DenseCRF to raw CAMs at two alpha levels (la/ha) for PSA training.

Reads .npy CAM dicts, applies CRF, writes refined .npy dicts.
Model-agnostic: works with any CAM source.

Example:
    python src/apply_crf.py cam_dir=outputs/cams/cam_npy
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.wsss.refinement.crf import apply_crf

log = logging.getLogger(__name__)


@dataclass
class CRFConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    cam_dir: str = "outputs/cams/cam_npy"
    image_dir: str = "data/VOC2012/JPEGImages"
    image_ext: str = ".jpg"

    la_crf_dir: str = "outputs/cams/la_crf"
    ha_crf_dir: str = "outputs/cams/ha_crf"

    bg_threshold: float = 0.3
    la_alpha: float = 4.0
    ha_alpha: float = 32.0
    crf_iters: int = 10
    num_cls: int = 21


cs = ConfigStore.instance()
cs.store(name="crf_config", node=CRFConfig)


def run_crf(cfg: CRFConfig) -> None:
    cam_dir = Path(cfg.cam_dir)
    image_dir = Path(cfg.image_dir)
    la_dir = Path(cfg.la_crf_dir)
    ha_dir = Path(cfg.ha_crf_dir)
    la_dir.mkdir(parents=True, exist_ok=True)
    ha_dir.mkdir(parents=True, exist_ok=True)

    cam_files = sorted(cam_dir.glob("*.npy"))
    names = [f.stem for f in cam_files]
    log.info(
        f"Applying CRF to {len(names)} images (la_alpha={cfg.la_alpha}, ha_alpha={cfg.ha_alpha})"
    )

    for name in tqdm(names, desc="CRF"):
        cam_dict = np.load(str(cam_dir / f"{name}.npy"), allow_pickle=True).item()
        img = np.array(Image.open(image_dir / f"{name}{cfg.image_ext}").convert("RGB"))

        la_probs = apply_crf(
            img, cam_dict, cfg.bg_threshold, cfg.la_alpha, cfg.crf_iters, cfg.num_cls
        )
        ha_probs = apply_crf(
            img, cam_dict, cfg.bg_threshold, cfg.ha_alpha, cfg.crf_iters, cfg.num_cls
        )

        np.save(str(la_dir / f"{name}.npy"), la_probs)
        np.save(str(ha_dir / f"{name}.npy"), ha_probs)

    log.info(f"Done. la_crf -> {la_dir}, ha_crf -> {ha_dir}")


@hydra.main(version_base=None, config_name="crf_config")
def main(cfg: DictConfig) -> None:
    run_crf(cfg)


if __name__ == "__main__":
    main()
