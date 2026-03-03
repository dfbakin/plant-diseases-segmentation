"""Apply DenseCRF to raw CAMs at two alpha levels (la/ha) for PSA training.

Reads .npy CAM dicts, applies CRF, writes refined .npy dicts.
Uses multiprocessing for speed (CRF is CPU-bound).

Example:
    python src/apply_crf.py cam_dir=outputs/cams/cam_npy num_workers=8
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
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
    la_scale_factor: float = 1.0
    ha_scale_factor: float = 12.0
    crf_iters: int = 10
    num_cls: int = 21
    num_workers: int = 8


cs = ConfigStore.instance()
cs.store(name="crf_config", node=CRFConfig)


def _process_one(
    name: str,
    cam_dir: str,
    image_dir: str,
    image_ext: str,
    la_dir: str,
    ha_dir: str,
    bg_threshold: float,
    la_scale_factor: float,
    ha_scale_factor: float,
    crf_iters: int,
    num_cls: int,
) -> str:
    la_path = Path(la_dir) / f"{name}.npy"
    ha_path = Path(ha_dir) / f"{name}.npy"
    if la_path.exists() and ha_path.exists():
        return name

    cam_dict = np.load(str(Path(cam_dir) / f"{name}.npy"), allow_pickle=True).item()
    img = np.array(Image.open(Path(image_dir) / f"{name}{image_ext}").convert("RGB"))

    la_probs = apply_crf(
        img, cam_dict, bg_threshold, t=crf_iters, num_cls=num_cls,
        scale_factor=la_scale_factor,
    )
    ha_probs = apply_crf(
        img, cam_dict, bg_threshold, t=crf_iters, num_cls=num_cls,
        scale_factor=ha_scale_factor,
    )

    np.save(str(la_path), np.argmax(la_probs, axis=0).astype(np.uint8))
    np.save(str(ha_path), np.argmax(ha_probs, axis=0).astype(np.uint8))
    return name


def run_crf(cfg: CRFConfig) -> None:
    cam_dir = Path(cfg.cam_dir)
    la_dir = Path(cfg.la_crf_dir)
    ha_dir = Path(cfg.ha_crf_dir)
    la_dir.mkdir(parents=True, exist_ok=True)
    ha_dir.mkdir(parents=True, exist_ok=True)

    cam_files = sorted(cam_dir.glob("*.npy"))
    names = [f.stem for f in cam_files]
    log.info(
        f"Applying CRF to {len(names)} images "
        f"(la_sf={cfg.la_scale_factor}, ha_sf={cfg.ha_scale_factor}, "
        f"iters={cfg.crf_iters}, workers={cfg.num_workers})"
    )

    worker_fn = partial(
        _process_one,
        cam_dir=str(cam_dir),
        image_dir=cfg.image_dir,
        image_ext=cfg.image_ext,
        la_dir=str(la_dir),
        ha_dir=str(ha_dir),
        bg_threshold=cfg.bg_threshold,
        la_scale_factor=cfg.la_scale_factor,
        ha_scale_factor=cfg.ha_scale_factor,
        crf_iters=cfg.crf_iters,
        num_cls=cfg.num_cls,
    )

    if cfg.num_workers > 1:
        with Pool(cfg.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(worker_fn, names), total=len(names), desc="CRF"):
                pass
    else:
        for name in tqdm(names, desc="CRF"):
            worker_fn(name)

    log.info(f"Done. la_crf -> {la_dir}, ha_crf -> {ha_dir}")


@hydra.main(version_base=None, config_name="crf_config")
def main(cfg: DictConfig) -> None:
    run_crf(cfg)


if __name__ == "__main__":
    main()
