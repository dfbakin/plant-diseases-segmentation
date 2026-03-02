"""Refine WeakCLIP probability maps into pseudo masks via CRF + label filtering.

Replicates WeakCLIP/tools/make_crf.py:
1. Load per-image softmax probability maps (.npy) from generate_weakclip_masks.py
2. Apply DenseCRF with the paper's VOC12 parameters
3. Filter predictions using image-level labels (paper eq. 14):
   classes not present in the image are reassigned to 255 (ignore)
4. Save final PNG pseudo masks

Image-level labels come from the exported labels file produced by export_labels.py
(a dict mapping image name -> binary vector of foreground class presence, e.g.
outputs/labels/voc_train_aug.npy).

Example:
    python src/refine_weakclip_masks.py \
        prob_dir=outputs/weakclip_probs \
        image_dir=data/VOC2012/JPEGImages \
        labels_file=outputs/labels/voc_train_aug.npy \
        output_dir=outputs/weakclip_masks
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

log = logging.getLogger(__name__)


@dataclass
class RefineMasksConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    prob_dir: str = "outputs/weakclip_probs"
    image_dir: str = "data/VOC2012/JPEGImages"
    image_ext: str = ".jpg"
    labels_file: str = "outputs/labels/voc_train_aug.npy"
    output_dir: str = "outputs/weakclip_masks"
    names_file: str = ""

    num_classes: int = 21
    crf_t: int = 10
    crf_sxy_gauss: float = 3.0
    crf_compat_gauss: float = 3.0
    crf_sxy_bilat: float = 83.0
    crf_srgb_bilat: float = 5.0
    crf_compat_bilat: float = 3.0

    n_jobs: int = 8


cs = ConfigStore.instance()
cs.store(name="refine_masks_config", node=RefineMasksConfig)


def crf_inference(
    img: np.ndarray,
    probs: np.ndarray,
    t: int = 10,
    sxy_gauss: float = 3.0,
    compat_gauss: float = 3.0,
    sxy_bilat: float = 83.0,
    srgb_bilat: float = 5.0,
    compat_bilat: float = 3.0,
) -> np.ndarray:
    """Run DenseCRF on a single image with configurable parameters.

    Matches WeakCLIP/tools/densecrf.py::crf_inference_voc12 active config.

    Args:
        img: (H, W, 3) uint8 RGB image at original resolution.
        probs: (C, H, W) float probability map (must match img spatial dims).
        t: Number of CRF inference iterations.

    Returns:
        (C, H, W) CRF-refined probability map.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    C, h, w = probs.shape
    probs = np.clip(probs, 1e-7, None)
    probs = probs / probs.sum(axis=0, keepdims=True)

    d = dcrf.DenseCRF2D(w, h, C)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(np.ascontiguousarray(unary))
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

    img_c = np.ascontiguousarray(img)
    d.addPairwiseBilateral(
        sxy=sxy_bilat, srgb=srgb_bilat, rgbim=img_c, compat=compat_bilat,
    )

    Q = d.inference(t)
    return np.array(Q).reshape(C, h, w)


def load_labels_dict(labels_file: str | Path) -> dict[str, np.ndarray] | None:
    """Load image-level multi-label classification annotations.

    The file is a dict mapping image name -> binary vector of length
    ``num_fg_classes``, produced by ``export_labels.py``.  Foreground index
    ``i`` in the vector corresponds to segmentation class ``i + 1`` (class 0
    is always background).
    """
    p = Path(labels_file)
    if not p.exists():
        return None
    return np.load(str(p), allow_pickle=True).item()


def get_valid_seg_classes(
    labels_dict: dict[str, np.ndarray] | None,
    name: str,
) -> np.ndarray | None:
    """Return the set of valid segmentation class indices for *name*.

    Converts the foreground-only binary vector (indices 0..N-1) into full
    segmentation class indices (1..N) and always includes background (0).
    """
    if labels_dict is None or name not in labels_dict:
        return None
    vec = np.asarray(labels_dict[name])
    fg_indices = np.where(vec > 0.5)[0]
    seg_classes = fg_indices + 1
    return np.concatenate([[0], seg_classes])


def filter_by_image_labels(
    pred: np.ndarray, valid_classes: np.ndarray | None,
) -> np.ndarray:
    """Paper eq. 14: mask out predicted classes not present in image labels.

    Args:
        pred: (H, W) integer prediction map (segmentation class indices).
        valid_classes: Array of valid segmentation class indices.
            If None, no filtering is applied.

    Returns:
        (H, W) filtered prediction, invalid classes set to 255.
    """
    if valid_classes is None:
        return pred
    mask = np.isin(pred, valid_classes)
    return np.where(mask, pred, 255).astype(np.uint8)


def process_single_image(
    name: str,
    prob_dir: Path,
    image_dir: Path,
    labels_dict: dict[str, np.ndarray] | None,
    output_dir: Path,
    image_ext: str,
    crf_t: int,
    crf_sxy_gauss: float,
    crf_compat_gauss: float,
    crf_sxy_bilat: float,
    crf_srgb_bilat: float,
    crf_compat_bilat: float,
) -> bool:
    """Process one image: load probs, run CRF, filter, save PNG."""
    prob_path = prob_dir / f"{name}.npy"
    img_path = image_dir / f"{name}{image_ext}"

    if not prob_path.exists():
        return False
    if not img_path.exists():
        return False

    probs = np.load(str(prob_path))
    if probs.ndim == 2:
        probs = probs[np.newaxis]

    img = np.array(Image.open(img_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]

    if probs.shape[1] != orig_h or probs.shape[2] != orig_w:
        import torch
        import torch.nn.functional as F_t

        probs_t = torch.from_numpy(probs).unsqueeze(0).float()
        probs_t = F_t.interpolate(
            probs_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False,
        )
        probs = probs_t.squeeze(0).numpy()
        probs = np.clip(probs, 1e-7, None)
        probs = probs / probs.sum(axis=0, keepdims=True)

    crf_probs = crf_inference(
        img, probs, t=crf_t,
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


def refine_weakclip_masks(cfg: RefineMasksConfig) -> None:
    prob_dir = Path(cfg.prob_dir)
    image_dir = Path(cfg.image_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_dict = load_labels_dict(cfg.labels_file)
    if labels_dict is not None:
        log.info(
            f"Loaded image-level labels for {len(labels_dict)} images "
            f"from {cfg.labels_file}"
        )
    else:
        log.warning(
            f"Labels file not found at {cfg.labels_file}, "
            f"label filtering will be disabled"
        )

    if cfg.names_file:
        names = [
            line.strip()
            for line in Path(cfg.names_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        names = sorted(f.stem for f in prob_dir.glob("*.npy"))

    log.info(
        f"Refining {len(names)} masks: CRF(t={cfg.crf_t}, "
        f"gauss_sxy={cfg.crf_sxy_gauss}, bilat_sxy={cfg.crf_sxy_bilat}, "
        f"bilat_srgb={cfg.crf_srgb_bilat}) + label filtering"
    )

    if cfg.n_jobs > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(process_single_image)(
                name, prob_dir, image_dir, labels_dict, output_dir, cfg.image_ext,
                cfg.crf_t,
                cfg.crf_sxy_gauss, cfg.crf_compat_gauss,
                cfg.crf_sxy_bilat, cfg.crf_srgb_bilat, cfg.crf_compat_bilat,
            )
            for name in names
        )
        success = sum(results)
    else:
        success = 0
        for name in tqdm(names, desc="Refining masks"):
            ok = process_single_image(
                name, prob_dir, image_dir, labels_dict, output_dir, cfg.image_ext,
                cfg.crf_t,
                cfg.crf_sxy_gauss, cfg.crf_compat_gauss,
                cfg.crf_sxy_bilat, cfg.crf_srgb_bilat, cfg.crf_compat_bilat,
            )
            if ok:
                success += 1

    log.info(f"Refined {success}/{len(names)} masks, saved to {output_dir}")


@hydra.main(version_base=None, config_name="refine_masks_config")
def main(cfg: DictConfig) -> None:
    refine_weakclip_masks(cfg)


if __name__ == "__main__":
    main()
