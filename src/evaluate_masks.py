"""Evaluate pseudo masks against ground truth (mIoU).

Dataset-agnostic: compares any prediction mask directory against GT masks.
Derives image list from pred_dir glob. Computes per-class IoU and mean IoU.

Example:
    python src/evaluate_masks.py pred_dir=outputs/pseudo_masks gt_dir=data/VOC2012/SegmentationClassAug
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

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class EvalConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    pred_dir: str = "outputs/pseudo_masks"
    gt_dir: str = "data/VOC2012/SegmentationClassAug"
    num_cls: int = 21
    class_names_file: str = ""


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)


def compute_miou(
    pred_dir: str | Path,
    gt_dir: str | Path,
    name_list: list[str],
    num_cls: int = 21,
) -> dict:
    """Compute per-class IoU and mIoU."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    intersection = np.zeros(num_cls, dtype=np.int64)
    union = np.zeros(num_cls, dtype=np.int64)

    for name in tqdm(name_list, desc="Evaluating"):
        name = name.strip()
        pred_path = pred_dir / f"{name}.png"
        if not pred_path.exists():
            pred_path = pred_dir / f"{name}.npy"
        gt_path = gt_dir / f"{name}.png"

        if not pred_path.exists() or not gt_path.exists():
            continue

        if pred_path.suffix == ".npy":
            pred = np.load(str(pred_path))
        else:
            pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))

        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST
                )
            )

        # Ignore 255 (void) in GT
        valid = gt != 255
        pred = pred[valid]
        gt = gt[valid]

        for c in range(num_cls):
            pred_c = pred == c
            gt_c = gt == c
            intersection[c] += np.sum(pred_c & gt_c)
            union[c] += np.sum(pred_c | gt_c)

    iou_per_class = np.zeros(num_cls, dtype=np.float64)
    for c in range(num_cls):
        if union[c] > 0:
            iou_per_class[c] = intersection[c] / union[c]
        else:
            iou_per_class[c] = float("nan")

    valid_classes = ~np.isnan(iou_per_class)
    miou = np.nanmean(iou_per_class)

    return {
        "miou": float(miou) * 100,
        "iou_per_class": iou_per_class,
        "valid_classes": valid_classes,
    }


def _load_class_names(path: str) -> list[str]:
    """Load class names from file (one per line).

    The file is expected to list *foreground* class names.  This function
    prepends "background" so that index 0 maps to "background".
    """
    names = Path(path).read_text().strip().splitlines()
    return ["background"] + [n.strip() for n in names]


def evaluate_masks(cfg: EvalConfig) -> None:
    pred_dir = Path(cfg.pred_dir)
    png_names = {f.stem for f in pred_dir.glob("*.png")}
    npy_names = {f.stem for f in pred_dir.glob("*.npy")}
    names = sorted(png_names | npy_names)
    log.info(f"Evaluating {len(names)} masks: {cfg.pred_dir} vs {cfg.gt_dir}")

    if cfg.class_names_file:
        class_names = _load_class_names(cfg.class_names_file)
        if cfg.num_cls != len(class_names):
            log.info(
                f"Overriding num_cls from class_names_file: "
                f"{cfg.num_cls} -> {len(class_names)}"
            )
            cfg.num_cls = len(class_names)
    else:
        class_names = VOC_CLASSES

    result = compute_miou(cfg.pred_dir, cfg.gt_dir, names, cfg.num_cls)

    log.info(f"\nmIoU: {result['miou']:.2f}%")
    log.info("Per-class IoU:")
    for c in range(cfg.num_cls):
        if result["valid_classes"][c]:
            cname = class_names[c] if c < len(class_names) else f"class_{c}"
            log.info(f"  {cname:40s}: {result['iou_per_class'][c] * 100:.2f}%")


@hydra.main(version_base=None, config_name="eval_config")
def main(cfg: DictConfig) -> None:
    evaluate_masks(cfg)


if __name__ == "__main__":
    main()
