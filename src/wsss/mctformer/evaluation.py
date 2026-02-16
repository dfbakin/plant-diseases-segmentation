"""CAM evaluation: mIoU computation for .npy CAM files against GT masks.

Ported from MCTformer's evaluation.py with cleaner interface and type hints.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

VOC_CATEGORIES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def evaluate_cam_miou(
    predict_dir: str | Path,
    gt_dir: str | Path,
    name_list: list[str],
    num_cls: int = 21,
    input_type: str = "npy",
    threshold: float = 1.0,
    categories: list[str] | None = None,
) -> dict[str, float]:
    """Compute mIoU for predicted CAMs/masks against ground truth.

    Args:
        predict_dir: Directory containing prediction files (.npy or .png)
        gt_dir: Directory containing GT segmentation masks (.png)
        name_list: List of image names (without extension)
        num_cls: Number of classes including background
        input_type: 'npy' for CAM .npy files, 'png' for mask PNGs
        threshold: Background threshold for npy predictions
        categories: Class names for logging (defaults to VOC)

    Returns:
        Dictionary with per-class IoU and overall mIoU (all in percentage)
    """
    if categories is None:
        categories = VOC_CATEGORIES

    predict_dir = Path(predict_dir)
    gt_dir = Path(gt_dir)

    TP = np.zeros(num_cls, dtype=np.int64)
    P = np.zeros(num_cls, dtype=np.int64)
    T = np.zeros(num_cls, dtype=np.int64)

    for name in name_list:
        if input_type == "png":
            predict_file = predict_dir / f"{name}.png"
            predict = np.array(Image.open(predict_file))
        elif input_type == "npy":
            predict_file = predict_dir / f"{name}.npy"
            predict_dict = np.load(str(predict_file), allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((num_cls, h, w), np.float32)
            for key in predict_dict.keys():
                tensor[key + 1] = predict_dict[key]
            tensor[0, :, :] = threshold
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        gt_file = gt_dir / f"{name}.png"
        gt = np.array(Image.open(gt_file))
        cal = gt < 255  # Ignore void pixels

        mask = (predict == gt) * cal

        for i in range(num_cls):
            P[i] += np.sum((predict == i) * cal)
            T[i] += np.sum((gt == i) * cal)
            TP[i] += np.sum((gt == i) * mask)

    IoU = TP / (T + P - TP + 1e-10)
    result = {}
    for i in range(num_cls):
        cat_name = categories[i] if i < len(categories) else f"class_{i}"
        result[cat_name] = float(IoU[i] * 100)

    result["mIoU"] = float(np.mean(IoU) * 100)
    return result


def evaluate_cam_threshold_sweep(
    predict_dir: str | Path,
    gt_dir: str | Path,
    name_list: list[str],
    num_cls: int = 21,
    start: int = 0,
    end: int = 60,
) -> dict[str, float]:
    """Sweep background thresholds to find best mIoU for CAM predictions.

    Args:
        predict_dir: Directory with .npy CAM files
        gt_dir: Directory with GT masks
        name_list: Image name list
        num_cls: Number of classes including background
        start: Start of threshold range (threshold = start/100)
        end: End of threshold range

    Returns:
        Dict with 'best_miou', 'best_threshold', and 'miou_curve'
    """
    best_miou = 0.0
    best_thr = 0.0
    miou_curve = []

    for i in range(start, end):
        t = i / 100.0
        result = evaluate_cam_miou(
            predict_dir, gt_dir, name_list, num_cls, input_type="npy", threshold=t
        )
        miou = result["mIoU"]
        miou_curve.append(miou)
        log.info(f"threshold={t:.2f}  mIoU={miou:.3f}%")

        if miou > best_miou:
            best_miou = miou
            best_thr = t
        else:
            # Early stop: mIoU started decreasing
            break

    log.info(f"Best threshold={best_thr:.2f}  mIoU={best_miou:.3f}%")
    return {
        "best_miou": best_miou,
        "best_threshold": best_thr,
        "miou_curve": miou_curve,
    }
