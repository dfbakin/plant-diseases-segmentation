"""CAM evaluation: mIoU computation for CAM/mask files against GT masks."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

VOC_CATEGORIES = [
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

        if predict.shape != gt.shape:
            predict = np.array(
                Image.fromarray(predict).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )

        cal = gt < 255
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


def _resolve_optimize_key(
    optimize_metric: str,
    result: dict[str, float],
    num_cls: int,
) -> str:
    """Map ``optimize_metric`` to the actual key in an ``evaluate_cam_miou`` result dict.

    Accepted values:
        ``"mIoU"``        – overall mean IoU (default, legacy behaviour)
        ``"disease_iou"`` – foreground IoU (last class when num_cls==2, else raises)
        any key present in *result* (e.g. a category name like ``"disease"``)
    """
    if optimize_metric == "mIoU":
        return "mIoU"
    if optimize_metric == "disease_iou":
        fg_keys = [k for k in result if k not in ("mIoU", "background")]
        if len(fg_keys) == 1:
            return fg_keys[0]
        if num_cls == 2:
            return fg_keys[0] if fg_keys else "mIoU"
        raise ValueError(
            f"optimize_metric='disease_iou' is ambiguous for num_cls={num_cls} "
            f"(foreground keys: {fg_keys}). Pass an explicit class name instead."
        )
    if optimize_metric in result:
        return optimize_metric
    raise ValueError(
        f"optimize_metric={optimize_metric!r} not found in evaluation result keys: "
        f"{list(result.keys())}"
    )


def evaluate_cam_threshold_sweep(
    predict_dir: str | Path,
    gt_dir: str | Path,
    name_list: list[str],
    num_cls: int = 21,
    start: int = 0,
    end: int = 100,
    max_samples: int = 0,
    seed: int = 42,
    optimize_metric: str = "mIoU",
    patience: int = 0,
) -> dict[str, float | list]:
    """Sweep background thresholds and find the best one for a chosen metric.

    Args:
        predict_dir: Directory with .npy CAM files.
        gt_dir: Directory with GT masks.
        name_list: Image name list.
        num_cls: Number of classes including background.
        start: Start of threshold range (threshold = start/100).
        end: End of threshold range.
        max_samples: When > 0, randomly subsample name_list before sweep.
        seed: RNG seed for reproducible subsampling.
        optimize_metric: Which metric to maximize.
            ``"mIoU"`` (default), ``"disease_iou"`` (foreground class for
            binary), or any per-class name returned by ``evaluate_cam_miou``.
        patience: Stop after *patience* consecutive non-improving thresholds.
            0 means run the full sweep (recommended).

    Returns:
        Dict with ``best_<metric>``, ``best_threshold``, per-threshold curves
        for mIoU and per-class IoUs, and full ``result_at_best`` with all
        per-class IoUs at the selected threshold.
    """
    if max_samples > 0 and len(name_list) > max_samples:
        rng = np.random.default_rng(seed)
        total = len(name_list)
        name_list = list(rng.choice(name_list, max_samples, replace=False))
        log.info(f"Subsampled {max_samples}/{total} images for threshold sweep")

    from tqdm import trange

    best_score = -1.0
    best_thr = 0.0
    best_result: dict[str, float] = {}
    no_improve_count = 0

    curves: dict[str, list[float]] = {"threshold": [], "mIoU": []}
    opt_key: str | None = None

    for i in trange(start, end, desc="Threshold sweep", unit="thr"):
        t = i / 100.0
        result = evaluate_cam_miou(
            predict_dir, gt_dir, name_list, num_cls, input_type="npy", threshold=t
        )

        if opt_key is None:
            opt_key = _resolve_optimize_key(optimize_metric, result, num_cls)
            for k in result:
                if k not in curves:
                    curves[k] = []
            log.info(f"Optimizing threshold for: {opt_key}")

        curves["threshold"].append(t)
        for k in result:
            curves.setdefault(k, []).append(result[k])

        score = result[opt_key]
        if score > best_score:
            best_score = score
            best_thr = t
            best_result = result
            no_improve_count = 0
        else:
            no_improve_count += 1
            if patience > 0 and no_improve_count >= patience:
                log.info(f"Early stop at threshold={t:.2f} (patience={patience})")
                break

    log.info(
        f"Best threshold={best_thr:.2f}  {opt_key}={best_score:.2f}%  "
        f"mIoU={best_result.get('mIoU', 0.0):.2f}%"
    )
    return {
        "best_threshold": best_thr,
        f"best_{opt_key}": best_score,
        "best_miou": best_result.get("mIoU", 0.0),
        "result_at_best": best_result,
        "curves": curves,
        "optimize_metric": opt_key,
    }
