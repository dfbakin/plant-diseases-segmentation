"""DenseCRF post-processing for CAM refinement."""

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

NUM_CLS_VOC = 21


def apply_crf(
    image: np.ndarray,
    cam_dict: dict[int, np.ndarray],
    bg_threshold: float = 0.3,
    alpha: float = 4.0,
    t: int = 10,
    num_cls: int = NUM_CLS_VOC,
) -> np.ndarray:
    """Apply DenseCRF to refine raw CAMs into a segmentation mask.

    Args:
        image: RGB image (H, W, 3), uint8.
        cam_dict: {class_idx: (H, W) cam} from MCTformer (0-indexed, no bg).
        bg_threshold: Background confidence score.
        alpha: Bilateral spatial kernel weight (low=conservative, high=aggressive).
        t: Number of CRF inference iterations.
        num_cls: Total classes including background.

    Returns:
        Probability map (num_cls, H, W) float32, summing to 1 along axis 0.
    """
    h, w = image.shape[:2]
    probs = np.zeros((num_cls, h, w), dtype=np.float32)

    for cls_idx, cam in cam_dict.items():
        probs[cls_idx + 1] = cam
    probs[0] = bg_threshold

    d = dcrf.DenseCRF2D(w, h, num_cls)

    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=alpha, srgb=13, rgbim=np.ascontiguousarray(image), compat=10)

    q = d.inference(t)
    q = np.array(q).reshape((num_cls, h, w))
    return q


def cam_to_label(
    image: np.ndarray,
    cam_dict: dict[int, np.ndarray],
    bg_threshold: float = 0.3,
    alpha: float = 4.0,
    t: int = 10,
    num_cls: int = NUM_CLS_VOC,
) -> np.ndarray:
    """Apply CRF and return argmax label map.

    Returns:
        Label map (H, W) uint8, values in [0, num_cls-1].
    """
    q = apply_crf(image, cam_dict, bg_threshold, alpha, t, num_cls)
    return np.argmax(q, axis=0).astype(np.uint8)
