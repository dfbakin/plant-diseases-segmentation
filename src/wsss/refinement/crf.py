"""DenseCRF post-processing for CAM refinement."""

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

NUM_CLS_VOC = 21


def apply_crf(
    image: np.ndarray,
    cam_dict: dict[int, np.ndarray],
    bg_threshold: float = 0.3,
    alpha: float = 80.0,
    t: int = 10,
    num_cls: int = NUM_CLS_VOC,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Apply DenseCRF to refine raw CAMs into a segmentation mask.

    Uses the PSA parameterization where both Gaussian and Bilateral
    kernels are scaled by ``scale_factor``:
        Gaussian  sxy = 3 / scale_factor
        Bilateral sxy = 80 / scale_factor

    For backward compatibility, ``alpha`` can override the bilateral sxy
    directly (set ``scale_factor=0`` to use raw ``alpha``).

    Args:
        image: RGB image (H, W, 3), uint8.
        cam_dict: {class_idx: (H, W) cam} from MCTformer (0-indexed, no bg).
        bg_threshold: Background confidence score.
        alpha: Bilateral sxy (only used when scale_factor == 0).
        t: Number of CRF inference iterations.
        num_cls: Total classes including background.
        scale_factor: PSA scale factor; controls both Gaussian and
            Bilateral sxy.  Paper defaults: la=1, ha=12.

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

    if scale_factor > 0:
        gauss_sxy = 3.0 / scale_factor
        bilat_sxy = 80.0 / scale_factor
    else:
        gauss_sxy = 3.0
        bilat_sxy = alpha

    d.addPairwiseGaussian(sxy=gauss_sxy, compat=3)
    d.addPairwiseBilateral(
        sxy=bilat_sxy, srgb=13, rgbim=np.ascontiguousarray(image), compat=10,
    )

    q = d.inference(t)
    q = np.array(q).reshape((num_cls, h, w))
    return q


def cam_to_label(
    image: np.ndarray,
    cam_dict: dict[int, np.ndarray],
    bg_threshold: float = 0.3,
    alpha: float = 80.0,
    t: int = 10,
    num_cls: int = NUM_CLS_VOC,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Apply CRF and return argmax label map.

    Returns:
        Label map (H, W) uint8, values in [0, num_cls-1].
    """
    q = apply_crf(image, cam_dict, bg_threshold, alpha, t, num_cls, scale_factor)
    return np.argmax(q, axis=0).astype(np.uint8)
