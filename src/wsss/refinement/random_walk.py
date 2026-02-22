"""Random walk refinement using trained PSA affinity network.

Given raw CAMs and a trained AffinityNet, build a transition matrix from
predicted pixel affinities and propagate CAMs via repeated matrix multiplication.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.wsss.refinement.affinity_net import AffinityNet


@torch.no_grad()
def random_walk_refine(
    model: AffinityNet,
    image: torch.Tensor,
    cam_dict: dict[int, np.ndarray],
    bg_threshold: float = 0.3,
    beta: int = 8,
    logt: int = 6,
    num_cls: int = 21,
    device: torch.device | None = None,
) -> np.ndarray:
    """Refine CAMs via random walk on affinity graph.

    Args:
        model: Trained AffinityNet.
        image: Preprocessed image tensor (1, 3, H, W).
        cam_dict: {class_idx: (H, W)} raw CAMs (0-indexed, no bg).
        bg_threshold: Background confidence.
        beta: Affinity threshold power (higher = sparser graph).
        logt: Number of random walk iterations (log2 of matrix powers).
        num_cls: Total classes including background.
        device: Torch device.

    Returns:
        Label map (H, W) uint8 with class indices [0, num_cls-1].
    """
    if device is None:
        device = next(model.parameters()).device

    orig_h, orig_w = image.shape[2], image.shape[3]
    image = image.to(device)

    # Get dense affinity matrix
    aff_mat = model(image, to_dense=True)  # (H_feat*W_feat, H_feat*W_feat)

    # Apply beta thresholding: raise affinities to power of beta, zero out weak ones
    aff_mat = aff_mat**beta
    aff_mat = aff_mat / (torch.sum(aff_mat, dim=0, keepdim=True) + 1e-5)

    # Repeated matrix multiplication (logt rounds = 2^logt power)
    for _ in range(logt):
        aff_mat = torch.matmul(aff_mat, aff_mat)

    # Build CAM probability volume
    h_feat = orig_h // 8
    w_feat = orig_w // 8

    cam_full = np.zeros((num_cls, orig_h, orig_w), dtype=np.float32)
    for cls_idx, cam in cam_dict.items():
        cam_full[cls_idx + 1] = cam
    cam_full[0] = bg_threshold

    # Downscale to feature map size
    cam_t = torch.from_numpy(cam_full).unsqueeze(0).to(device)
    cam_ds = F.interpolate(cam_t, size=(h_feat, w_feat), mode="bilinear", align_corners=False)
    cam_ds = cam_ds.squeeze(0)  # (num_cls, h_feat, w_feat)
    cam_flat = cam_ds.view(num_cls, -1)  # (num_cls, h_feat*w_feat)

    # Random walk: propagate
    cam_rw = torch.matmul(cam_flat, aff_mat)  # (num_cls, h_feat*w_feat)
    cam_rw = cam_rw.view(1, num_cls, h_feat, w_feat)

    # Upsample back to original resolution
    cam_rw = F.interpolate(cam_rw, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    cam_rw = cam_rw.squeeze(0).cpu().numpy()  # (num_cls, H, W)

    return np.argmax(cam_rw, axis=0).astype(np.uint8)
