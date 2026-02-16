"""WeakCLIP losses: balanced seeding loss and CRF boundary loss.

Extracted from WeakCLIP/weakclip/dgcnutils.py.
Only the functions used in the WeakCLIP training loop (dgcn_lite mode).
"""

import numpy as np
import torch


def seeding_loss(pred: torch.Tensor, cues: torch.Tensor) -> torch.Tensor:
    """Balanced seeding loss: separate bg/fg averaging to handle class imbalance."""
    device = pred.device
    pred_bg = pred[:, 0, :, :]
    labels_bg = cues[:, 0, :, :].float().to(device)
    pred_fg = pred[:, 1:, :, :]
    labels_fg = cues[:, 1:, :, :].float().to(device)

    eps = torch.tensor(0.0001, device=device)

    count_bg = labels_bg.sum(dim=(1, 2), keepdim=True)
    count_fg = labels_fg.sum(dim=(1, 2, 3), keepdim=True)

    log_bg = torch.log(pred_bg.clamp(min=1e-7))
    log_fg = torch.log(pred_fg.clamp(min=1e-7))
    sum_bg = (labels_bg * log_bg).sum(dim=(1, 2), keepdim=True)
    sum_fg = (labels_fg * log_fg).sum(dim=(1, 2, 3), keepdim=True)

    loss_bg = -(sum_bg / torch.max(count_bg, eps)).mean()
    loss_fg = -(sum_fg / torch.max(count_fg, eps)).mean()
    return loss_bg + loss_fg


def crf_boundary_loss(probs: torch.Tensor, crf_result: np.ndarray) -> torch.Tensor:
    """KL-divergence boundary loss between model probs and CRF-smoothed probs."""
    probs_smooth = torch.exp(torch.from_numpy(crf_result)).float().to(probs.device)
    ratio = probs_smooth / probs
    ratio = ratio.clamp(0.05, 20.0)
    loss = torch.mean(torch.sum(probs_smooth * torch.log(ratio), dim=1))
    return loss


def stable_softmax(logits: torch.Tensor, min_prob: float = 1e-7) -> torch.Tensor:
    """Softmax with min-probability floor to prevent log(0)."""
    preds_max = torch.max(logits, dim=1, keepdim=True).values
    preds_exp = torch.exp(logits - preds_max)
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True)
    probs = probs + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


def cues_from_pseudo_mask(
    pseudo_mask: torch.Tensor,
    num_classes: int,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    """Convert integer pseudo mask to per-class binary cue maps."""
    mask = torch.nn.functional.interpolate(
        pseudo_mask.float(),
        size=spatial_size,
        mode="nearest",
    )
    B = mask.shape[0]
    H, W = spatial_size
    cues = torch.zeros(B, num_classes, H, W, dtype=torch.float32, device=mask.device)
    for c in range(num_classes):
        pos = torch.where(mask.squeeze(1) == c)
        if len(pos[0]) > 0:
            cues[pos[0], c, pos[1], pos[2]] = 1.0
    return cues
