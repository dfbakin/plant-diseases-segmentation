"""Auxiliary spatial losses for SPDNet training.

Three losses, plus the supporting modules they need:

* ``equivariance_loss``      -- enforces ``T(M(q, r)) ≈ M(T(q), r)`` on the
  head-averaged spatial cross-attention map.
* ``patch_contrastive_loss`` -- supervised contrastive loss on patch
  embeddings, anchored at CAM-peak positions of the active class.
* ``self_distillation_loss`` -- DINO-style centering+sharpening KL between an
  EMA teacher and the student, on per-class spatial logits.
* ``ProjectionHead``         -- 1x1 conv used by ``patch_contrastive_loss``.
* ``EMATeacher``             -- frozen EMA copy of the student SPDNet that
  also EMAs BatchNorm running stats.

The math (formulas, defaults, degenerate-fixed-point analysis) is locked in
``RESEARCH_CONTEXT.md`` §5.11.1 and mirrored in
``/root/.cursor/plans/spdnet_auxiliary_spatial_losses_*.plan.md``. Any change
in this file MUST update both documents in the same commit.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wsss.spdnet import equivariance_transforms as ET

if TYPE_CHECKING:  # pragma: no cover
    from src.wsss.spdnet.model import SPDNet


# ---------------------------------------------------------------------------
# 1. Equivariance loss
# ---------------------------------------------------------------------------


def equivariance_loss(
    attention_orig: torch.Tensor,
    attention_aug: torch.Tensor,
    transform_id: int,
) -> torch.Tensor:
    """MSE between the geometrically-transformed original attention and the
    attention computed on the transformed query.

    Args:
        attention_orig: ``(B, H', W')`` head-averaged attention ``M(q, r)``.
        attention_aug:  ``(B, H', W')`` head-averaged attention ``M(T(q), r)``.
        transform_id: which transform from
            ``src.wsss.spdnet.equivariance_transforms`` was applied to the
            query batch this step.

    Returns:
        Scalar mean-squared error ``mean((T(M_orig) - M_aug)**2)``.
    """
    if attention_orig.shape != attention_aug.shape:
        raise ValueError(
            f"attention_orig {tuple(attention_orig.shape)} and "
            f"attention_aug {tuple(attention_aug.shape)} must agree"
        )
    target = ET.apply(attention_orig, transform_id)
    return F.mse_loss(attention_aug, target)


# ---------------------------------------------------------------------------
# 2. Patch contrastive loss
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """1x1 conv head used by :func:`patch_contrastive_loss`.

    Maps ``in_channels`` (256 by default, the FPN width) to ``out_channels``
    (128 by default, the contrastive embedding dim). Output is intentionally
    not L2-normalised here; the caller normalises after projection so the
    same head can be reused for other contrastive variants.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _first_active_class(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(active_first, valid_mask)`` for a multilabel batch.

    ``active_first[i]`` is the smallest class index where ``labels[i]`` is
    nonzero, or ``-1`` if the row has no active label. ``valid_mask`` is
    ``active_first >= 0``.

    Vectorised: for ``{0, 1}`` labels, ``argmax`` returns the smallest index
    of the maximum, which is the smallest index where ``label == 1`` (or 0
    on an all-zero row, which we mask to ``-1``).
    """
    valid_mask = labels.sum(dim=1) > 0
    active_first = labels.argmax(dim=1).long().masked_fill(~valid_mask, -1)
    return active_first, valid_mask


def patch_contrastive_loss(
    p3_query: torch.Tensor,
    p4_fused: torch.Tensor,
    cls_weight: torch.Tensor,
    labels: torch.Tensor,
    proj_head: nn.Module,
    top_k: int = 8,
    m_negatives: int = 16,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised patch-level InfoNCE on CAM-peak anchors.

    See RESEARCH_CONTEXT.md §5.11.1 for the formula. The denominator for each
    positive ``p`` is ``{p} ∪ N(a)`` (the SupCon "single positive vs N
    negatives" form), so the chance loss for both random and constant
    embeddings is ``log(1 + |N|)``.

    Background sampling deviates from the spec in one place: instead of
    ``sample_M{p : S(p) < median}`` (random) we use a deterministic
    bottom-``M`` selection (the ``M`` lowest-scoring positions, excluding
    anchors). This is a stricter specialisation that removes test
    flakiness; in expectation it picks the same kinds of patches.

    Args:
        p3_query: ``(B, C_in, H', W')`` pre-fusion query features
            (``P3_query_merged``). Gradients flow through this tensor.
        p4_fused: ``(B, C_in, H', W')`` post-fusion features. Used only for
            anchor selection; detached internally so no gradients flow back
            to the SCA module via this loss.
        cls_weight: ``(C, C_in)`` classifier weight matrix
            (``cam_classifier.weight``). Detached internally.
        labels: ``(B, C)`` multilabel one-hot.
        proj_head: 1x1 conv projector ``C_in -> D``; gradients flow through
            it.
        top_k: number of CAM peak anchors per image (must be ``>= 2``).
        m_negatives: number of background negative positions per image.
        temperature: InfoNCE temperature ``τ``.

    Returns:
        Scalar contrastive loss. Returns ``p3_query.sum() * 0`` (zero with
        intact grad chain) in degenerate cases (no active labels in batch,
        spatial dim too small for the requested anchor + negative pool).
    """
    if top_k < 2:
        raise ValueError(f"top_k={top_k}; need >= 2 (at least one positive per anchor)")

    B, C_in, H, W = p3_query.shape
    P = H * W
    device = p3_query.device

    # Early sanity checks (kept OUTSIDE no_grad so the zero-fallback keeps a
    # grad chain through ``p3_query`` -- callers can still ``.backward()``).
    active_first, valid_mask = _first_active_class(labels)
    if not valid_mask.any():
        return p3_query.sum() * 0  # no anchors to contrast
    K = min(top_k, P - 1)  # leave at least one bg position
    if K < 2:
        return p3_query.sum() * 0
    bg_pool_size = P - K
    M_actual = min(m_negatives, bg_pool_size)
    if M_actual < 1:
        return p3_query.sum() * 0

    # 1) Project & L2-normalise the patch embeddings.
    z = proj_head(p3_query)                                             # (B, D, H, W)
    z = F.normalize(z, dim=1, eps=1e-8)
    z_flat = z.flatten(2).permute(0, 2, 1).contiguous()                 # (B, P, D)
    D = z_flat.shape[-1]

    # 2-4) Anchor / background selection (no grad).
    with torch.no_grad():
        cls_w = cls_weight.detach()
        p4 = p4_fused.detach()

        # Per-class spatial logits, restricted to the chosen anchor class per image.
        S_full = torch.einsum("nc,bchw->bnhw", cls_w, p4)              # (B, C, H, W)
        cls_idx = active_first.clamp(min=0)[:, None, None, None].expand(-1, 1, H, W)
        S_anchor = torch.gather(S_full, 1, cls_idx).squeeze(1).flatten(1)  # (B, P)

        # Per-image min-max norm into [0, 1].
        s_min = S_anchor.amin(dim=1, keepdim=True)
        s_max = S_anchor.amax(dim=1, keepdim=True)
        S_norm = (S_anchor - s_min) / (s_max - s_min + 1e-8)

        # Sort positions by score (descending).
        _, all_sorted = torch.sort(S_norm, dim=1, descending=True)      # (B, P)

        anchor_idx = all_sorted[:, :K]                                  # (B, K)
        bg_idx = all_sorted[:, -M_actual:]                              # (B, M_actual)

        # Cross-class mask: True iff label sets are disjoint.
        label_overlap = labels @ labels.t()                             # (B, B)
        cross_mask = label_overlap == 0

    # 5) Gather embeddings (with grad through z_flat).
    anchor_emb = torch.gather(
        z_flat, 1, anchor_idx.unsqueeze(-1).expand(-1, -1, D),
    )                                                                   # (B, K, D)
    bg_emb = torch.gather(
        z_flat, 1, bg_idx.unsqueeze(-1).expand(-1, -1, D),
    )                                                                   # (B, M, D)

    # 6) Per-image InfoNCE.
    losses: list[torch.Tensor] = []
    eye = torch.eye(K, dtype=torch.bool, device=device)
    for i in range(B):
        if not bool(valid_mask[i]):
            continue
        A_i = anchor_emb[i]                                             # (K, D)
        B_i = bg_emb[i]                                                 # (M, D)

        cross_j = torch.nonzero(cross_mask[i], as_tuple=False).squeeze(-1)
        cross_j = cross_j[valid_mask[cross_j]]
        if cross_j.numel() > 0:
            C_i = anchor_emb[cross_j].reshape(-1, D)                    # (J*K, D)
            N_i = torch.cat([B_i, C_i], dim=0)                          # (M + J*K, D)
        else:
            N_i = B_i

        l_pos = (A_i @ A_i.t()) / temperature                           # (K, K)
        l_neg = (A_i @ N_i.t()) / temperature                           # (K, |N_i|)

        # Per-positive denom: log(exp(l_pos[a, p]) + sum_n exp(l_neg[a, n])).
        log_neg_sum = torch.logsumexp(l_neg, dim=1)                     # (K,)
        log_denom = torch.logaddexp(l_pos, log_neg_sum.unsqueeze(1))    # (K, K)
        contrib = (l_pos - log_denom).masked_fill(eye, 0.0)             # (K, K), 0 on diag
        per_anchor = -contrib.sum(dim=1) / (K - 1)                      # (K,)
        losses.append(per_anchor.mean())

    if not losses:
        return p3_query.sum() * 0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 3. Self-distillation
# ---------------------------------------------------------------------------


class EMATeacher(nn.Module):
    """Stop-gradient EMA copy of the student SPDNet.

    Updates teacher parameters AND BatchNorm running stats every optimizer
    step via in-place EMA (the EMA on running stats prevents the "BN drift"
    failure mode noted in the spec).

    The teacher is held in eval mode so its BN uses the running stats during
    forward; ``forward()`` runs under ``torch.no_grad()`` and returns the
    per-class spatial logits ``S^(t)`` used as the distillation target.
    """

    def __init__(self, student: "SPDNet", alpha: float = 0.999) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in [0, 1], got {alpha!r}")
        self.alpha = alpha
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: "SPDNet", alpha: float | None = None) -> None:
        """Apply ``θ_t ← α θ_t + (1 - α) θ_s`` in-place to every floating
        tensor in the teacher's ``state_dict()`` (so BN running stats also
        get EMAed). Integer buffers (e.g. ``num_batches_tracked``) are
        copied verbatim.
        """
        a = self.alpha if alpha is None else alpha
        student_sd = student.state_dict()
        teacher_sd = self.teacher.state_dict()
        for k, v_t in teacher_sd.items():
            v_s = student_sd[k]
            if v_t.dtype.is_floating_point:
                v_t.mul_(a).add_(v_s, alpha=1.0 - a)
            else:
                v_t.copy_(v_s)

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        """Return per-class spatial logits ``S^(t) = W_cls · F^(t)P4`` from
        the teacher, shape ``(B, C, H', W')``. Output is detached.
        """
        feats = self.teacher.extract_merged_features(query, reference)
        fused = feats["fused"]
        return torch.einsum("nc,bchw->bnhw", self.teacher.classifier.weight, fused)


def self_distillation_loss(
    s_student: torch.Tensor,
    s_teacher: torch.Tensor,
    labels: torch.Tensor,
    center: torch.Tensor,
    center_beta: float = 0.9,
    T_teacher: float = 0.04,
    T_student: float = 0.1,
) -> torch.Tensor:
    """DINO-style KL distillation on per-class spatial logits.

    Student loss only; teacher is treated as fixed (callers must pass an
    already-detached ``s_teacher``).

    Args:
        s_student: ``(B, C, H', W')`` student per-class spatial logits.
        s_teacher: ``(B, C, H', W')`` teacher logits, detached.
        labels: ``(B, C)`` multilabel one-hot.
        center: ``(H'·W',)`` running EMA of the teacher's per-position bias.
            Mutated in place AFTER the loss is computed (so the next batch
            sees the updated center, matching DINO).
        center_beta: EMA momentum on ``center`` (default 0.9).
        T_teacher: teacher temperature (sharper).
        T_student: student temperature (softer).

    Returns:
        Scalar ``mean_i KL(P_t_i || P_s_i)``.
    """
    if s_student.shape != s_teacher.shape:
        raise ValueError(
            f"s_student {tuple(s_student.shape)} and s_teacher "
            f"{tuple(s_teacher.shape)} must agree"
        )
    B, C, H, W = s_student.shape
    P = H * W
    if center.shape != (P,):
        raise ValueError(
            f"center shape {tuple(center.shape)} != ({P},); resize before training"
        )
    if T_teacher <= 0 or T_student <= 0:
        raise ValueError(
            f"temperatures must be > 0, got T_teacher={T_teacher}, T_student={T_student}"
        )

    active_first, valid_mask = _first_active_class(labels)
    if not valid_mask.any():
        return s_student.sum() * 0

    cls_idx = active_first.clamp(min=0)[:, None, None, None].expand(-1, 1, H, W)
    S_t = torch.gather(s_teacher, 1, cls_idx).squeeze(1).flatten(1)     # (B, P)
    S_s = torch.gather(s_student, 1, cls_idx).squeeze(1).flatten(1)     # (B, P)

    P_t = torch.softmax((S_t - center.unsqueeze(0)) / T_teacher, dim=1)  # (B, P)
    P_s = torch.softmax(S_s / T_student, dim=1)                          # (B, P)

    log_P_t = torch.log(P_t.clamp_min(1e-12))
    log_P_s = torch.log(P_s.clamp_min(1e-12))
    kl = (P_t * (log_P_t - log_P_s)).sum(dim=1)                          # (B,)

    kl = kl.masked_fill(~valid_mask, 0.0)
    n_valid = int(valid_mask.sum().item())
    loss = kl.sum() / n_valid

    # DINO center EMA on the current batch's teacher logits.
    with torch.no_grad():
        S_t_valid = S_t[valid_mask]                                      # (n_valid, P)
        batch_mean = S_t_valid.mean(dim=0)                               # (P,)
        center.mul_(center_beta).add_(batch_mean, alpha=1.0 - center_beta)

    return loss
