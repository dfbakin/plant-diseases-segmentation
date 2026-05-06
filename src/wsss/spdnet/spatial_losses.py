"""Auxiliary spatial losses for SPDNet training.

Five losses, plus the supporting modules they need:

* ``equivariance_loss``       -- enforces ``T(M(q, r)) ≈ M(T(q), r)`` on the
  head-averaged spatial cross-attention map.
* ``patch_contrastive_loss``  -- supervised contrastive loss on patch
  embeddings, anchored at CAM-peak positions of the active class (or the
  element-wise max-rank of classifier score and chvar saliency, when
  ``anchor_source="union_cls_chvar"``).
* ``self_distillation_loss``  -- DINO-style centering+sharpening KL between an
  EMA teacher and the student, on per-class spatial logits.
* ``attention_concentration_loss`` (D1) -- pushes the attention-concentration
  map away from its uniform fixed point so ``L_eq`` has structure to
  preserve. Loss is ``-mean(attn_map)``.
* ``cam_pseudo_mask_loss`` (D2) -- direct pseudo-mask supervision of the
  active-class CAM slice, with positive seeds from channel-variance saliency
  (optionally intersected with CAM's own top-alpha) and negative seeds from
  the chvar bottom-beta. Loss is a per-image MSE on min-max-normalised CAM
  vs the pseudo-mask target, weighted by the pos+neg supervision region.
* ``ProjectionHead``          -- 1x1 conv used by ``patch_contrastive_loss``.
* ``EMATeacher``              -- frozen EMA copy of the student SPDNet that
  also EMAs BatchNorm running stats.

The math (formulas, defaults, degenerate-fixed-point analysis) is locked in
``RESEARCH_CONTEXT.md`` §5.11.1 / §5.13.7 and mirrored in
``/root/.cursor/plans/spdnet_auxiliary_spatial_losses_*.plan.md``. Any change
in this file MUST update both documents in the same commit.
"""

from __future__ import annotations

import copy
import math
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
    anchor_source: str = "classifier",
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
        anchor_source: one of ``"classifier"`` (default; classifier-CAM
            argsort only) or ``"union_cls_chvar"`` (anchor rank = element-wise
            max of classifier-CAM rank and ``p3_query`` channel-variance
            rank). Background negatives always come from the bottom of the
            classifier score, independent of this setting, so the polarity
            between anchors and negatives is preserved.

    Returns:
        Scalar contrastive loss. Returns ``p3_query.sum() * 0`` (zero with
        intact grad chain) in degenerate cases (no active labels in batch,
        spatial dim too small for the requested anchor + negative pool).
    """
    if top_k < 2:
        raise ValueError(f"top_k={top_k}; need >= 2 (at least one positive per anchor)")
    if anchor_source not in {"classifier", "union_cls_chvar"}:
        raise ValueError(
            f"anchor_source={anchor_source!r} must be one of "
            "{'classifier', 'union_cls_chvar'}"
        )

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

        # Per-image min-max norm into [0, 1] (only needed for classifier;
        # union path uses raw ranks which are scale-invariant).
        s_min = S_anchor.amin(dim=1, keepdim=True)
        s_max = S_anchor.amax(dim=1, keepdim=True)
        S_norm = (S_anchor - s_min) / (s_max - s_min + 1e-8)

        # Always sort classifier scores for background (negatives) selection.
        _, cls_sorted = torch.sort(S_norm, dim=1, descending=True)      # (B, P)
        bg_idx = cls_sorted[:, -M_actual:]                              # (B, M_actual)

        if anchor_source == "classifier":
            anchor_idx = cls_sorted[:, :K]                              # (B, K)
        else:  # union_cls_chvar
            # chvar: class-agnostic saliency from pre-fusion features.
            chvar = p3_query.detach().var(dim=1, unbiased=False).flatten(1)  # (B, P)
            # Per-position rank of classifier score (0=low, P-1=high).
            rank_cls = S_norm.argsort(dim=1).argsort(dim=1)              # (B, P)
            rank_cv = chvar.argsort(dim=1).argsort(dim=1)                # (B, P)
            combined_rank = torch.maximum(rank_cls, rank_cv)             # (B, P)
            anchor_idx = torch.topk(combined_rank, k=K, dim=1).indices   # (B, K)

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
# 3. Attention concentration regulariser (D1)
# ---------------------------------------------------------------------------


def attention_concentration_loss(attn_map: torch.Tensor) -> torch.Tensor:
    """Push the per-query attention concentration map away from uniformity.

    ``attn_map`` is the ``(B, H, W)`` normalised negative entropy of each
    query's softmax distribution over reference keys (built in
    ``SpatialCrossAttention._spatial_attn_with_map``). Its range is
    ``[0, 1]``: ``0`` means uniform attention across all keys (maximum
    entropy), ``1`` means perfectly peaked on a single key (zero entropy).

    On the eq-only checkpoint, the observed per-image spatial std of
    ``attn_map`` is ~0.004 on the [0, 1] scale -- i.e. it is a near-constant
    field around ~0.345. That fixed point is what makes ``L_eq`` trivially
    satisfied (any transform of a constant map is the same constant map,
    so MSE ≈ 0).

    This loss maximises the mean concentration by minimising its negative:

    .. math::

        \\mathcal{L}_{\\mathrm{ac}} = -\\frac{1}{B H W} \\sum_{b, y, x} M(q, r)_{b, y, x}

    Gradients flow through the entropy term to the attention weights and
    therefore to the SCA in-projection of queries, keys and values (the map
    is computed by a dropout-free second MHA forward).

    Args:
        attn_map: ``(B, H, W)`` tensor whose values are assumed to lie in
            ``[0, 1]``.

    Returns:
        Scalar ``-mean(attn_map)``; lower is "more concentrated".
    """
    if attn_map.dim() != 3:
        raise ValueError(
            f"attn_map must be (B, H, W); got shape {tuple(attn_map.shape)}"
        )
    return -attn_map.mean()


def attention_marginal_entropy_loss(
    attn_w: torch.Tensor, beta: float = 0.25,
) -> torch.Tensor:
    """Mode-collapse-free attention regulariser (D4 candidate, RQ2).

    Combines per-query concentration (``L_ac`` term) with a per-key
    *marginal* dispersion term so that the loss has a non-degenerate
    minimum at "every query sharp, marginal over keys uniform". That
    minimum corresponds to queries spreading their peaks across many
    distinct keys, which is the desired spatial behaviour. In contrast,
    ``attention_concentration_loss`` alone admits the trivial minimum
    "all queries peak on the same key" (D1's observed collapse).

    Formally, for attention weights :math:`p_{b, q, k}` with
    :math:`\\sum_k p_{b, q, k} = 1`:

    .. math::

        M_{b, q}         &= 1 + \\tfrac{1}{\\log N}
                            \\sum_k p_{b, q, k} \\log p_{b, q, k}
                           \\in [0, 1] \\\\
        \\mu_k           &= \\tfrac{1}{B P}
                            \\sum_{b, q} p_{b, q, k}
                           \\quad \\text{(per-key marginal)} \\\\
        H(\\mu)          &= -\\sum_k \\mu_k \\log \\mu_k \\\\
        \\mathrm{KL}(\\mu \\Vert U)
                         &= \\log N - H(\\mu) \\\\
        \\mathcal{L}_{\\mathrm{marg\\,H}}
                         &= -\\overline{M} + \\beta \\,
                            \\mathrm{KL}(\\mu \\Vert U)

    Fixed-point analysis (see
    ``reports/notes/rq2_attention_regularizer_analysis.md``):

    * ``-mean(M)`` term alone: minimum at ``p = e_{k*}`` for any fixed
      key ``k*`` (mode collapse admissible).
    * ``KL(mu || U)`` alone: minimum at ``p`` uniform in keys (which
      ``L_eq`` is trivially satisfied by).
    * Sum: minimum is reached only when ``p_{b, q, ·}`` are individually
      sharp AND their average over queries is uniform -- i.e. different
      queries pick different keys. No trivial collapse.

    Note:
        At ``beta = 0.0``, ``attention_marginal_entropy_loss(attn_w)``
        reduces to ``attention_concentration_loss(attn_map)`` up to a
        floating-point tolerance (``attn_map`` is derived from the same
        ``attn_w`` by ``1 + mean_k(p · log p) / log N``).

    Args:
        attn_w: ``(B, P, N)`` post-softmax attention weights per query
            position. Rows must sum to 1 and be in ``[0, 1]``.
        beta: weight on the marginal-uniformity KL term. 0.25 is the
            RQ2 recommendation; larger values push more strongly toward
            spread-out marginals at the cost of per-query sharpness.

    Returns:
        Scalar ``-mean(M) + beta * KL(mu || U)``.
    """
    if attn_w.dim() != 3:
        raise ValueError(
            f"attn_w must be (B, P, N); got shape {tuple(attn_w.shape)}"
        )
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    N = attn_w.shape[-1]
    log_N = math.log(N)
    p = attn_w.clamp_min(1e-12)
    neg_ent_q = (p * p.log()).sum(dim=-1)                     # (B, P) in [-log N, 0]
    M = 1.0 + neg_ent_q / log_N                               # (B, P) in [0, 1]
    mu = p.mean(dim=(0, 1)).clamp_min(1e-12)                  # (N,)
    H_mu = -(mu * mu.log()).sum()                             # in [0, log N]
    kl_to_uniform = log_N - H_mu                              # in [0, log N]
    return -M.mean() + beta * kl_to_uniform


def attention_argmax_share_loss(
    attn_w: torch.Tensor, beta: float = 2.0,
) -> torch.Tensor:
    """Backup attention regulariser: directly penalise single-key dominance.

    Same intent as :func:`attention_marginal_entropy_loss` but measures
    "dominance of the most-attended key" more literally via the
    per-query argmax. Useful as a sanity comparison if the smoother
    marginal-entropy term ever misbehaves in practice.

    Let :math:`k^*_{b, q} = \\arg\\max_k p_{b, q, k}` (computed no-grad).
    Define the *argmax share* of key ``k`` as

    .. math::

        s_k = \\tfrac{1}{B P} \\sum_{b, q}
              \\mathbf{1}[k^*_{b, q} = k] .

    Then :math:`\\max_k s_k` is the fraction of all queries whose peak
    lands on the single most-popular key. A healthy attention has
    :math:`\\max_k s_k \\approx 1 / N`; D1-style mode collapse pushes it
    to 1.

    The loss is

    .. math::

        \\mathcal{L}_{\\mathrm{argmax\\,share}}
            = -\\overline{M} + \\beta \\cdot \\max_k s_k .

    Because ``argmax`` has zero gradient almost everywhere, this
    function uses a *soft* argmax surrogate for the second term:
    ``softmax(p / tau)`` with ``tau = 0.1`` is summed over queries, and
    we take its max-over-keys via ``torch.amax`` (which has a valid
    subgradient). The first term keeps gradients flowing through every
    ``p``.

    Args:
        attn_w: ``(B, P, N)`` post-softmax attention weights.
        beta: weight on the dominance-penalty term. 2.0 is the RQ2
            recommendation.

    Returns:
        Scalar ``-mean(M) + beta * max_k (soft-share_k)``.
    """
    if attn_w.dim() != 3:
        raise ValueError(
            f"attn_w must be (B, P, N); got shape {tuple(attn_w.shape)}"
        )
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    B, P, N = attn_w.shape
    log_N = math.log(N)
    p = attn_w.clamp_min(1e-12)
    neg_ent_q = (p * p.log()).sum(dim=-1)
    M = 1.0 + neg_ent_q / log_N

    # Soft argmax surrogate: sharpen each (b, q) distribution and
    # average over (b, q). Taking amax over keys picks out the key with
    # the highest aggregate soft-indicator -- a differentiable proxy for
    # max_k share_k. tau=0.1 is sharp but not degenerate.
    tau = 0.1
    soft_argmax = torch.softmax(p / tau, dim=-1)              # (B, P, N)
    soft_share = soft_argmax.mean(dim=(0, 1))                 # (N,)
    dominance = soft_share.amax()
    return -M.mean() + beta * dominance


# ---------------------------------------------------------------------------
# 4. Pseudo-mask CAM supervision (D2)
# ---------------------------------------------------------------------------


def _kth_threshold(values: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    """Return a ``(B, 1)`` threshold such that ``k`` elements of each row
    are at or beyond it.

    For ``largest=True`` the threshold is the ``k``-th largest value, so
    ``values >= threshold`` selects exactly ``k`` positions (ties included).
    For ``largest=False`` it's the ``k``-th smallest, selecting ``values <=
    threshold``.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k > values.shape[1]:
        raise ValueError(
            f"k={k} exceeds number of positions {values.shape[1]}"
        )
    topk_vals, _ = torch.topk(values, k=k, dim=1, largest=largest)
    return topk_vals[:, -1:]                                          # (B, 1)


_VALID_MASK_COMBINERS = ("intersection", "chvar_only", "union")


def cam_pseudo_mask_loss(
    p3_query: torch.Tensor,
    p4_fused: torch.Tensor,
    cls_weight: torch.Tensor,
    labels: torch.Tensor,
    alpha_pos: float = 0.25,
    beta_neg: float = 0.5,
    use_intersection: bool | None = None,
    mask_combiner: str = "intersection",
) -> torch.Tensor:
    """Pseudo-mask MSE supervision on the active-class CAM slice.

    The training signal is derived from ``chvar = Var_c(p3_query)``, a
    class-agnostic saliency map that the anchor diagnostic (200 val images
    on the eq-only ckpt) showed to have ``IoU(top-25%, GT) = 0.27`` vs the
    eq-only CAM's single-threshold IoU of ~0.25 -- i.e. chvar is a mildly
    better localiser than the model's own CAM, and therefore a valid
    teaching signal.

    Per image:

    1. Compute ``CAM[c_active]`` from ``p4_fused`` and ``cls_weight``, then
       per-image min-max normalise to ``[0, 1]`` (``cam_norm``).
    2. Compute ``chvar = Var_c(p3_query).detach()`` (B, H, W).
    3. Positive supervision mask (``pos_mask``): built from ``chvar`` and,
       optionally, ``cam_norm`` (detached) according to ``mask_combiner``:

       * ``"intersection"`` (old D2): ``chvar_top_alpha AND cam_top_alpha``.
         Highest precision (~0.41 at alpha=0.25, measured on eq-only) at
         the cost of coverage.
       * ``"chvar_only"``: just ``chvar_top_alpha``. Precision ~0.32 but
         higher recall; useful when the model's own CAM is unreliable.
       * ``"union"`` (D4, per RQ5 `reports/notes/rq5_teacher_ceiling.md`):
         ``chvar_top_alpha OR cam_top_alpha``. Measured +3 pp IoU vs
         either source alone on the eq-only validation split because it
         catches disease regions that each signal misses individually.
    4. Negative supervision mask (``neg_mask``): bottom-``beta_neg``
       fraction of ``chvar``. Negatives are high-precision because disease
       occupies a minority of the image (~21% mean at feat-res), so the
       bottom 50% of chvar rarely contains disease. If a positive position
       falls inside the negative band (chvar ties), the negative wins so
       the two masks stay disjoint.
    5. Loss: MSE of ``cam_norm`` vs a binary target (1 at positives, 0 at
       negatives), weighted per-pixel by ``pos | neg`` and reduced per image
       as a supervised-positions mean.

    Args:
        p3_query: ``(B, C_in, H, W)`` pre-fusion features. Used for chvar
            only; detached internally, no gradients flow through this arg.
        p4_fused: ``(B, C_in, H, W)`` post-fusion features. Gradients flow
            through the CAM back to the fusion module.
        cls_weight: ``(C, C_in)`` classifier weights. Detached internally.
        labels: ``(B, C)`` multilabel one-hot. Images with no active label
            are skipped in the mean.
        alpha_pos: fraction of positions (per image) to treat as positives.
            Must satisfy ``0 < alpha_pos < 1 - beta_neg``.
        beta_neg: fraction of positions (per image) to treat as negatives.
        use_intersection: DEPRECATED alias for ``mask_combiner``. When
            provided (non-``None``), overrides ``mask_combiner``:
            ``True -> "intersection"``, ``False -> "chvar_only"``. Kept
            for one release so existing configs keep working; new code
            should pass ``mask_combiner`` directly.
        mask_combiner: one of ``"intersection"``, ``"chvar_only"``,
            ``"union"``. See the description of ``pos_mask`` construction
            above. Default ``"intersection"`` preserves the old D2
            behaviour.

    Returns:
        Scalar MSE loss. ``p4_fused.sum() * 0`` (grad-preserving zero) when
        no image has any active label or the supervised region is empty
        for every image.
    """
    if not (0.0 < alpha_pos < 1.0):
        raise ValueError(f"alpha_pos must be in (0, 1); got {alpha_pos}")
    if not (0.0 < beta_neg < 1.0):
        raise ValueError(f"beta_neg must be in (0, 1); got {beta_neg}")
    if alpha_pos + beta_neg >= 1.0:
        raise ValueError(
            f"alpha_pos + beta_neg must be < 1 (to leave an unsupervised middle); "
            f"got alpha={alpha_pos}, beta={beta_neg}"
        )

    if use_intersection is not None:
        mask_combiner = "intersection" if use_intersection else "chvar_only"
    if mask_combiner not in _VALID_MASK_COMBINERS:
        raise ValueError(
            f"mask_combiner={mask_combiner!r} must be one of "
            f"{_VALID_MASK_COMBINERS}"
        )

    B, C_in, H, W = p4_fused.shape
    P = H * W

    active_first, valid_mask = _first_active_class(labels)
    if not valid_mask.any():
        return p4_fused.sum() * 0

    # Active-class CAM slice with grad.
    S_full = torch.einsum("nc,bchw->bnhw", cls_weight, p4_fused)       # (B, C, H, W)
    cls_idx = active_first.clamp(min=0)[:, None, None, None].expand(-1, 1, H, W)
    cam_act = torch.gather(S_full, 1, cls_idx).squeeze(1)              # (B, H, W)

    # Per-image min-max normalisation into [0, 1]. Prevents the CAM scale
    # from dominating the MSE target.
    cam_flat = cam_act.flatten(1)                                      # (B, P)
    cam_min = cam_flat.amin(dim=1, keepdim=True)
    cam_max = cam_flat.amax(dim=1, keepdim=True)
    cam_norm = ((cam_flat - cam_min) / (cam_max - cam_min + 1e-8)).view(B, H, W)

    # Seed selection (no grad).
    with torch.no_grad():
        chvar = p3_query.detach().var(dim=1, unbiased=False)           # (B, H, W)
        chvar_flat = chvar.flatten(1)                                  # (B, P)

        k_pos = max(1, int(round(alpha_pos * P)))
        k_neg = max(1, int(round(beta_neg * P)))
        # Safety: ensure pos+neg disjoint and fit.
        if k_pos + k_neg > P:
            k_pos = max(1, P // 4)
            k_neg = max(1, P // 2)

        thr_pos = _kth_threshold(chvar_flat, k_pos, largest=True)      # (B, 1)
        thr_neg = _kth_threshold(chvar_flat, k_neg, largest=False)     # (B, 1)
        chvar_top = (chvar_flat >= thr_pos).view(B, H, W).float()
        neg_mask = (chvar_flat <= thr_neg).view(B, H, W).float()

        if mask_combiner == "chvar_only":
            pos_mask = chvar_top
        else:
            cam_flat_det = cam_norm.detach().flatten(1)
            thr_cam = _kth_threshold(cam_flat_det, k_pos, largest=True)
            cam_top = (cam_flat_det >= thr_cam).view(B, H, W).float()
            if mask_combiner == "intersection":
                pos_mask = chvar_top * cam_top                         # AND
            else:  # "union"
                pos_mask = torch.maximum(chvar_top, cam_top)           # OR

        # Guarantee pos and neg are disjoint (numerics safety when chvar
        # has ties that straddle both thresholds).
        pos_mask = pos_mask * (1.0 - neg_mask)

    # Target: 1 at positives, 0 at negatives, unsupervised elsewhere.
    target = pos_mask
    weight = pos_mask + neg_mask                                        # (B, H, W) in {0, 1}

    # Per-image supervised-pixel MSE. Skip images whose supervision area
    # (or active label) is empty.
    sq = (cam_norm - target) ** 2                                       # (B, H, W)
    per_image_num = (sq * weight).sum(dim=(1, 2))                       # (B,)
    per_image_den = weight.sum(dim=(1, 2))                              # (B,)

    active = valid_mask & (per_image_den > 0)
    if not active.any():
        return p4_fused.sum() * 0

    per_image = per_image_num[active] / per_image_den[active].clamp_min(1.0)
    return per_image.mean()


# ---------------------------------------------------------------------------
# 5. Self-distillation
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
