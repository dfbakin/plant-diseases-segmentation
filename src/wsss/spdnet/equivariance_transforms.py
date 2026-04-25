"""Tensor-level geometric transforms used by the equivariance loss.

The transforms operate on any tensor whose last two dims are spatial ``(H, W)``,
so the same ``t_id`` can be applied to a batched image ``(B, 3, H, W)``, an
attention map ``(B, H', W')``, or a feature map ``(B, C, H', W')``. Each
transform has an explicit inverse used by ``L_eq`` to align the augmented
attention back into the original frame.

Color jitter is intentionally excluded: there is no inverse on the attention
map for color perturbations, so ``L_eq`` would not be well-defined.

Square spatial dims (``H == W``) are required for the rotation transforms;
SPDNet always feeds square inputs, so this is not a real restriction.
"""

from __future__ import annotations

import torch

T_ID_IDENTITY: int = 0
T_ID_HFLIP: int = 1
T_ID_ROT90: int = 2
T_ID_ROT180: int = 3
T_ID_ROT270: int = 4

NUM_TRANSFORMS: int = 5
TRANSFORM_NAMES: tuple[str, ...] = ("identity", "hflip", "rot90", "rot180", "rot270")

ALL_TRANSFORMS: frozenset[int] = frozenset({
    T_ID_IDENTITY, T_ID_HFLIP, T_ID_ROT90, T_ID_ROT180, T_ID_ROT270,
})


def apply(x: torch.Tensor, t_id: int) -> torch.Tensor:
    """Apply the geometric transform identified by ``t_id`` to ``x``.

    Args:
        x: tensor with at least 2 dims; the last two are treated as ``(H, W)``.
        t_id: one of the ``T_ID_*`` constants.

    Returns:
        Transformed tensor with the same shape as ``x`` (assuming ``H == W``
        for rot90/rot270).
    """
    if t_id == T_ID_IDENTITY:
        return x
    if t_id == T_ID_HFLIP:
        return x.flip(dims=(-1,))
    if t_id == T_ID_ROT90:
        return x.rot90(k=1, dims=(-2, -1))
    if t_id == T_ID_ROT180:
        return x.rot90(k=2, dims=(-2, -1))
    if t_id == T_ID_ROT270:
        return x.rot90(k=3, dims=(-2, -1))
    raise ValueError(
        f"Unknown transform id: {t_id!r}; expected 0..{NUM_TRANSFORMS - 1}."
    )


def inverse(y: torch.Tensor, t_id: int) -> torch.Tensor:
    """Apply the inverse of the transform identified by ``t_id``.

    ``identity``, ``hflip``, and ``rot180`` are their own inverses; ``rot90``
    inverts to ``rot270`` and vice versa.
    """
    if t_id == T_ID_IDENTITY:
        return y
    if t_id == T_ID_HFLIP:
        return y.flip(dims=(-1,))
    if t_id == T_ID_ROT90:
        return y.rot90(k=-1, dims=(-2, -1))
    if t_id == T_ID_ROT180:
        return y.rot90(k=2, dims=(-2, -1))
    if t_id == T_ID_ROT270:
        return y.rot90(k=1, dims=(-2, -1))
    raise ValueError(
        f"Unknown transform id: {t_id!r}; expected 0..{NUM_TRANSFORMS - 1}."
    )


def sample_transform_id(generator: torch.Generator | None = None) -> int:
    """Uniformly sample a transform id from ``{0, ..., NUM_TRANSFORMS - 1}``.

    Pass an explicit ``generator`` for reproducibility (the EquivarianceLoss
    sampler in ``SPDNetModule`` does this with a per-step generator seeded
    from the global step).
    """
    if generator is None:
        return int(torch.randint(0, NUM_TRANSFORMS, (1,)).item())
    return int(torch.randint(0, NUM_TRANSFORMS, (1,), generator=generator).item())
