"""Gradient-based CAM methods for SPDNet seed generation.

Implements three class-activation-map extractors that differ from the
classifier-projection CAM used by ``generate_all_cams``:

- **LayerCAM** (Jiang et al., TIP 2021): preserves fine-grained gradient
  by taking the element-wise ``ReLU(grad) * act`` before channel-summing.
  Strong for small objects (plant disease lesions).
- **GradCAM++** (Chattopadhay et al., WACV 2018): reweights gradients
  with second- and third-order derivatives to emphasise pixels that
  contribute to the class score.
- **XGradCAM** (Fu et al., BMVC 2020): normalises channel weights by
  activation magnitude; theoretically justified correction to GradCAM.

Default target layer is ``query_merged`` (pre-fusion FPN-merged features,
probe position P3). This is the same signal that ``feat_chvar`` /
``feat_chmean`` read, so the comparison is strictly "different extraction
algorithm on the same representation". Other supported target layers:
``fused`` (post-fusion, P4), ``layer4`` (backbone C5, P1).

Interface mirrors ``generate_spdnet_seed`` in ``cam_generator.py``:
the driver function accepts a list of (multi-scale / flipped) query +
reference tensors and returns ``{0: (H, W) min-max normalised}``.

Implementation notes:
    * Gradients are obtained via ``torch.autograd.grad`` with explicit
      ``inputs=``; this does NOT populate ``param.grad`` on any parameter,
      so there is no leakage across images.
    * The model is expected to be in ``eval()`` mode (BatchNorm frozen)
      before calling these functions.
    * Multi-class images: a per-class CAM is computed for each active
      class and the per-pixel max is taken. Active-class count is
      clamped to ``MAX_CLASSES_PER_IMAGE`` to bound the backward budget.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from src.wsss.spdnet.model import SPDNet

TargetLayer = Literal["query_merged", "fused", "layer4"]
Method = Literal["layercam", "gradcam_pp", "xgradcam"]

# One backward per active class per image; plant-disease images usually
# have 1--2 active classes but multi-disease samples exist. Clamp to
# cap worst-case compute per image.
MAX_CLASSES_PER_IMAGE = 4

# Small epsilon used in the three CAM formulas. Matches standard
# implementations in pytorch-grad-cam.
_EPS = 1e-7


# -----------------------------------------------------------------------
# Forward pass with intermediate activation capture.
# -----------------------------------------------------------------------


def _forward_with_captured_layer(
    model: SPDNet,
    query: torch.Tensor,
    reference: torch.Tensor | list[torch.Tensor],
    target_layer: TargetLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SPDNet forward and return (logits, captured_activation).

    ``captured_activation`` is part of the autograd graph leading to
    ``logits``, so ``torch.autograd.grad(logits[:, c], captured)`` is
    valid for any class ``c``.

    Replicates ``SPDNet._merge_and_fuse`` + classifier head inline so
    we can capture intermediate tensors without monkey-patching the
    model or registering forward hooks (hooks cannot expose the
    ``_merge_fpn`` output because it is produced by a method, not a
    ``nn.Module``).
    """
    if isinstance(reference, torch.Tensor):
        refs = [reference]
    else:
        refs = list(reference)

    # Backbone features (shared for query_merged and layer4 capture).
    feats = model.extract_features(query)  # list of [c2, c3, c4, c5]
    c5 = feats[-1]

    q_fpn_raw = model.fpn(feats)
    q_fpn = [model.mse(p) for p in q_fpn_raw]
    query_merged = model._merge_fpn(q_fpn)

    all_r_fpn: list[list[torch.Tensor]] = []
    for ref in refs:
        r_feats = model.extract_features(ref)
        r_fpn_raw = model.fpn(r_feats)
        all_r_fpn.append([model.mse(p) for p in r_fpn_raw])
    n_refs = len(all_r_fpn)

    if model.fusion_mode == "token":
        avg_tokens: list[torch.Tensor] = []
        for lvl in range(len(q_fpn)):
            lvl_tokens = [model.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
            avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]
        fused = model.adpl_cam.fuse(query_merged, avg_tokens)
    elif model.fusion_mode == "spatial":
        ref_merged_list = [model._merge_fpn(r) for r in all_r_fpn]
        ref_merged = sum(ref_merged_list) / n_refs  # type: ignore[arg-type]
        fused = model.spatial_attn(query_merged, ref_merged)
    else:
        raise ValueError(f"Unknown fusion_mode: {model.fusion_mode!r}")

    pooled = fused.mean(dim=[2, 3])
    logits = model.classifier(pooled)

    if target_layer == "query_merged":
        captured = query_merged
    elif target_layer == "fused":
        captured = fused
    elif target_layer == "layer4":
        captured = c5
    else:
        raise ValueError(f"Unknown target_layer: {target_layer!r}")

    return logits, captured


# -----------------------------------------------------------------------
# CAM aggregation formulas.
#
# Each takes (act, grad) of matching shape ``(B, C, H, W)`` and returns
# a ``(B, H, W)`` per-image CAM. All three ReLU at the end to keep
# only positive evidence, matching standard implementations.
# -----------------------------------------------------------------------


def _cam_layercam(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """LayerCAM: element-wise ReLU(grad) * act, then sum over channels.

    See Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation
    Maps for Localization", TIP 2021, eq. 6.
    """
    weights = F.relu(grad)
    cam = (weights * act).sum(dim=1)
    return F.relu(cam)


def _cam_gradcam_pp(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """GradCAM++: alpha-reweighted channel averages of ReLU(grad).

    See Chattopadhay et al., "Grad-CAM++", WACV 2018, eq. 14.
    Numerator: ``grad**2``. Denominator: ``2*grad**2 + sum_ij(act*grad**3)``.
    """
    g2 = grad.pow(2)
    g3 = grad.pow(3)
    sum_ag3 = (act * g3).sum(dim=(2, 3), keepdim=True)
    alpha = g2 / (2.0 * g2 + sum_ag3 + _EPS)
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    cam = (weights * act).sum(dim=1)
    return F.relu(cam)


def _cam_xgradcam(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """XGradCAM: per-channel weight is ``sum_ij(act*grad) / sum_ij(act)``.

    See Fu et al., "Axiom-based Grad-CAM", BMVC 2020, eq. 9.
    """
    act_sum = act.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    weights = (act * grad).sum(dim=(2, 3), keepdim=True) / (act_sum + _EPS)
    cam = (weights * act).sum(dim=1)
    return F.relu(cam)


_METHOD_TABLE: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "layercam": _cam_layercam,
    "gradcam_pp": _cam_gradcam_pp,
    "xgradcam": _cam_xgradcam,
}


def list_methods() -> list[str]:
    """Return the sorted list of supported gradient-CAM method names."""
    return sorted(_METHOD_TABLE)


def is_gradient_cam_mode(mode: str) -> bool:
    """Return True if ``mode`` is a gradient-based CAM seed-mode name."""
    return mode in _METHOD_TABLE


# -----------------------------------------------------------------------
# Single-image entry point.
# -----------------------------------------------------------------------


def compute_gradient_cam(
    model: SPDNet,
    query: torch.Tensor,
    reference: torch.Tensor | list[torch.Tensor],
    active_classes: list[int],
    method: Method,
    target_layer: TargetLayer = "query_merged",
    max_classes_per_image: int = MAX_CLASSES_PER_IMAGE,
) -> torch.Tensor:
    """Return the feature-resolution CAM for one (query, ref) pair.

    Args:
        model: SPDNet in ``eval()`` mode.
        query: ``(1, 3, H, W)`` query image on the same device as ``model``.
        reference: ``(1, 3, H, W)`` reference tensor or a list of them.
        active_classes: list of foreground class indices to aggregate.
            Clamped to ``max_classes_per_image`` (keeps first N); empty
            list raises ``ValueError``.
        method: one of ``"layercam"``, ``"gradcam_pp"``, ``"xgradcam"``.
        target_layer: one of ``"query_merged"`` (default, P3),
            ``"fused"`` (P4), ``"layer4"`` (P1).
        max_classes_per_image: hard cap on active class count.

    Returns:
        ``(H_f, W_f)`` float32 tensor on ``query.device``, NOT yet
        min-max normalised (the caller handles TTA aggregation + norm).

    Shape agnostic: works for any input resolution; relies on the
    backbone + FPN + SCA being shape-agnostic (they are).
    """
    if method not in _METHOD_TABLE:
        raise ValueError(
            f"method={method!r} is not a gradient-CAM method; valid: {list_methods()}"
        )
    if query.dim() != 4 or query.shape[0] != 1:
        raise ValueError(
            f"query must be (1, 3, H, W); got {tuple(query.shape)}"
        )
    if not active_classes:
        raise ValueError("active_classes must contain at least one class index")

    classes = list(active_classes)[:max_classes_per_image]
    method_fn = _METHOD_TABLE[method]

    # Build a forward graph where the captured activation is in the
    # path between the learnable parameters and the logits. We do NOT
    # require input gradients (``query`` stays detached) -- autograd.grad
    # with explicit ``inputs=`` pulls the gradient w.r.t. the captured
    # tensor regardless, as long as the graph has been constructed
    # with ``requires_grad=True`` somewhere. The model parameters
    # satisfy that by default.
    with torch.enable_grad():
        logits, captured = _forward_with_captured_layer(
            model, query, reference, target_layer=target_layer,
        )
        # captured: (B, C, Hf, Wf); for "layer4" C is 2048, for
        # "query_merged" and "fused" C is the FPN channel count (256).
        per_class: list[torch.Tensor] = []
        for idx, c in enumerate(classes):
            # retain_graph for all but the last class so we can issue
            # multiple backward passes from the same forward.
            retain = idx != len(classes) - 1
            (grad,) = torch.autograd.grad(
                outputs=logits[:, c].sum(),
                inputs=captured,
                retain_graph=retain,
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )
            per_class.append(method_fn(captured, grad))  # each (B, Hf, Wf)

    # Aggregate across classes: per-pixel max = "is this pixel evidence
    # for ANY of the active classes?". Same reducer used by
    # ``generate_all_cams`` for ``binary_aggregate="max"``.
    stacked = torch.stack(per_class, dim=0)  # (K, B, Hf, Wf)
    cam = stacked.amax(dim=0)[0].detach()  # (Hf, Wf)
    return cam


# -----------------------------------------------------------------------
# Multi-scale / flip driver matching ``generate_spdnet_seed``'s interface.
# -----------------------------------------------------------------------


def generate_gradient_spdnet_seed(
    model: SPDNet,
    query_images: list[torch.Tensor],
    ref_image_lists: list[list[torch.Tensor]],
    active_classes: list[int],
    device: torch.device,
    method: Method,
    target_layer: TargetLayer = "query_merged",
    max_classes_per_image: int = MAX_CLASSES_PER_IMAGE,
) -> dict[int, np.ndarray]:
    """Multi-scale/flip gradient-CAM seed, returned as ``{0: (H, W)}`` in [0,1].

    Mirrors ``generate_spdnet_seed``: ``query_images`` is a list of
    (1, 3, H_s, W_s) tensors at different (scale, flip) augmentations;
    ``ref_image_lists`` pairs references to each scale/flip. We assume
    the caller alternates (flip=0, flip=1) within each scale -- same
    convention as ``generate_all_seeds`` (see ``cam_generator.py``).

    Output shape equals the FIRST query's spatial shape (H0, W0). Flip
    augs are un-flipped before averaging.

    Normalised to ``[0, 1]`` across all TTA augs.
    """
    if not query_images:
        raise ValueError("query_images must not be empty")
    if len(query_images) != len(ref_image_lists):
        raise ValueError(
            f"query_images ({len(query_images)}) vs ref_image_lists "
            f"({len(ref_image_lists)}) length mismatch"
        )

    h_orig = query_images[0].shape[2]
    w_orig = query_images[0].shape[3]
    seed_2d_list: list[np.ndarray] = []

    for s, (q_img, r_imgs) in enumerate(zip(query_images, ref_image_lists)):
        q_img = q_img.to(device)
        refs = [r.to(device) for r in r_imgs]
        ref_arg = refs[0] if len(refs) == 1 else refs

        cam_f = compute_gradient_cam(
            model=model, query=q_img, reference=ref_arg,
            active_classes=active_classes, method=method,
            target_layer=target_layer,
            max_classes_per_image=max_classes_per_image,
        )  # (Hf, Wf) on device

        cam_up = F.interpolate(
            cam_f[None, None], size=(h_orig, w_orig),
            mode="bilinear", align_corners=False,
        )[0, 0]
        seed = cam_up.detach().cpu().numpy()

        # Undo hflip (same convention as ``generate_spdnet_seed``).
        if s % 2 == 1:
            seed = np.flip(seed, axis=-1)
        seed_2d_list.append(np.asarray(seed, dtype=np.float32))

    merged = np.mean(np.stack(seed_2d_list, axis=0), axis=0)
    vmin, vmax = float(merged.min()), float(merged.max())
    if vmax - vmin > 1e-8:
        merged = (merged - vmin) / (vmax - vmin)
    else:
        merged = np.zeros_like(merged)
    return {0: merged.astype(np.float32).copy()}
