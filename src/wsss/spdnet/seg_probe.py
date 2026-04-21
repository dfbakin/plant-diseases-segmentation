"""SPDNet localization-capacity probe.

A small two-layer 1x1 conv head trained to predict a binary disease mask
from the features emitted at one of six "probe positions" inside SPDNet.
The host SPDNet may be either fully frozen (Phase 1 -- pure probing) or
fully unfrozen (Phase 2 -- targeted fine-tune).

The probe head is intentionally tiny so that any IoU it achieves is mostly
a property of the underlying feature map, not the head capacity. The head
is large enough to express ``feat_chmean`` exactly (uniform 1/C weights)
and to approximate ``cam_classifier_max`` (weighted ReLU sum), but cannot
exactly represent quadratic statistics like ``feat_chvar`` or
``feat_l2norm`` -- which is *intentional* and motivates running those
non-trainable baselines alongside the probe in eval (see
``scripts/eval_seg_probes.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint
from src.wsss.spdnet.model import SPDNet


PROBE_POSITIONS = (
    "P1_layer4",
    "P2_fpn_p2",
    "P3_query_merged",
    "P4_fused",
    "P5_cam_classifier",
    "P6_attn_map",
)
SPATIAL_ONLY_POSITIONS = ("P6_attn_map",)
NEEDS_REFERENCE = ("P4_fused", "P5_cam_classifier", "P6_attn_map")


def channels_for_position(model: SPDNet, position: str) -> int:
    """Return the channel count of the activation at *position* for *model*.

    Cheap helper -- avoids needing a forward pass to know the head input
    size when constructing the probe.
    """
    if position == "P1_layer4":
        return 2048
    if position == "P2_fpn_p2":
        return model.fpn_channels
    if position == "P3_query_merged":
        return model.fpn_channels
    if position == "P4_fused":
        return model.fpn_channels
    if position == "P5_cam_classifier":
        return model.num_classes
    if position == "P6_attn_map":
        return 1
    raise ValueError(f"Unknown probe position: {position!r}")


class ProbeHead(nn.Module):
    """``Conv1x1(C_in -> H) -> ReLU -> Conv1x1(H -> 1) -> bilinear upsample``.

    The output is the raw segmentation logit (no sigmoid) at the target
    resolution. Sigmoid is applied inside the loss / at eval time only.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        target_size: tuple[int, int] = (448, 448),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.target_size = target_size

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = F.interpolate(
            h, size=self.target_size, mode="bilinear", align_corners=False,
        )
        return h


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on the foreground (positive) class.

    *logits*: (B, 1, H, W) raw, *target*: (B, 1, H, W) in {0, 1}.
    """
    probs = torch.sigmoid(logits)
    p = probs.flatten(1)
    t = target.flatten(1).float()
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Convex combination of pixel-wise BCE and soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice


@dataclass
class SPDNetWithProbesConfig:
    position: str
    head_hidden_dim: int = 64
    target_size: tuple[int, int] = (448, 448)
    freeze_backbone: bool = True


class SPDNetWithProbes(nn.Module):
    """SPDNet wrapped with a single ``ProbeHead`` at *position*.

    The wrapper:
      - Loads a pretrained SPDNet checkpoint (token or spatial).
      - Optionally sets ``requires_grad_(False)`` on every module that is
        not the probe head, so that the optimizer can be safely passed
        ``self.head.parameters()`` (frozen mode) or ``self.parameters()``
        (unfrozen mode -- but make sure to filter out non-grad params if
        you mix modes).
      - Exposes ``forward(query, refs)`` returning ``(seg_logits, cls_logits)``.

    Note: there are NO PyTorch forward hooks. We achieve the same effect
    by routing through ``model.extract_probe_features`` which returns a
    dict of activations and is part of the model's public API. Hooks are
    avoided to keep the lifecycle trivial (no .remove() needed) and to
    make backward-pass dependencies explicit.
    """

    def __init__(
        self,
        spdnet: SPDNet,
        position: str,
        head_hidden_dim: int = 64,
        target_size: tuple[int, int] = (448, 448),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if position not in PROBE_POSITIONS:
            raise ValueError(
                f"Unknown probe position: {position!r}. "
                f"Allowed: {PROBE_POSITIONS}"
            )
        if position in SPATIAL_ONLY_POSITIONS and spdnet.fusion_mode != "spatial":
            raise ValueError(
                f"Probe position {position!r} requires fusion_mode='spatial' "
                f"but checkpoint has fusion_mode={spdnet.fusion_mode!r}."
            )

        self.spdnet = spdnet
        self.position = position
        self.needs_reference = position in NEEDS_REFERENCE
        self.freeze_backbone = freeze_backbone

        in_channels = channels_for_position(spdnet, position)
        self.head = ProbeHead(
            in_channels=in_channels,
            hidden_channels=head_hidden_dim,
            target_size=target_size,
        )

        if freeze_backbone:
            for p in self.spdnet.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        position: str,
        num_classes: int = 115,
        fpn_channels: int = 256,
        head_hidden_dim: int = 64,
        target_size: tuple[int, int] = (448, 448),
        freeze_backbone: bool = True,
    ) -> "SPDNetWithProbes":
        """Convenience: load checkpoint then wrap."""
        spdnet = load_spdnet_from_checkpoint(
            checkpoint, num_classes=num_classes, fpn_channels=fpn_channels,
        )
        return cls(
            spdnet=spdnet,
            position=position,
            head_hidden_dim=head_hidden_dim,
            target_size=target_size,
            freeze_backbone=freeze_backbone,
        )

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def extract_features_at_position(
        self,
        query: torch.Tensor,
        ref_images: list[torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the activation at this probe's position + the full feature dict.

        ``ref_images`` may be ``None`` for positions that do not need it
        (P1/P2/P3) -- we then skip the reference forward path entirely.
        """
        ref = None if not self.needs_reference else ref_images
        feats = self.spdnet.extract_probe_features(query, reference=ref)
        if self.position not in feats:
            raise RuntimeError(
                f"Position {self.position!r} not present in features dict "
                f"(keys={list(feats.keys())}). Did you forget to pass refs?"
            )
        return feats[self.position], feats

    def forward(
        self,
        query: torch.Tensor,
        ref_images: list[torch.Tensor] | torch.Tensor | None = None,
        return_cls: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the host SPDNet, extract the chosen activation, and apply the head.

        If ``return_cls`` is True, also returns the SPDNet classification
        logits (re-used as the multi-task aux head). For frozen-backbone
        mode the classification logits are computed under ``no_grad`` to
        save memory.
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            feat, feats = self.extract_features_at_position(query, ref_images)

        if not self.freeze_backbone:
            seg_logits = self.head(feat)
        else:
            seg_logits = self.head(feat.detach())

        if not return_cls:
            return seg_logits

        if self.freeze_backbone:
            with torch.no_grad():
                cls_logits = self._cls_logits(feats)
        else:
            cls_logits = self._cls_logits(feats)
        return seg_logits, cls_logits

    def _cls_logits(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reproduce SPDNet's classification head from the cached fused feature.

        Uses the ``P4_fused`` activation when available (training-time
        spatial path); falls back to ``P3_query_merged`` for the reference-
        less branch. The result is ``classifier(GAP(feature_map))`` exactly
        as in ``SPDNet.forward``.
        """
        feat = feats.get("P4_fused", feats["P3_query_merged"])
        pooled = feat.mean(dim=[2, 3])
        return self.spdnet.classifier(pooled)
