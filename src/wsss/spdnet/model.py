"""SPDNet model: Siamese ResNet50 + FPN + MSE + ADPL-CAM.

Architecture:
  - Shared ResNet50 backbone extracts multi-scale features (layer1-4)
  - FPN merges them into a common 256-ch representation
  - MSE (Multi-Scale Excitation) applies channel attention
  - ADPL-CAM fuses reference tokens into query features for CAM generation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

ADPL_CAM_LEVELS = 4  # number of FPN levels used in ADPL-CAM


class MSE(nn.Module):
    """Multi-Scale Excitation: channel attention via GAP+GMP (Eq 5-6)."""

    def __init__(self, channels: int, reduction: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        hidden = channels // reduction
        self.fc1 = nn.Linear(channels * 2, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, channels)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        z = torch.cat([avg, mx], dim=1)  # (B, 2C)
        a = self.drop(self.prelu(self.bn(self.fc1(z))))
        a = torch.sigmoid(self.fc2(a))  # (B, C)
        return x * a.view(b, c, 1, 1)


class FPN(nn.Module):
    """Feature Pyramid Network (Eq 13-15).

    Takes ResNet layer outputs {C2..C5} and produces {P2..P5} all at
    ``out_channels`` depth via lateral connections + top-down merging.
    """

    def __init__(
        self,
        in_channels: list[int] = [256, 512, 1024, 2048],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ic, out_channels, 1) for ic in in_channels]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="nearest")
            laterals[i] = laterals[i] + up
        return [s(l) for s, l in zip(self.smooth, laterals)]


class ADPLCam(nn.Module):
    """ADPL-CAM: reference-guided token fusion (Eq 17-18).

    Tokenizes reference FPN features via GlobalMaxPool, then fuses
    them into query features with learnable per-level weights.
    """

    def __init__(self, num_levels: int = 4) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_levels) * 0.1)

    def tokenize(self, ref_fpn: list[torch.Tensor]) -> list[torch.Tensor]:
        """GlobalMaxPool each FPN level -> (B, C) tokens."""
        return [f.amax(dim=[2, 3]) for f in ref_fpn]

    def fuse(
        self,
        query_feat: torch.Tensor,
        ref_tokens: list[torch.Tensor],
    ) -> torch.Tensor:
        """Add weighted reference tokens to query feature map (Eq 18)."""
        out = query_feat
        for i, t in enumerate(ref_tokens):
            out = out + self.alpha[i] * t.unsqueeze(-1).unsqueeze(-1)
        return out


class SPDNet(nn.Module):
    """Siamese Plant Disease Network.

    Shared ResNet50 backbone + FPN + MSE + ADPL-CAM.
    Reference tokens are fused into query features BEFORE classification,
    so the reference image influences both training logits and inference CAMs.
    """

    def __init__(
        self,
        num_classes: int = 115,
        fpn_channels: int = 256,
        mse_reduction: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels

        backbone = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=fpn_channels)
        self.mse = MSE(channels=fpn_channels, reduction=mse_reduction)
        self.classifier = nn.Linear(fpn_channels, num_classes)
        self.adpl_cam = ADPLCam(num_levels=ADPL_CAM_LEVELS)

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from shared backbone."""
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

    def _merge_and_fuse(
        self,
        q_fpn: list[torch.Tensor],
        all_r_fpn: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Merge query FPN levels, tokenize reference(s), and fuse (Eq 16-18)."""
        query_merged = self._merge_fpn(q_fpn)

        n_refs = len(all_r_fpn)
        avg_tokens: list[torch.Tensor] = []
        for lvl in range(len(q_fpn)):
            lvl_tokens = [self.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
            avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]

        return self.adpl_cam.fuse(query_merged, avg_tokens)

    def _get_fpn_features(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run backbone + FPN + MSE on an image batch."""
        feats = self.extract_features(x)
        fpn_out = self.fpn(feats)
        return [self.mse(p) for p in fpn_out]

    def _merge_fpn(self, fpn_levels: list[torch.Tensor]) -> torch.Tensor:
        """Average-merge FPN levels to finest resolution.

        Returns:
            Merged feature map (B, C, H0, W0) at the resolution of level 0.
        """
        target_size = fpn_levels[0].shape[2:]
        merged = torch.zeros_like(fpn_levels[0])
        for level in fpn_levels:
            merged = merged + F.interpolate(
                level, size=target_size, mode="bilinear", align_corners=False
            )
        return merged / len(fpn_levels)

    def extract_merged_features(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract intermediate feature maps for seed generation.

        Returns dict with:
            query_merged: (B, C, Hf, Wf) merged FPN features before fusion
            ref_merged:   (B, C, Hf, Wf) merged reference FPN features (if ref given)
            fused:        (B, C, Hf, Wf) features after token fusion (if ref given)
        """
        q_fpn = self._get_fpn_features(query)
        query_merged = self._merge_fpn(q_fpn)
        result = {"query_merged": query_merged}

        if reference is not None:
            refs = [reference] if isinstance(reference, torch.Tensor) else reference
            all_r_fpn = [self._get_fpn_features(r) for r in refs]

            n_refs = len(all_r_fpn)
            avg_tokens: list[torch.Tensor] = []
            for lvl in range(len(q_fpn)):
                lvl_tokens = [self.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
                avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]
            fused = self.adpl_cam.fuse(query_merged, avg_tokens)
            result["fused"] = fused

            r_fpn_all = [self._merge_fpn(r) for r in all_r_fpn]
            result["ref_merged"] = sum(r_fpn_all) / n_refs  # type: ignore[arg-type]

        return result

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor],
        return_cam: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: (B, 3, H, W) query images
            reference: single (B, 3, H, W) tensor or list of N such tensors
            return_cam: if True, also return ADPL-CAM maps

        Returns:
            logits: (B, num_classes) classification logits
            cam (optional): (B, num_classes, Hf, Wf) class activation maps
        """
        if isinstance(reference, torch.Tensor):
            references = [reference]
        else:
            references = reference

        q_fpn = self._get_fpn_features(query)
        all_r_fpn = [self._get_fpn_features(ref) for ref in references]

        fused = self._merge_and_fuse(q_fpn, all_r_fpn)

        pooled = fused.mean(dim=[2, 3])
        logits = self.classifier(pooled)

        if not return_cam:
            return logits

        cam = F.relu(torch.einsum("nc,bchw->bnhw", self.classifier.weight, fused))
        return logits, cam
