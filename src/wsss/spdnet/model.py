"""SPDNet model: Siamese ResNet50 + FPN + MSE + ADPL-CAM.

Architecture:
  - Shared ResNet50 backbone extracts multi-scale features (layer1-4)
  - FPN merges them into a common 256-ch representation
  - MSE (Multi-Scale Excitation) applies channel attention
  - Reference fusion: either token-based (ADPL-CAM) or spatial cross-attention

Fusion modes:
  - "token": original ADPL-CAM with GlobalMaxPool tokenisation (spatially blind)
  - "spatial": multi-head cross-attention preserving spatial structure
"""

from __future__ import annotations

import math

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


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention fusion: query locations attend to reference locations.

    Instead of collapsing reference features to a single vector (GlobalMaxPool),
    this module preserves spatial structure by pooling the reference to a small
    grid and applying multi-head cross-attention from query to reference.
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        ref_pool_size: int = 14,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ref_pool_size = ref_pool_size
        self.pool = nn.AdaptiveAvgPool2d((ref_pool_size, ref_pool_size))
        self.cross_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse reference spatial features into query via cross-attention.

        Args:
            query_feat: ``(B, C, H, W)`` merged query FPN features.
            ref_feat:   ``(B, C, Hr, Wr)`` merged reference FPN features.
            return_attn: if ``True``, also return a per-query attention
                concentration map ``(B, H, W)`` in ``[0, 1]``: ``0`` when
                the query attends uniformly across all reference keys,
                ``1`` when its attention is perfectly peaked on a single
                key. This is the ``M(q, r)`` map consumed by the
                equivariance loss and the P6 probe input.

        Returns:
            ``(B, C, H, W)`` fused query, or ``(fused, attn_BHW)`` when
            ``return_attn=True``.
        """
        B, C, H, W = query_feat.shape

        ref_pooled = self.pool(ref_feat)  # (B, C, h, w)

        q = query_feat.flatten(2).permute(0, 2, 1)         # (B, HW, C)
        kv = ref_pooled.flatten(2).permute(0, 2, 1)        # (B, h*w, C)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Main forward: regular MHA preserves attention-weight dropout in
        # train mode so ``fused`` retains its usual regularisation.
        attended, _ = self.cross_attn(q, kv, kv)
        attended = attended.permute(0, 2, 1).view(B, C, H, W)
        fused = query_feat + self.gate * attended

        if not return_attn:
            return fused

        # Second MHA call dedicated to the attention map. Dropout is
        # disabled so each per-query softmax row remains a valid
        # probability distribution (PyTorch's MHA applies dropout AFTER
        # softmax which can drive row sums and entropy out of range, and
        # it would also make ``M(q, r) != M(q, r)`` across calls with the
        # same input -- killing the identity case of L_eq).
        #
        # Cost: one extra MHA forward when ``return_attn=True`` (only
        # during equivariance training and probe extraction). Gradients
        # still flow through the in-proj weights via this call.
        saved_train = self.cross_attn.training
        self.cross_attn.eval()
        try:
            _, attn_w = self.cross_attn(
                q, kv, kv, need_weights=True, average_attn_weights=True,
            )
        finally:
            self.cross_attn.train(saved_train)

        # attn_w: (B, H*W, h_ref*w_ref) is a softmax distribution over
        # keys PER query position. Its mean over keys is mathematically
        # constant (= 1/(h_ref*w_ref)) for every query, which makes the
        # equivariance loss identically zero. Use normalised attention
        # concentration instead: ``1 - H(p) / log N``, in ``[0, 1]`` per
        # query, with 0 for uniform attention and 1 for perfectly peaked.
        # This is query-position-indexed and structurally equivariant
        # under spatial query reorderings (MHA is permutation-equivariant
        # in queries), and gradients flow through every key via the
        # entropy term.
        log_N = math.log(attn_w.shape[-1])
        attn_p = attn_w.clamp_min(1e-12)
        neg_ent = (attn_p * attn_p.log()).sum(dim=-1)        # (B, H*W) in [-log N, 0]
        attn_map = (1.0 + neg_ent / log_N).view(B, H, W)     # (B, H, W) in [0, 1]
        return fused, attn_map


class SPDNet(nn.Module):
    """Siamese Plant Disease Network.

    Shared ResNet50 backbone + FPN + MSE + configurable reference fusion.

    Args:
        fusion_mode: ``"token"`` for original ADPL-CAM (GlobalMaxPool),
            ``"spatial"`` for multi-head cross-attention.
    """

    def __init__(
        self,
        num_classes: int = 115,
        fpn_channels: int = 256,
        mse_reduction: int = 4,
        pretrained: bool = True,
        fusion_mode: str = "token",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.fusion_mode = fusion_mode

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

        if fusion_mode == "token":
            self.adpl_cam = ADPLCam(num_levels=ADPL_CAM_LEVELS)
        elif fusion_mode == "spatial":
            self.spatial_attn = SpatialCrossAttention(channels=fpn_channels)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from shared backbone."""
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

    def extract_probe_features(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return intermediate activations at the six SPDNet probe positions.

        Used by the seg-probe diagnostic only. Mirrors ``extract_merged_features``
        but additionally exposes the C5 backbone output (P1) and the high-res
        FPN-P2 level (P2) and, for the spatial fusion, the head-averaged
        attention map (P6) and the classifier-projected fused CAM (P5).

        Returns dict with keys (some absent for token mode / when no reference):
            P1_layer4:      (B, 2048, H/32, W/32)  -- C5 of ResNet50 backbone
            P2_fpn_p2:      (B, C_fpn, H/8, W/8)   -- highest-res FPN level *after MSE*
            P3_query_merged:(B, C_fpn, H/8, W/8)   -- "Point A" pre-fusion
            P4_fused:       (B, C_fpn, H/8, W/8)   -- "Point B" post-fusion (refs needed)
            P5_cam_classifier:(B, num_classes, H/8, W/8) -- "Point C" classifier@fused
            P6_attn_map:    (B, 1, H/8, W/8)       -- per-query attention concentration in [0, 1]
        """
        feats = self.extract_features(query)
        c5 = feats[-1]
        fpn_out = self.fpn(feats)
        mse_out = [self.mse(p) for p in fpn_out]
        query_merged = self._merge_fpn(mse_out)

        result: dict[str, torch.Tensor] = {
            "P1_layer4": c5,
            "P2_fpn_p2": mse_out[0],
            "P3_query_merged": query_merged,
        }

        if reference is None:
            return result

        refs = [reference] if isinstance(reference, torch.Tensor) else reference
        all_r_fpn = [self._get_fpn_features(r) for r in refs]
        n_refs = len(all_r_fpn)

        if self.fusion_mode == "token":
            avg_tokens: list[torch.Tensor] = []
            for lvl in range(len(mse_out)):
                lvl_tokens = [self.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
                avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]
            fused = self.adpl_cam.fuse(query_merged, avg_tokens)
            attn_map = None
        else:
            ref_merged_list = [self._merge_fpn(r) for r in all_r_fpn]
            ref_merged = sum(ref_merged_list) / n_refs  # type: ignore[arg-type]
            fused, attn_map = self._spatial_attn_with_map(query_merged, ref_merged)

        result["P4_fused"] = fused
        cam_cls = F.relu(
            torch.einsum("nc,bchw->bnhw", self.classifier.weight, fused)
        )
        result["P5_cam_classifier"] = cam_cls
        if attn_map is not None:
            result["P6_attn_map"] = attn_map
        return result

    def _spatial_attn_with_map(
        self, query_feat: torch.Tensor, ref_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run spatial cross-attention and return ``(fused, attn (B, 1, H, W))``.

        Thin wrapper around ``SpatialCrossAttention.forward(..., return_attn=True)``
        that adds a singleton channel dim so the resulting heatmap is a valid
        probe input ``(B, 1, H, W)``.
        """
        fused, attn_bhw = self.spatial_attn(query_feat, ref_feat, return_attn=True)
        return fused, attn_bhw.unsqueeze(1)

    def attention_map(
        self,
        query: torch.Tensor,
        reference: torch.Tensor | list[torch.Tensor] | None = None,
        ref_merged_cached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the per-query attention concentration map ``M(q, r)``.

        Returned tensor has shape ``(B, H, W)`` with values in ``[0, 1]``:
        0 = query attends uniformly across all reference keys,
        1 = query has a perfectly peaked attention on a single key.
        This is the canonical attention map fed to ``equivariance_loss``.

        Spatial fusion mode only: raises ``RuntimeError`` for token mode
        because there is no attention map to expose.

        Args:
            query: ``(B, 3, H, W)`` query images.
            reference: optional reference image(s); ignored when
                ``ref_merged_cached`` is provided.
            ref_merged_cached: optional pre-computed merged reference FPN
                tensor ``(B, C, Hf, Wf)``. Pass this from the LightningModule
                to avoid re-running the reference backbone + FPN when
                computing both ``M(q, r)`` and ``M(T(q), r)`` within the same
                training step.
        """
        if self.fusion_mode != "spatial":
            raise RuntimeError(
                f"attention_map() requires fusion_mode='spatial' "
                f"(got {self.fusion_mode!r})"
            )

        q_fpn = self._get_fpn_features(query)
        query_merged = self._merge_fpn(q_fpn)

        if ref_merged_cached is not None:
            ref_merged = ref_merged_cached
        else:
            if reference is None:
                raise ValueError(
                    "attention_map() requires either `reference` or "
                    "`ref_merged_cached`"
                )
            refs = [reference] if isinstance(reference, torch.Tensor) else reference
            all_r_fpn = [self._get_fpn_features(r) for r in refs]
            r_merged_list = [self._merge_fpn(r) for r in all_r_fpn]
            ref_merged = sum(r_merged_list) / len(r_merged_list)  # type: ignore[arg-type]

        _, attn = self.spatial_attn(query_merged, ref_merged, return_attn=True)
        return attn  # (B, H, W)

    def _merge_and_fuse(
        self,
        q_fpn: list[torch.Tensor],
        all_r_fpn: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Merge query FPN levels and fuse with reference features."""
        query_merged = self._merge_fpn(q_fpn)

        if self.fusion_mode == "token":
            n_refs = len(all_r_fpn)
            avg_tokens: list[torch.Tensor] = []
            for lvl in range(len(q_fpn)):
                lvl_tokens = [self.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
                avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]
            return self.adpl_cam.fuse(query_merged, avg_tokens)

        # spatial: merge ref FPN levels, average across references
        ref_merged_list = [self._merge_fpn(r) for r in all_r_fpn]
        ref_merged = sum(ref_merged_list) / len(ref_merged_list)  # type: ignore[arg-type]
        return self.spatial_attn(query_merged, ref_merged)

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
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract intermediate feature maps for seed generation / aux losses.

        Returns dict with:
            ``query_merged``: ``(B, C, Hf, Wf)`` merged FPN features before fusion (P3).
            ``ref_merged``:   ``(B, C, Hf, Wf)`` merged reference FPN features (if ref given).
            ``fused``:        ``(B, C, Hf, Wf)`` features after fusion (P4) (if ref given).
            ``attn_map``:     ``(B, Hf, Wf)`` per-query attention concentration map
                in ``[0, 1]`` (only when ``return_attn=True`` AND
                ``fusion_mode='spatial'``).

        Note:
            ``return_attn`` is silently ignored for ``fusion_mode='token'`` (no
            attention map exists). The LightningModule only requests it on the
            spatial branch.
        """
        q_fpn = self._get_fpn_features(query)
        query_merged = self._merge_fpn(q_fpn)
        result: dict[str, torch.Tensor] = {"query_merged": query_merged}

        if reference is not None:
            refs = [reference] if isinstance(reference, torch.Tensor) else reference
            all_r_fpn = [self._get_fpn_features(r) for r in refs]
            n_refs = len(all_r_fpn)

            r_fpn_all = [self._merge_fpn(r) for r in all_r_fpn]
            ref_merged = sum(r_fpn_all) / n_refs  # type: ignore[arg-type]
            result["ref_merged"] = ref_merged

            if self.fusion_mode == "token":
                avg_tokens: list[torch.Tensor] = []
                for lvl in range(len(q_fpn)):
                    lvl_tokens = [self.adpl_cam.tokenize([r[lvl]])[0] for r in all_r_fpn]
                    avg_tokens.append(sum(lvl_tokens) / n_refs)  # type: ignore[arg-type]
                fused = self.adpl_cam.fuse(query_merged, avg_tokens)
            else:
                if return_attn:
                    fused, attn_bhw = self.spatial_attn(
                        query_merged, ref_merged, return_attn=True,
                    )
                    result["attn_map"] = attn_bhw
                else:
                    fused = self.spatial_attn(query_merged, ref_merged)

            result["fused"] = fused

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
