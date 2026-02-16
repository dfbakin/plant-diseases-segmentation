"""CLIP ViT-B/16 vision backbone with FPN adapters for WeakCLIP.

Extracted from WeakCLIP/weakclip/models.py CLIPVisionTransformer.
Stripped mmseg BaseBackbone; pure nn.Module. Loads weights via open_clip.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """LayerNorm that handles fp16 by casting to fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = nn.Identity()
        if drop_path > 0.0:
            from timm.layers import DropPath

            self.drop_path = DropPath(drop_path)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class CLIPVisionTransformer(nn.Module):
    """CLIP ViT backbone with FPN adapters for multi-scale features + embeddings."""

    def __init__(
        self,
        input_resolution: int = 512,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
        drop_path_rate: float = 0.0,
        out_indices: list[int] | None = None,
        get_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        if out_indices is None:
            out_indices = [3, 5, 7, 11]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, drop_path=dpr[i]) for i in range(layers)]
        )

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.fpn1 = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.GroupNorm(1, width)
        self.fpn4 = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (feat_1/4, feat_1/8, feat_1/16, feat_1/32, [global_emb, visual_emb])."""
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)

        cls_emb = self.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, C, dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_emb, x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:, :].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj
            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            features.append([global_embedding, visual_embedding])

        return tuple(features)
