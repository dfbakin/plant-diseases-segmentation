"""MSCAN (Multi-Scale Convolutional Attention Network) backbone.

Pure PyTorch implementation without mmcv dependencies.
Reference: https://arxiv.org/abs/2209.08575
"""

import math
from typing import Literal

import torch
import torch.nn as nn
from timm.layers import DropPath


class DWConv(nn.Module):
    """Depthwise convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):
    """MLP with depthwise convolution."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    """Stem convolution for initial feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class AttentionModule(nn.Module):
    """Multi-scale convolutional attention module."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Local context
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # Multi-scale strip convolutions
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    """Spatial attention with gating mechanism."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    """MSCAN transformer block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        x = x.view(b, c, n).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding."""

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


# Variant configurations
MSCANVariant = Literal["tiny", "small", "base", "large"]

MSCAN_CONFIGS: dict[MSCANVariant, dict] = {
    "tiny": {
        "embed_dims": [32, 64, 160, 256],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 3, 5, 2],
        "drop_path_rate": 0.1,
    },
    "small": {
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [2, 2, 4, 2],
        "drop_path_rate": 0.1,
    },
    "base": {
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 3, 12, 3],
        "drop_path_rate": 0.1,
    },
    "large": {
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [3, 5, 27, 3],
        "drop_path_rate": 0.3,
    },
}


class MSCAN(nn.Module):
    """Multi-Scale Convolutional Attention Network backbone.

    A hierarchical vision backbone that uses multi-scale convolutional
    attention for efficient feature extraction.

    Attributes:
        embed_dims: Channel dimensions for each stage.
        depths: Number of blocks in each stage.
        num_stages: Number of hierarchical stages.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: list[int] | None = None,
        mlp_ratios: list[float] | None = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        depths: list[int] | None = None,
        num_stages: int = 4,
    ) -> None:
        """Initialize MSCAN backbone.

        Args:
            in_chans: Number of input channels.
            embed_dims: Channel dimensions for each stage.
            mlp_ratios: MLP expansion ratios for each stage.
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth rate.
            depths: Number of blocks in each stage.
            num_stages: Number of hierarchical stages.
        """
        super().__init__()

        # Default values
        if embed_dims is None:
            embed_dims = [64, 128, 320, 512]
        if mlp_ratios is None:
            mlp_ratios = [8, 8, 4, 4]
        if depths is None:
            depths = [3, 3, 12, 3]

        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature maps from each stage.
        """
        b = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, h, w = patch_embed(x)
            for blk in block:
                x = blk(x, h, w)
            x = norm(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    @classmethod
    def from_variant(cls, variant: MSCANVariant, **kwargs) -> "MSCAN":
        """Create MSCAN from a predefined variant.

        Args:
            variant: One of 'tiny', 'small', 'base', 'large'.
            **kwargs: Additional arguments to override config.

        Returns:
            Configured MSCAN instance.
        """
        if variant not in MSCAN_CONFIGS:
            raise ValueError(
                f"Unknown variant: {variant}. Use one of {list(MSCAN_CONFIGS.keys())}"
            )

        config = MSCAN_CONFIGS[variant].copy()
        config.update(kwargs)
        return cls(**config)

