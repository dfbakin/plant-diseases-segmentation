"""Context decoder: cross-attention between text and visual features.

Extracted from WeakCLIP/weakclip/models.py ContextDecoder.
Pure nn.Module, no mmseg dependency.
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class Attention(nn.Module):
    """Multi-head cross-attention with separate Q/K/V projections."""

    def __init__(self, dim: int, num_heads: int = 8, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = q.shape
        M = k.shape[1]
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum("bnkc,bmkc->bknm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum("bknm,bmkc->bnkc", attn, v).reshape(B, N, C)

        return self.proj_drop(self.proj(x))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    """Cross-attention decoder refining text embeddings via visual context."""

    def __init__(
        self,
        transformer_width: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        visual_dim: int = 512,
        dropout: float = 0.1,
        if_decouple: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(transformer_width, transformer_heads, dropout)
                for _ in range(transformer_layers)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim),
        )

        self.if_decouple = if_decouple
        if if_decouple:
            self.anti_decoder = nn.ModuleList(
                [
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout)
                    for _ in range(transformer_layers)
                ]
            )
            self.anti_out_proj = nn.Sequential(
                nn.LayerNorm(transformer_width),
                nn.Linear(transformer_width, visual_dim),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        visual = self.memory_proj(visual)
        text = self.text_proj(text)

        for layer in self.decoder:
            text_diff = layer(text, visual)

        if self.if_decouple:
            for layer in self.anti_decoder:
                visual_diff = layer(visual, text)
            return self.out_proj(text_diff), self.anti_out_proj(visual_diff)

        return self.out_proj(text_diff)
