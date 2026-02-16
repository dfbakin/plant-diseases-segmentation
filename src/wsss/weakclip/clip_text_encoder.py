"""CLIP text encoder with learnable context prompts for WeakCLIP.

Extracted from WeakCLIP/weakclip/models.py CLIPTextContextEncoder.
Stripped mmseg registration; pure nn.Module.
"""

import torch
import torch.nn as nn

from src.wsss.weakclip.clip_backbone import LayerNorm, ResidualAttentionBlock


class CLIPTextContextEncoder(nn.Module):
    """CLIP text encoder with learnable context token injection."""

    def __init__(
        self,
        context_length: int = 13,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask=mask)
                for _ in range(transformer_layers)
            ]
        )
        self.transformer = resblocks

    def forward(
        self,
        text: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (B, K, embed_dim) if context provided, else (K, embed_dim)."""
        if context is not None:
            x_text = self.token_embedding(text)
            K, N1, C = x_text.shape
            B, N2, C = context.shape

            eos_indx = text.argmax(dim=-1) + N2
            eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

            x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

            x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2)
            x = x.reshape(B * K, N1 + N2, C)

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
            x = x.reshape(B, K, self.embed_dim)
        else:
            x = self.token_embedding(text)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
