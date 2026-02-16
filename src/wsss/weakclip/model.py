"""WeakCLIP: main model combining CLIP backbone, text encoder, and decoder.

Extracted from WeakCLIP/weakclip/weakclip.py.
Stripped mmseg BaseSegmentor; pure nn.Module with explicit forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wsss.weakclip.clip_backbone import CLIPVisionTransformer
from src.wsss.weakclip.clip_text_encoder import CLIPTextContextEncoder
from src.wsss.weakclip.context_decoder import ContextDecoder
from src.wsss.weakclip.decode_head import FPNDecodeHead
from src.wsss.weakclip.fpn import FPN


class WeakCLIP(nn.Module):
    """WeakCLIP: CLIP-guided weakly-supervised semantic segmentation.

    Backbone -> text-pixel matching via context decoder -> FPN -> decode head.
    """

    def __init__(
        self,
        num_classes: int = 21,
        class_tokens: torch.Tensor | None = None,
        context_length: int = 5,
        backbone_cfg: dict | None = None,
        text_encoder_cfg: dict | None = None,
        context_decoder_cfg: dict | None = None,
        fpn_cfg: dict | None = None,
        decode_head_cfg: dict | None = None,
        tau: float = 0.07,
        score_concat_index: int = 2,
        if_decouple: bool = True,
        if_pyramid_queried_feature: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.score_concat_index = score_concat_index
        self.if_decouple = if_decouple
        self.if_pyramid_queried_feature = if_pyramid_queried_feature

        _bb = backbone_cfg or {}
        self.backbone = CLIPVisionTransformer(**_bb)

        _te = text_encoder_cfg or {}
        self.text_encoder = CLIPTextContextEncoder(**_te)

        _cd = context_decoder_cfg or {}
        _cd["if_decouple"] = if_decouple
        self.context_decoder = ContextDecoder(**_cd)

        text_ctx_length = self.text_encoder.context_length - context_length
        token_embed_dim = self.text_encoder.embed_dim
        self.contexts = nn.Parameter(torch.randn(1, text_ctx_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        self.gamma = nn.Parameter(torch.ones(token_embed_dim) * 1e-1)
        if if_decouple:
            self.beta = nn.Parameter(torch.ones(token_embed_dim) * 1e-1)

        _fpn = fpn_cfg or {}
        self.neck = FPN(**_fpn)

        _dh = decode_head_cfg or {}
        self.decode_head = FPNDecodeHead(**_dh)

        if class_tokens is not None:
            self.register_buffer("texts", class_tokens)
        else:
            self.register_buffer(
                "texts",
                torch.zeros(0, dtype=torch.long),
                persistent=False,
            )

    def set_class_tokens(self, tokens: torch.Tensor) -> None:
        self.texts = tokens

    def after_extract_feat(
        self,
        x: tuple,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Text-pixel matching with context decoder refinement."""
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape

        visual_context = torch.cat(
            [global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
            dim=2,
        ).permute(0, 2, 1)

        text_embeddings = self.text_encoder(
            self.texts.to(global_feat.device),
            self.contexts,
        ).expand(B, -1, -1)

        if self.if_decouple:
            text_diff, visual_diff = self.context_decoder(
                text_embeddings,
                visual_context,
            )
            visual_context = visual_context + self.beta * visual_diff
            visual_embeddings = visual_context[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
        else:
            text_diff = self.context_decoder(text_embeddings, visual_context)

        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum("bchw,bkc->bkhw", visual_embeddings, text_norm)

        if self.if_pyramid_queried_feature:
            for i in range(len(x_orig)):
                score_map_i = F.interpolate(
                    score_map,
                    size=x_orig[i].shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                x_orig[i] = torch.cat([x_orig[i], score_map_i], dim=1)
        else:
            x_orig[self.score_concat_index] = torch.cat(
                [x_orig[self.score_concat_index], score_map],
                dim=1,
            )

        return text_embeddings, x_orig, score_map

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (seg_logits, score_map)."""
        x = self.backbone(img)
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)
        x_orig = self.neck(x_orig)
        seg_logits = self.decode_head(x_orig)
        return seg_logits, score_map
