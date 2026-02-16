"""MCTformer-V2: Multi-Class Token Transformer for weakly-supervised CAM generation.

Ported from https://github.com/xulianuwa/MCTformer to modern Python 3.12 / timm >= 1.0.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """MLP block used in transformer layers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention that returns attention weights."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N
        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class Block(nn.Module):
    """Transformer block with attention weight output."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Custom Vision Transformer with attention weight output.

    This is the MCTformer-specific ViT, not timm's ViT. It returns
    attention weights from each block and supports position encoding
    interpolation for variable input sizes.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_classes=num_classes,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: torch.Tensor, n: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], attn_weights

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self, x: torch.Tensor, n: int = 12
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x_feat, attn_weights = self.forward_features(x, n)
        x_out = self.head(x_feat)
        if self.training:
            return x_out
        else:
            return x_out, attn_weights


class MCTformerPlus(VisionTransformer):
    """MCTformer-V2: Multi-Class Token Transformer.

    Extends VisionTransformer with per-class tokens instead of a single [CLS]
    token. Each class token attends to patch tokens, producing class-specific
    attention maps usable as CAMs for weakly-supervised segmentation.

    Key differences from standard ViT:
    - `num_classes` separate class tokens (instead of one [CLS])
    - Separate positional embeddings for class and patch tokens
    - Conv2d classification head on patch features (instead of linear on [CLS])
    - Weighted patch logits with exponential decay
    - Returns attention weights and per-layer class embeddings for loss computation
    """

    def __init__(
        self,
        decay_parameter: float = 0.996,
        input_size: int = 224,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed_cls, std=0.02)
        trunc_normal_(self.pos_embed_pat, std=0.02)

        if hasattr(self, "pos_embed"):
            del self.pos_embed

        self.decay_parameter = decay_parameter

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate patch positional embeddings for variable input sizes."""
        npatch = x.shape[1]  # x is pure patch embeddings (cls tokens not yet added)
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(
        self, x: torch.Tensor, n: int = 12
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Extract features with multi-class tokens.

        Returns:
            x_cls: (B, num_classes, embed_dim) class token outputs
            x_patch: (B, num_patches, embed_dim) patch token outputs
            attn_weights: list of (B, num_heads, N, N) attention weight tensors
            class_embeddings: list of (B, num_classes, embed_dim) per-layer class embeddings
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        attn_weights = []
        class_embeddings = []

        for blk in self.blocks:
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
            class_embeddings.append(x[:, 0 : self.num_classes])

        return (
            x[:, 0 : self.num_classes],
            x[:, self.num_classes :],
            attn_weights,
            class_embeddings,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_att: bool = False,
        n_layers: int = 12,
        attention_type: str = "fused",
    ) -> list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            return_att: If True, return CAMs and patch attention (for inference)
            n_layers: Number of last layers to average attention from
            attention_type: One of 'fused', 'patchcam', 'mct'

        Returns:
            If return_att=False (training):
                list of [cls_logits, all_cls_embeddings, patch_logits]
            If return_att=True (CAM generation):
                (combined_logits, cams, patch_attn)
        """
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, all_x_cls = self.forward_features(x)

        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.head(x_patch)

        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)

        sorted_patch_token, _indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(
            start=0,
            end=x_patch_flattened.size(-2) - 1,
            steps=x_patch_flattened.size(-2),
            base=self.decay_parameter,
            device=x_patch_flattened.device,
        )
        x_patch_logits = (
            torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2)
            / weights.sum()
        )

        x_cls_logits = x_cls.mean(-1)

        if not return_att:
            return [x_cls_logits, torch.stack(all_x_cls), x_patch_logits]

        feature_map = x_patch.detach().clone()
        feature_map = F.relu(feature_map)
        n, c, fh, fw = feature_map.shape

        attn_stack = torch.stack(attn_weights)
        attn_stack = torch.mean(attn_stack, dim=2)
        mtatt = (
            attn_stack[-n_layers:]
            .mean(0)[:, 0 : self.num_classes, self.num_classes :]
            .reshape([n, c, fh, fw])
        )
        patch_attn = attn_stack[:, :, self.num_classes :, self.num_classes :]

        if attention_type == "fused":
            cams = mtatt * feature_map
            cams = torch.sqrt(cams)
        elif attention_type == "patchcam":
            cams = feature_map
        elif attention_type == "mct":
            cams = mtatt
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        x_logits = (x_cls_logits + x_patch_logits) / 2
        return x_logits, cams, patch_attn


def create_mctformer_v2(
    num_classes: int = 20,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    input_size: int = 224,
    **kwargs,
) -> MCTformerPlus:
    """Create an MCTformer-V2 model.

    Args:
        num_classes: Number of output classes (20 for VOC, 80 for COCO)
        pretrained: If True and checkpoint_path is None, load DeiT-small pretrained
            weights from torch hub (ImageNet, excluding cls_token/pos_embed)
        checkpoint_path: Path to a pre-trained MCTformer checkpoint (.pth).
            If provided, loads full model state dict.
        input_size: Input image size for positional embedding initialization
        **kwargs: Additional arguments passed to MCTformerPlus

    Returns:
        MCTformerPlus model instance
    """
    model = MCTformerPlus(
        input_size=input_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        **kwargs,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        # Checkpoint uses combined pos_embed [1, num_classes+num_patches, D].
        # Split into pos_embed_cls and pos_embed_pat for this model.
        if "pos_embed" in state and "pos_embed_cls" not in state:
            combined = state.pop("pos_embed")
            state["pos_embed_cls"] = combined[:, :num_classes, :]
            state["pos_embed_pat"] = combined[:, num_classes:, :]
        # Filter out keys with shape mismatches
        model_state = model.state_dict()
        state = {
            k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(state, strict=False)
    elif pretrained:
        # Load DeiT-small pretrained weights (ImageNet)
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )["model"]
        model_dict = model.state_dict()
        # Remove head keys with shape mismatch
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in checkpoint and checkpoint[k].shape != model_dict.get(k, torch.empty(0)).shape:
                del checkpoint[k]

        # Replicate DeiT single cls_token → num_classes class tokens
        if "cls_token" in checkpoint:
            checkpoint["cls_token"] = checkpoint["cls_token"].repeat(1, num_classes, 1)

        # Interpolate DeiT pos_embed to MCTformer split format
        if "pos_embed" in checkpoint:
            pos_embed_ckpt = checkpoint.pop("pos_embed")
            embed_dim = pos_embed_ckpt.shape[-1]
            # DeiT has 1 cls token + patch tokens
            cls_pos = pos_embed_ckpt[:, :1, :].repeat(1, num_classes, 1)
            pat_pos = pos_embed_ckpt[:, 1:, :]
            orig_size = int(pat_pos.shape[1] ** 0.5)
            target_size = input_size // 16

            if orig_size != target_size:
                pat_pos = pat_pos.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
                pat_pos = torch.nn.functional.interpolate(
                    pat_pos, size=(target_size, target_size), mode="bicubic", align_corners=False
                )
                pat_pos = pat_pos.permute(0, 2, 3, 1).flatten(1, 2)

            checkpoint["pos_embed_cls"] = cls_pos
            checkpoint["pos_embed_pat"] = pat_pos

        pretrained_dict = {
            k: v
            for k, v in checkpoint.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
