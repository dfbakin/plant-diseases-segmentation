"""Train WeakCLIP on pseudo masks from MCTformer + CRF + PSA pipeline.

Loads CLIP ViT-B/16 pretrained backbone, freezes it and the text encoder,
trains context decoder + FPN + decode head with seeding + identity loss.

Example:
    python src/train_weakclip.py pseudo_mask_dir=outputs/pseudo_masks
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import open_clip
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.voc_wsss import VOCWSSDataset
from src.wsss.weakclip.lightning import WeakCLIPModule
from src.wsss.weakclip.model import WeakCLIP

log = logging.getLogger(__name__)

VOC_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
)


@dataclass
class WeakCLIPTrainConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    voc_root: str = "data/VOC2012"
    pseudo_mask_dir: str = "outputs/pseudo_masks"
    train_split: str = "train_aug_id"
    val_split: str = "val"

    clip_pretrained: str = "pretrained/ViT-B-16.pt"
    context_length: int = 5
    num_classes: int = 21
    image_size: int = 512
    tau: float = 0.07

    batch_size: int = 8
    max_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 3e-5
    warmup_iters: int = 1500
    identity_loss_weight: float = 0.4
    num_workers: int = 8
    precision: str = "16-mixed"
    seed: int = 0

    experiment_name: str = "weakclip_voc"
    output_dir: str = "outputs/weakclip"


cs = ConfigStore.instance()
cs.store(name="weakclip_train_config", node=WeakCLIPTrainConfig)


def tokenize_class_names(class_names: tuple[str, ...], context_length: int = 5) -> torch.LongTensor:
    """Tokenize VOC class names using CLIP tokenizer (context_length tokens each)."""
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(list(class_names))
    return tokens[:, :context_length].long()


def build_weakclip_model(cfg: WeakCLIPTrainConfig) -> WeakCLIP:
    """Build WeakCLIP model with CLIP pretrained weights."""
    class_tokens = tokenize_class_names(VOC_CLASSES, cfg.context_length)

    model = WeakCLIP(
        num_classes=cfg.num_classes,
        class_tokens=class_tokens,
        context_length=cfg.context_length,
        backbone_cfg={
            "input_resolution": cfg.image_size,
            "patch_size": 16,
            "width": 768,
            "layers": 12,
            "heads": 12,
            "output_dim": 512,
            "drop_path_rate": 0.1,
            "get_embeddings": True,
        },
        text_encoder_cfg={
            "context_length": 13,
            "embed_dim": 512,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12,
        },
        context_decoder_cfg={
            "transformer_width": 256,
            "transformer_heads": 4,
            "transformer_layers": 3,
            "visual_dim": 512,
            "dropout": 0.1,
        },
        fpn_cfg={
            "in_channels": [768 + cfg.num_classes] * 4,
            "out_channels": 256,
            "num_outs": 4,
        },
        decode_head_cfg={
            "in_channels": [256, 256, 256, 256],
            "channels": 256,
            "num_classes": cfg.num_classes,
            "feature_strides": [4, 8, 16, 32],
            "feature_size": (cfg.image_size // 8, cfg.image_size // 8),
            "dropout": 0.1,
        },
        tau=cfg.tau,
        score_concat_index=2,
        if_decouple=True,
        if_pyramid_queried_feature=True,
    )

    # Load CLIP pretrained weights for backbone + text encoder
    clip_path = Path(cfg.clip_pretrained)
    if clip_path.exists():
        log.info(f"Loading CLIP weights from {clip_path}")
        sd = torch.load(str(clip_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        backbone_mapping = _map_clip_to_backbone(sd, model.backbone.state_dict())
        model.backbone.load_state_dict(backbone_mapping, strict=False)

        text_mapping = _map_clip_to_text_encoder(sd, model.text_encoder.state_dict())
        model.text_encoder.load_state_dict(text_mapping, strict=False)
        log.info("Loaded CLIP backbone + text encoder weights")
    else:
        log.warning(f"CLIP weights not found at {clip_path}, training from scratch")

    return model


def _map_clip_to_backbone(clip_sd: dict, backbone_sd: dict) -> dict:
    """Map CLIP state_dict keys to CLIPVisionTransformer keys."""
    mapped = {}
    prefix = "visual."
    for k, v in clip_sd.items():
        if not k.startswith(prefix):
            continue
        new_k = k[len(prefix) :]
        new_k = new_k.replace("transformer.resblocks", "resblocks")
        if new_k in backbone_sd and v.shape == backbone_sd[new_k].shape:
            mapped[new_k] = v
    return mapped


def _map_clip_to_text_encoder(clip_sd: dict, te_sd: dict) -> dict:
    """Map CLIP state_dict keys to CLIPTextContextEncoder keys.

    open_clip uses 'transformer.resblocks.N.*', our encoder uses 'transformer.N.*'.
    """
    mapped = {}
    rename = {
        "token_embedding.weight": "token_embedding.weight",
        "positional_embedding": "positional_embedding",
        "ln_final.weight": "ln_final.weight",
        "ln_final.bias": "ln_final.bias",
        "text_projection": "text_projection",
    }
    for ck, tk in rename.items():
        if ck in clip_sd and tk in te_sd:
            v = clip_sd[ck]
            if tk == "positional_embedding":
                v = v[: te_sd[tk].shape[0]]
            if tk == "text_projection":
                v = v.T if v.shape != te_sd[tk].shape else v
            if v.shape == te_sd[tk].shape:
                mapped[tk] = v

    for k, v in clip_sd.items():
        if k.startswith("transformer.resblocks."):
            new_k = k.replace("transformer.resblocks.", "transformer.")
            if new_k in te_sd and v.shape == te_sd[new_k].shape:
                mapped[new_k] = v
    return mapped


def train_weakclip(cfg: WeakCLIPTrainConfig) -> None:
    L.seed_everything(cfg.seed)
    model = build_weakclip_model(cfg)

    lit_module = WeakCLIPModule(
        model=model,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_iters=cfg.warmup_iters,
        identity_loss_weight=cfg.identity_loss_weight,
    )

    train_ds = VOCWSSDataset(
        root=cfg.voc_root,
        pseudo_mask_dir=cfg.pseudo_mask_dir,
        split=cfg.train_split,
        image_size=cfg.image_size,
    )
    val_ds = VOCWSSDataset(
        root=cfg.voc_root,
        pseudo_mask_dir="SegmentationClassAug",
        split=cfg.val_split,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="weakclip-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        precision=cfg.precision,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=50,
    )

    log.info(f"Training WeakCLIP: {len(train_ds)} train, {len(val_ds)} val images")
    trainer.fit(lit_module, train_loader, val_loader)
    log.info(f"Training complete. Checkpoints at {output_dir / 'checkpoints'}")


@hydra.main(version_base=None, config_name="weakclip_train_config")
def main(cfg: DictConfig) -> None:
    train_weakclip(cfg)


if __name__ == "__main__":
    main()
