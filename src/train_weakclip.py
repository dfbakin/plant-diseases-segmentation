"""Train WeakCLIP on pseudo masks from the CAM refinement pipeline.

Dataset-agnostic: reads class names from a text file, images and masks
from explicit directories. Works with any dataset that went through
export_labels -> generate_cams -> apply_crf -> train_psa -> random_walk.

Example:
    python src/train_weakclip.py \
        class_names_file=outputs/labels/voc_classes.txt \
        train_image_dir=data/VOC2012/JPEGImages \
        train_mask_dir=outputs/pseudo_masks
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
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.voc_wsss import WSSDataset
from src.wsss.weakclip.lightning import WeakCLIPModule
from src.wsss.weakclip.model import WeakCLIP

log = logging.getLogger(__name__)


@dataclass
class WeakCLIPTrainConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    class_names_file: str = "outputs/labels/voc_classes.txt"

    train_image_dir: str = "data/VOC2012/JPEGImages"
    train_mask_dir: str = "outputs/pseudo_masks"
    val_image_dir: str = "data/VOC2012/JPEGImages"
    val_mask_dir: str = "data/VOC2012/SegmentationClassAug"
    val_names_file: str = "data/VOC2012/ImageSets/Segmentation/val.txt"
    image_ext: str = ".jpg"

    clip_pretrained: str = "pretrained/ViT-B-16.pt"
    context_length: int = 5
    num_classes: int = 0
    image_size: int = 512
    tau: float = 0.07

    batch_size: int = 8
    max_epochs: int = 60
    learning_rate: float = 2e-4
    weight_decay: float = 3e-5
    warmup_iters: int = 1500
    poly_power: float = 0.9
    identity_loss_weight: float = 0.4
    num_workers: int = 8
    precision: str = "16-mixed"
    seed: int = 0

    limit_val_batches: int | float = 1.0

    experiment_name: str = "weakclip"
    output_dir: str = "outputs/weakclip"

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "${experiment_name}"


cs = ConfigStore.instance()
cs.store(name="weakclip_train_config", node=WeakCLIPTrainConfig)


def load_class_names(path: str | Path) -> tuple[str, ...]:
    """Load class names from text file (one name per line)."""
    lines = Path(path).read_text().strip().splitlines()
    names = tuple(line.strip() for line in lines if line.strip())
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def tokenize_class_names(
    class_names: tuple[str, ...], context_length: int = 5
) -> torch.LongTensor:
    """Tokenize class names using CLIP tokenizer."""
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(list(class_names))
    return tokens[:, :context_length].long()


def build_weakclip_model(
    cfg: WeakCLIPTrainConfig, class_names: tuple[str, ...]
) -> WeakCLIP:
    """Build WeakCLIP model with CLIP pretrained weights."""
    class_tokens = tokenize_class_names(class_names, cfg.context_length)

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
        new_k = k[len(prefix):]
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

    class_names = load_class_names(cfg.class_names_file)
    log.info(f"Loaded {len(class_names)} class names from {cfg.class_names_file}")

    if cfg.num_classes == 0:
        cfg.num_classes = len(class_names)
        log.info(f"Auto-set num_classes={cfg.num_classes} from class names file")
    elif cfg.num_classes != len(class_names):
        raise ValueError(
            f"num_classes={cfg.num_classes} does not match "
            f"{len(class_names)} names in {cfg.class_names_file}"
        )

    model = build_weakclip_model(cfg, class_names)

    train_ds = WSSDataset(
        image_dir=cfg.train_image_dir,
        mask_dir=cfg.train_mask_dir,
        image_ext=cfg.image_ext,
        image_size=cfg.image_size,
        is_train=True,
    )
    log.info(f"Train set: {len(train_ds)} images from {cfg.train_mask_dir}")

    steps_per_epoch = len(train_ds) // cfg.batch_size
    total_iters = steps_per_epoch * cfg.max_epochs
    log.info(f"LR schedule: {total_iters} total iters, {cfg.warmup_iters} warmup, poly^{cfg.poly_power}")

    lit_module = WeakCLIPModule(
        model=model,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_iters=cfg.warmup_iters,
        poly_power=cfg.poly_power,
        total_iters=total_iters,
        identity_loss_weight=cfg.identity_loss_weight,
    )

    val_loader = None
    if cfg.val_mask_dir and Path(cfg.val_mask_dir).exists():
        val_ds = WSSDataset(
            image_dir=cfg.val_image_dir,
            mask_dir=cfg.val_mask_dir,
            image_ext=cfg.image_ext,
            image_size=cfg.image_size,
            is_train=False,
        )
        if cfg.val_names_file and Path(cfg.val_names_file).exists():
            allowed = set(
                l.strip()
                for l in Path(cfg.val_names_file).read_text().splitlines()
                if l.strip()
            )
            val_ds.names = [n for n in val_ds.names if n in allowed]
            log.info(f"Filtered val to {len(val_ds.names)} images via {cfg.val_names_file}")
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        log.info(f"Val set: {len(val_ds)} images from {cfg.val_mask_dir}")
    else:
        log.warning("No val_mask_dir provided or directory missing, training without validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"weakclip_{cfg.image_size}_{cfg.seed}",
        tags={
            "model": "weakclip",
            "image_size": str(cfg.image_size),
            "num_classes": str(cfg.num_classes),
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    monitor = "val/mIoU" if val_loader is not None else "train/loss"
    monitor_mode = "max" if monitor == "val/mIoU" else "min"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="weakclip-{epoch:02d}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=5,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        precision=cfg.precision,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=50,
        limit_val_batches=cfg.limit_val_batches,
    )

    trainer.fit(lit_module, train_loader, val_loader)
    log.info(f"Training complete. Checkpoints at {output_dir / 'checkpoints'}")


@hydra.main(version_base=None, config_name="weakclip_train_config")
def main(cfg: DictConfig) -> None:
    train_weakclip(cfg)


if __name__ == "__main__":
    main()
