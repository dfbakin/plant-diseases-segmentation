"""WeakCLIP WSSS training configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeakCLIPBackboneConfig:
    input_resolution: int = 512
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    output_dim: int = 512
    drop_path_rate: float = 0.1
    get_embeddings: bool = True


@dataclass
class WeakCLIPTextEncoderConfig:
    context_length: int = 13
    embed_dim: int = 512
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12


@dataclass
class WeakCLIPContextDecoderConfig:
    transformer_width: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 3
    visual_dim: int = 512
    dropout: float = 0.1


@dataclass
class WeakCLIPConfig:
    num_classes: int = 21
    context_length: int = 5
    tau: float = 0.07
    if_decouple: bool = True
    if_pyramid_queried_feature: bool = True
    fpn_out_channels: int = 256
    decode_feature_size: int = 64
    backbone: WeakCLIPBackboneConfig = field(default_factory=WeakCLIPBackboneConfig)
    text_encoder: WeakCLIPTextEncoderConfig = field(
        default_factory=WeakCLIPTextEncoderConfig,
    )
    context_decoder: WeakCLIPContextDecoderConfig = field(
        default_factory=WeakCLIPContextDecoderConfig,
    )


@dataclass
class WSSTrainerConfig:
    max_steps: int = 20000
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: int = 2000


@dataclass
class WSSDataConfig:
    voc_root: str = "data/VOC2012"
    pseudo_mask_dir: str = "SegmentationClassAugPseudoMaskMCT"
    train_split: str = "train_aug_id"
    val_split: str = "val"
    image_size: int = 512
    batch_size: int = 8
    num_workers: int = 8
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class WSSConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])
    experiment_name: str = "weakclip_voc"
    seed: int = 42
    model: WeakCLIPConfig = field(default_factory=WeakCLIPConfig)
    data: WSSDataConfig = field(default_factory=WSSDataConfig)
    trainer: WSSTrainerConfig = field(default_factory=WSSTrainerConfig)
    learning_rate: float = 1e-4
    weight_decay: float = 3e-5
    warmup_iters: int = 1500
    identity_loss_weight: float = 0.4
    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
