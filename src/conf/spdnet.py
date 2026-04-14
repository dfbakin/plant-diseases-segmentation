"""Hydra configuration dataclasses for SPDNet Siamese training."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SPDNetModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    fpn_channels: int = 256
    num_classes: int = 115
    input_size: int = 448
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    mse_reduction: int = 4


@dataclass
class SPDNetDataConfig:
    root: str = "data/plantsegv3"
    pv_root: str = "data/plant-village"
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    include_plantvillage: bool = False
    num_references: int = 1
    augmentation: str = "heavy"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class SPDNetTrainerConfig:
    max_epochs: int = 80
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-5


@dataclass
class SPDNetConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    experiment_name: str = "spdnet_plantseg"
    seed: int = 42

    model: SPDNetModelConfig = field(default_factory=SPDNetModelConfig)
    data: SPDNetDataConfig = field(default_factory=SPDNetDataConfig)
    trainer: SPDNetTrainerConfig = field(default_factory=SPDNetTrainerConfig)

    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "${experiment_name}"
    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
