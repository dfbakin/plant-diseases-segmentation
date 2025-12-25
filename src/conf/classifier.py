"""Classifier training configuration."""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ClassifierModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


@dataclass
class ClassifierDataConfig:
    plantvillage_root: str = "data/plant-village"
    plantseg_root: str = "data/plantsegv3"
    image_size: int = 384
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    use_weighted_sampler: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_samples: Optional[int] = None  # Limit samples per dataset (for testing)


@dataclass
class ClassifierTrainerConfig:
    max_epochs: int = 50
    min_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    early_stopping_patience: int = 10


@dataclass
class CAMConfig:
    enabled: bool = True
    method: str = "gradcam"
    threshold: float = 0.5


@dataclass
class ClassifierConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"augmentation": "spatial_color_light"},
        ]
    )

    experiment_name: str = "classifier_benchmark"
    seed: int = 42

    model: ClassifierModelConfig = field(default_factory=ClassifierModelConfig)
    data: ClassifierDataConfig = field(default_factory=ClassifierDataConfig)
    trainer: ClassifierTrainerConfig = field(default_factory=ClassifierTrainerConfig)
    cam: CAMConfig = field(default_factory=CAMConfig)
    augmentation: Any = MISSING

    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "${experiment_name}"

    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"


def register_classifier_configs() -> None:
    """Register classifier configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="classifier_config", node=ClassifierConfig)

    from src.conf.augmentation import (
        BaselineAugConfig,
        FullAugConfig,
        SpatialColorLightAugConfig,
        SpatialLightAugConfig,
    )

    cs.store(group="augmentation", name="baseline", node=BaselineAugConfig)
    cs.store(group="augmentation", name="spatial_light", node=SpatialLightAugConfig)
    cs.store(group="augmentation", name="spatial_color_light", node=SpatialColorLightAugConfig)
    cs.store(group="augmentation", name="full", node=FullAugConfig)

