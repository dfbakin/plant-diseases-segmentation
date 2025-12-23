from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.conf.augmentation import (
    ArtificialColorAugConfig,
    BaselineAugConfig,
    ColorNaturalAugConfig,
    FullAugConfig,
    NaturalColorAugConfig,
    NoiseBlurAugConfig,
    SpatialColorLightAugConfig,
    SpatialHeavyAugConfig,
    SpatialLightAugConfig,
)
from src.conf.data import DataConfig
from src.conf.model import (
    DeepLabV3PlusConfig,
    SegFormerConfig,
    SegNeXtConfig,
    UNetConfig,
)
from src.conf.trainer import TrainerConfig


@dataclass
class ExperimentConfig:
    name: str = "tmp_exp"
    seed: int = 42


@dataclass
class MLflowConfig:
    tracking_uri: Optional[str] = None  # null = local ./mlruns
    experiment_name: str = "${experiment.name}"


@dataclass
class PathsConfig:
    output_dir: str = "outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}"
    checkpoints: str = "${paths.output_dir}/checkpoints"
    logs: str = "${paths.output_dir}/logs"


@dataclass
class Config:
    """Root configuration containing all config groups.

    Defaults:
        - data: plantseg (DataConfig)
        - augmentation: spatial_color_light (SpatialColorLightAugConfig)
        - model: deeplabv3plus (DeepLabV3PlusConfig)
        - trainer: default (TrainerConfig)
    """

    # Use defaults for nested configs
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"data": "plantseg"},
            {"augmentation": "spatial_color_light"},
            {"model": "deeplabv3plus"},
            {"trainer": "default"},
        ]
    )

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    paths: PathsConfig = field(default_factory=PathsConfig)

    data: Any = MISSING
    augmentation: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING

    checkpoint_path: Optional[str] = None
    save_predictions: bool = False
    predictions_dir: str = "outputs/predictions"
    split: str = "val"


def register_configs() -> None:
    """Register all structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="config", node=Config)

    cs.store(group="data", name="plantseg", node=DataConfig)

    cs.store(group="model", name="deeplabv3plus", node=DeepLabV3PlusConfig)
    cs.store(group="model", name="segformer", node=SegFormerConfig)
    cs.store(group="model", name="segnext", node=SegNeXtConfig)
    cs.store(group="model", name="unet", node=UNetConfig)

    cs.store(group="trainer", name="default", node=TrainerConfig)

    cs.store(group="augmentation", name="baseline", node=BaselineAugConfig)
    cs.store(group="augmentation", name="spatial_light", node=SpatialLightAugConfig)
    cs.store(group="augmentation", name="spatial_heavy", node=SpatialHeavyAugConfig)
    cs.store(group="augmentation", name="color_natural", node=ColorNaturalAugConfig)
    cs.store(
        group="augmentation", name="artificial_color", node=ArtificialColorAugConfig
    )
    cs.store(group="augmentation", name="noise_blur", node=NoiseBlurAugConfig)
    cs.store(group="augmentation", name="natural_color", node=NaturalColorAugConfig)
    cs.store(
        group="augmentation",
        name="spatial_color_light",
        node=SpatialColorLightAugConfig,
    )
    cs.store(group="augmentation", name="full", node=FullAugConfig)
