from dataclasses import dataclass, field


@dataclass
class NormalizationConfig:
    mean: list[float] = field(default_factory=lambda: [0.444959, 0.493009, 0.336009])
    std: list[float] = field(default_factory=lambda: [0.244765, 0.230322, 0.242037])


@dataclass
class DataConfig:
    root: str = "data/plantsegv3"

    # Segmentation mode: false = binary (2 classes), true = multiclass (116 classes)
    multiclass: bool = False

    image_size: int = 512

    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
