"""Augmentation configuration dataclasses.

Each augmentation preset defines a list of transforms with their parameters.
The transforms are instantiated by the `build` method.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class TransformConfig:
    """Single transform configuration to enable user to add custom transforms"""

    _target_: str
    p: float = 1.0
    # Additional parameters stored as dict
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    name: str = "baseline"

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        """Build the augmentation pipeline. Override in subclasses."""
        raise NotImplementedError


@dataclass
class BaselineAugConfig(AugmentationConfig):
    name: str = "baseline"

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class SpatialLightAugConfig(AugmentationConfig):
    """Light spatial augmentations: flips and rotations."""

    name: str = "spatial_light"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    affine_p: float = 0.4

    translate_percent: tuple[float, float] = (-0.1, 0.1)
    scale: tuple[float, float] = (0.9, 1.1)
    rotate: tuple[int, int] = (-30, 30)

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=self.horizontal_flip_p),
            A.VerticalFlip(p=self.vertical_flip_p),
            A.RandomRotate90(p=self.rotate90_p),
            A.Affine(
                translate_percent={"x": self.translate_percent, "y": self.translate_percent},
                scale=self.scale,
                rotate=self.rotate,
                border_mode=0,
                p=self.affine_p,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class SpatialHeavyAugConfig(AugmentationConfig):
    name: str = "spatial_heavy"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    affine_p: float = 0.6
    distortion_p: float = 0.4
    coarse_dropout_p: float = 0.3

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=self.horizontal_flip_p),
            A.VerticalFlip(p=self.vertical_flip_p),
            A.RandomRotate90(p=self.rotate90_p),
            A.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                scale=(0.7, 1.3),
                rotate=(-45, 45),
                border_mode=0,
                p=self.affine_p,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=6, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                ],
                p=self.distortion_p,
            ),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(0.06, 0.12),
                hole_width_range=(0.06, 0.12),
                fill=0,
                p=self.coarse_dropout_p,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class ColorNaturalAugConfig(AugmentationConfig):
    """Pure natural color augmentations"""

    name: str = "color_natural"

    brightness_contrast_p: float = 0.5
    hsv_p: float = 0.4

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.brightness_contrast_p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=self.hsv_p,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class ArtificialColorAugConfig(AugmentationConfig):
    """Artificial color augmentations: includes color jitter with hue shifts"""

    name: str = "artificial_color"

    brightness_contrast_p: float = 0.5
    hsv_p: float = 0.5
    color_jitter_p: float = 0.4
    gamma_p: float = 0.3
    noise_p: float = 0.3
    blur_p: float = 0.2

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=self.brightness_contrast_p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=self.hsv_p,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=self.color_jitter_p,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=self.gamma_p),
            A.GaussNoise(std_range=(0.1, 0.3), p=self.noise_p),
            A.GaussianBlur(blur_limit=(3, 7), p=self.blur_p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class NoiseBlurAugConfig(AugmentationConfig):
    """Noise and blur augmentations"""

    name: str = "noise_blur"

    noise_p: float = 0.4
    blur_p: float = 0.3

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.GaussNoise(std_range=(0.1, 0.35), p=self.noise_p),
            A.GaussianBlur(blur_limit=(3, 5), p=self.blur_p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class NaturalColorAugConfig(AugmentationConfig):
    """Natural color + noise/blur combined"""

    name: str = "natural_color"

    brightness_contrast_p: float = 0.5
    hsv_p: float = 0.4
    noise_p: float = 0.3
    blur_p: float = 0.2

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.brightness_contrast_p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=self.hsv_p,
            ),
            A.GaussNoise(std_range=(0.1, 0.3), p=self.noise_p),
            A.GaussianBlur(blur_limit=(3, 5), p=self.blur_p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class SpatialColorLightAugConfig(AugmentationConfig):
    """Combined light spatial + natural color augmentations"""

    name: str = "spatial_color_light"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    affine_p: float = 0.4

    brightness_contrast_p: float = 0.5
    hsv_p: float = 0.4
    noise_p: float = 0.2
    blur_p: float = 0.15

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=self.horizontal_flip_p),
            A.VerticalFlip(p=self.vertical_flip_p),
            A.RandomRotate90(p=self.rotate90_p),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-30, 30),
                border_mode=0,
                p=self.affine_p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.brightness_contrast_p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=self.hsv_p,
            ),
            A.GaussNoise(std_range=(0.08, 0.25), p=self.noise_p),
            A.GaussianBlur(blur_limit=(3, 5), p=self.blur_p),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


@dataclass
class FullAugConfig(AugmentationConfig):
    name: str = "full"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    affine_p: float = 0.5
    distortion_p: float = 0.25
    coarse_dropout_p: float = 0.2

    brightness_contrast_p: float = 0.5
    hsv_p: float = 0.4
    noise_p: float = 0.25
    blur_p: float = 0.2

    def build(self, image_size: int, mean: list[float], std: list[float]) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=self.horizontal_flip_p),
            A.VerticalFlip(p=self.vertical_flip_p),
            A.RandomRotate90(p=self.rotate90_p),
            A.Affine(
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                border_mode=0,
                p=self.affine_p,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=80, sigma=5, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                ],
                p=self.distortion_p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.brightness_contrast_p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=self.hsv_p,
            ),
            A.GaussNoise(std_range=(0.1, 0.28), p=self.noise_p),
            A.GaussianBlur(blur_limit=(3, 5), p=self.blur_p),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(0.06, 0.12),
                hole_width_range=(0.06, 0.12),
                fill=0,
                p=self.coarse_dropout_p,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# Registry for augmentation configs
AUGMENTATION_REGISTRY: dict[str, type[AugmentationConfig]] = {
    "baseline": BaselineAugConfig,
    "spatial_light": SpatialLightAugConfig,
    "spatial_heavy": SpatialHeavyAugConfig,
    "color_natural": ColorNaturalAugConfig,
    "artificial_color": ArtificialColorAugConfig,
    "noise_blur": NoiseBlurAugConfig,
    "natural_color": NaturalColorAugConfig,
    "spatial_color_light": SpatialColorLightAugConfig,
    "full": FullAugConfig,
}


def get_augmentation_config(name: str) -> type[AugmentationConfig]:
    """Get augmentation config class by name."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY.keys())}")
    return AUGMENTATION_REGISTRY[name]
