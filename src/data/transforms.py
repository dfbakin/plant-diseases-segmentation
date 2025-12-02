"""Albumentations transform pipelines for segmentation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: int = 512,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Training pipeline with spatial and color augmentations."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0, fill_mask=0),
        A.RandomCrop(image_size, image_size),
        # Spatial
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5),
        # Color (preserve disease coloration patterns)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: int = 512,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Deterministic resize + normalize for val/test."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0, fill_mask=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# TODO: refine when using for FixMatch
def get_strong_augment_transforms(
    image_size: int = 512,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Strong augmentation for semi-supervised learning."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, value=0, mask_value=0),
        A.RandomCrop(image_size, image_size),
        # Strong spatial
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=90, border_mode=0, p=0.7),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.GridDistortion(p=0.3),
        # Strong color
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

