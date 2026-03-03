"""Dataset-agnostic WSSS dataset: images + pseudo/GT segmentation masks.

Derives image list from mask_dir glob. Works with any dataset whose
pseudo masks were produced by the CAM refinement pipeline.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def _mmseg_resize(image: np.ndarray, mask: np.ndarray,
                  base_short: int = 512, ratio_range: tuple[float, float] = (0.5, 2.0),
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Resize matching mmseg Resize(img_scale=(2048, base_short), ratio_range).

    1. Scale so the shorter side equals base_short (keep aspect ratio).
    2. Multiply by a random ratio from ratio_range.
    """
    h, w = image.shape[:2]
    short_side = min(h, w)
    scale = base_short / short_side
    ratio = random.uniform(*ratio_range)
    scale *= ratio

    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _random_crop_with_cat_max_ratio(
    image: np.ndarray, mask: np.ndarray,
    crop_h: int, crop_w: int,
    cat_max_ratio: float = 0.75,
    ignore_index: int = 255,
    max_attempts: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Random crop rejecting patches dominated by a single class."""
    h, w = image.shape[:2]
    for _ in range(max_attempts):
        y = random.randint(0, max(0, h - crop_h))
        x = random.randint(0, max(0, w - crop_w))
        crop_mask = mask[y:y + crop_h, x:x + crop_w]

        valid = crop_mask[crop_mask != ignore_index]
        if valid.size == 0:
            continue

        _, counts = np.unique(valid, return_counts=True)
        if counts.max() / valid.size <= cat_max_ratio:
            return image[y:y + crop_h, x:x + crop_w], crop_mask

    y = random.randint(0, max(0, h - crop_h))
    x = random.randint(0, max(0, w - crop_w))
    return image[y:y + crop_h, x:x + crop_w], mask[y:y + crop_h, x:x + crop_w]


def _photo_metric_distortion(image: np.ndarray,
                             brightness_delta: int = 32,
                             contrast_range: tuple[float, float] = (0.5, 1.5),
                             saturation_range: tuple[float, float] = (0.5, 1.5),
                             hue_delta: int = 18,
                             ) -> np.ndarray:
    """Matches mmseg PhotoMetricDistortion: per-transform independent 50% chance."""
    img = image.astype(np.float32)

    if random.random() < 0.5:
        delta = random.uniform(-brightness_delta, brightness_delta)
        img += delta

    contrast_first = random.random() < 0.5

    if contrast_first and random.random() < 0.5:
        alpha = random.uniform(*contrast_range)
        img *= alpha

    if random.random() < 0.5:
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(*saturation_range)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    if random.random() < 0.5:
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-hue_delta, hue_delta)) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    if not contrast_first and random.random() < 0.5:
        alpha = random.uniform(*contrast_range)
        img *= alpha

    return np.clip(img, 0, 255).astype(np.uint8)


class MMSegTrainTransform:
    """Training augmentation matching the original mmseg WeakCLIP pipeline.

    Pipeline: Resize(base_short + random ratio) -> RandomCrop(cat_max_ratio=0.75)
    -> RandomFlip -> PhotoMetricDistortion -> Normalize -> Pad
    """

    def __init__(self, image_size: int = 512,
                 ratio_range: tuple[float, float] = (0.5, 2.0),
                 cat_max_ratio: float = 0.75) -> None:
        self.image_size = image_size
        self.ratio_range = ratio_range
        self.cat_max_ratio = cat_max_ratio
        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = ToTensorV2()

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:
        image, mask = _mmseg_resize(image, mask, base_short=self.image_size,
                                    ratio_range=self.ratio_range)

        h, w = image.shape[:2]
        if h < self.image_size or w < self.image_size:
            pad_h = max(0, self.image_size - h)
            pad_w = max(0, self.image_size - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=255)

        image, mask = _random_crop_with_cat_max_ratio(
            image, mask, self.image_size, self.image_size,
            cat_max_ratio=self.cat_max_ratio)

        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = _photo_metric_distortion(image)

        result = self.normalize(image=image, mask=mask)
        result = self.to_tensor(image=result["image"], mask=result["mask"])
        return result


def get_weakclip_val_transform(image_size: int = 512) -> A.Compose:
    """Validation transform: just resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class WSSDataset(Dataset):
    """Images + segmentation masks for WSSS training.

    Args:
        image_dir: Directory containing source images.
        mask_dir: Directory containing .png masks (pseudo or GT).
        image_ext: Image file extension (e.g. ".jpg", ".png").
        image_size: Resize target for both images and masks.
        transform: Optional albumentations pipeline (must handle both image and mask).
        is_train: If True, uses strong augmentation; if False, resize-only.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_ext: str = ".jpg",
        image_size: int = 512,
        transform: A.Compose | MMSegTrainTransform | None = None,
        is_train: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_ext = image_ext

        self.names = sorted(f.stem for f in self.mask_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No .png masks found in {self.mask_dir}")

        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = MMSegTrainTransform(image_size)
        else:
            self.transform = get_weakclip_val_transform(image_size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        img = np.array(
            Image.open(self.image_dir / f"{name}{self.image_ext}").convert("RGB")
        )
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        result = self.transform(image=img, mask=mask)
        image = result["image"]
        mask_tensor = result["mask"].long().unsqueeze(0)  # (1, H, W)

        return {"image": image, "mask": mask_tensor, "name": name}
