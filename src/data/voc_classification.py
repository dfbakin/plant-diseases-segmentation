"""PASCAL VOC 2012 multi-label classification dataset."""

from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

VOC_CLASSES = [
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
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCClassificationDataset(Dataset):
    """VOC 2012 multi-label classification: images + multi-hot labels from masks.

    Supports both albumentations (dict-style) and torchvision (callable) transforms.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train_aug_id",
        image_size: int = 448,
        transform: A.Compose | Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClassAug"
        self.num_classes = len(VOC_CLASSES)

        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        self.names = split_file.read_text().strip().splitlines()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx].strip()
        pil_img = Image.open(self.image_dir / f"{name}.jpg").convert("RGB")
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx in np.unique(mask):
            if 0 < cls_idx < 255:
                label[cls_idx - 1] = 1.0

        if isinstance(self.transform, A.Compose):
            augmented = self.transform(image=np.array(pil_img))
            image = augmented["image"]
        else:
            image = self.transform(pil_img)

        return {"image": image, "label": label, "name": name}
