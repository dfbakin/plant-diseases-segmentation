"""PASCAL VOC 2012 WSSS dataset: images + pseudo segmentation masks."""

from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class VOCWSSDataset(Dataset):
    """VOC 2012 images + pseudo segmentation masks for WSSS training."""

    def __init__(
        self,
        root: str | Path,
        pseudo_mask_dir: str = "SegmentationClassAug",
        split: str = "train_aug_id",
        image_size: int = 512,
        transform: A.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / pseudo_mask_dir

        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        self.names = split_file.read_text().strip().splitlines()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx].strip()
        img = np.array(Image.open(self.image_dir / f"{name}.jpg").convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        augmented = self.transform(image=img, mask=mask)
        image = augmented["image"]
        mask_tensor = augmented["mask"].long().unsqueeze(0)  # (1, H, W)

        return {"image": image, "mask": mask_tensor, "name": name}
