"""Multi-label classification datasets for MCTformer (VOC + PlantSeg + Binary)."""

from pathlib import Path
from typing import Callable, Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.data.plantseg import DISEASE_CLASSES
from src.data.plantvillage import PlantVillageDataset

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


NUM_PLANTSEG_FG_CLASSES = len(DISEASE_CLASSES) - 1  # 115 diseases


class PlantSegMCTformerDataset(Dataset):
    """PlantSeg multi-label classification from GT segmentation masks.

    Uses DISEASE_CLASSES[1:] (115 diseases) as foreground classes so that
    CAM foreground index i maps to segmentation class i+1, matching the
    GT annotation indices directly.  Healthy-only images are skipped.

    Expected structure:
        {root}/images/{split}/*.jpg
        {root}/annotations/{split}/*.png   (multiclass, values 0-115)
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        image_size: int = 448,
        transform: A.Compose | Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "annotations" / split
        self.num_classes = NUM_PLANTSEG_FG_CLASSES

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Masks not found: {self.mask_dir}")

        self.names = self._collect_names()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _collect_names(self) -> list[str]:
        names = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path))
            fg_classes = set(np.unique(mask)) - {0, 255}
            if not fg_classes:
                continue
            names.append(img_path.stem)
        return names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        pil_img = Image.open(self.image_dir / f"{name}.jpg").convert("RGB")
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx in np.unique(mask):
            if 1 <= cls_idx <= self.num_classes:
                label[cls_idx - 1] = 1.0

        if isinstance(self.transform, A.Compose):
            augmented = self.transform(image=np.array(pil_img))
            image = augmented["image"]
        else:
            image = self.transform(pil_img)

        return {"image": image, "label": label, "name": name}


class BinaryPlantDataset(Dataset):
    """PlantSeg (all diseased) + PlantVillage (all samples, binary label) for MCTformer.

    Binary label: 0 = healthy, 1 = disease.
    PlantSeg images are always labeled 1 (all contain disease regions).
    PlantVillage images are labeled based on their original class: healthy (0) or diseased (1).
    """

    def __init__(
        self,
        plantseg_root: str | Path,
        plantvillage_root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        image_size: int = 448,
        transform: A.Compose | Callable | None = None,
        include_plantvillage: bool = True,
    ) -> None:
        self.num_classes = 1
        plantseg_root = Path(plantseg_root)
        self.image_dir = plantseg_root / "images" / split
        self.mask_dir = plantseg_root / "annotations" / split

        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        self.samples: list[tuple[Path, float]] = []
        self._names: set[str] = set()

        for img_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path))
            if not (set(np.unique(mask)) - {0, 255}):
                continue
            self.samples.append((img_path, 1.0))
            self._names.add(img_path.stem)

        if include_plantvillage:
            pv_ds = PlantVillageDataset(root=plantvillage_root, split=split)
            for sample in pv_ds.samples:
                lbl = 0.0 if sample["label"] == 0 else 1.0
                self.samples.append((sample["image_path"], lbl))
                self._names.add(sample["name"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, lbl = self.samples[idx]
        pil_img = Image.open(path).convert("RGB")

        if isinstance(self.transform, A.Compose):
            augmented = self.transform(image=np.array(pil_img))
            image = augmented["image"]
        else:
            image = self.transform(pil_img)

        label = torch.tensor([lbl], dtype=torch.float32)
        return {"image": image, "label": label, "name": Path(path).stem}
