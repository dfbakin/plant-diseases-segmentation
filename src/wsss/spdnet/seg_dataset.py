"""Siamese segmentation dataset for the SPDNet probe diagnostic.

Returns ``(query_image, multi_label, gt_binary_mask)`` triples, where the
binary mask is derived on-the-fly from the multi-class PlantSeg annotation
``mask > 0 & mask != 255`` so this dataset has no extra DVC dependency.

Reference selection reuses ``SiamesePlantSegDataset``'s same-class logic
so the spatial fusion path receives a meaningful reference at every step
(otherwise P4/P5/P6 are undefined for the spatial checkpoint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.data.voc_classification import NUM_PLANTSEG_FG_CLASSES
from src.wsss.spdnet._split_index_cache import (
    filter_class_index_to_subset,
    scan_or_load_split,
)
from src.wsss.spdnet.dataset import SiamesePlantSegDataset


def build_seg_transform(image_size: int) -> A.Compose:
    """Albumentations pipeline that keeps image and mask spatially aligned."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_train_seg_transform(image_size: int) -> A.Compose:
    """Light geometric aug only -- preserves mask pixel correctness."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            interpolation=1,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


class _PlantSegBase(Dataset):
    """Plain PlantSeg dataset that returns ``(image, multi_label, binary_mask)``.

    Image+mask are jointly transformed by the given Albumentations pipeline
    so spatial augs stay aligned. Only PlantSeg images are included (no
    PlantVillage; PV has no GT segmentation).
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
        image_size: int = 448,
        transform: A.Compose | None = None,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "annotations" / split
        self.num_classes = NUM_PLANTSEG_FG_CLASSES
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Masks not found: {self.mask_dir}")

        all_names, all_class_to_indices = scan_or_load_split(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            num_classes=self.num_classes,
        )

        if limit is not None:
            self.names = list(all_names[:limit])
            self._cached_class_to_indices = filter_class_index_to_subset(
                all_class_to_indices, range(limit),
            )
        else:
            self.names = list(all_names)
            self._cached_class_to_indices = all_class_to_indices

        if transform is None:
            transform = build_seg_transform(image_size)
        self.transform = transform

        # PV samples are absent here; SiamesePlantSegDataset queries it.
        self._pv_samples: list = []

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _open_aligned(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Open image+mask, applying EXIF rotation so their (H, W) match.

        ~0.1% of PlantSeg train images have EXIF orientation tag 6 (90 deg CW).
        The annotators stored the mask in the visually-correct rotated frame, but
        PIL's ``Image.open(...).convert("RGB")`` does NOT apply EXIF rotation,
        so we end up with image (848, 636) and mask (636, 848). Albumentations'
        shape consistency check then refuses the pair, killing the worker.

        ``ImageOps.exif_transpose`` is a no-op when no EXIF orientation is set,
        so we apply it unconditionally to both image and mask.
        """
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            pil_img = np.array(im)
        with Image.open(mask_path) as mm:
            mm = ImageOps.exif_transpose(mm)
            mc_mask = np.array(mm)
        if pil_img.shape[:2] != mc_mask.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch even after EXIF transpose for {img_path.stem}: "
                f"image={pil_img.shape}, mask={mc_mask.shape}"
            )
        return pil_img, mc_mask

    def _load_pair(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        pil_img, mc_mask = self._open_aligned(
            self.image_dir / f"{name}.jpg", self.mask_dir / f"{name}.png",
        )
        binary = ((mc_mask > 0) & (mc_mask != 255)).astype(np.uint8)
        return pil_img, binary

    def _multi_label(self, mc_mask: np.ndarray) -> torch.Tensor:
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx in np.unique(mc_mask):
            if 1 <= cls_idx <= self.num_classes:
                label[cls_idx - 1] = 1.0
        return label

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        pil_img, mc_mask = self._open_aligned(
            self.image_dir / f"{name}.jpg", self.mask_dir / f"{name}.png",
        )
        binary = ((mc_mask > 0) & (mc_mask != 255)).astype(np.uint8)

        out = self.transform(image=pil_img, mask=binary)
        image: torch.Tensor = out["image"]
        mask = out["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        label = self._multi_label(mc_mask)
        return {"image": image, "label": label, "name": name, "mask": mask}


class SiamesePlantSegSegDataset(Dataset):
    """Wraps ``_PlantSegBase`` to produce paired ``(query, reference)`` samples
    AND the query's binary GT mask, for the seg-probe training.

    Each ``__getitem__`` returns a dict with::

        query:      {image, label, name, mask}
        references: list of {image, label, name}   (no masks needed)
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
        image_size: int = 448,
        train_aug: bool = True,
        num_references: int = 1,
        limit: int | None = None,
    ) -> None:
        if train_aug and split == "train":
            tfm = build_train_seg_transform(image_size)
        else:
            tfm = build_seg_transform(image_size)
        self.base = _PlantSegBase(
            root=root, split=split, image_size=image_size, transform=tfm, limit=limit,
        )
        self.siamese = SiamesePlantSegDataset(
            base_dataset=self.base, num_references=num_references,
        )

    def __len__(self) -> int:
        return len(self.siamese)

    def __getitem__(self, idx: int) -> dict:
        return self.siamese[idx]


def siamese_seg_collate_fn(batch: list[dict]) -> dict:
    """Batched ``(query_image, query_label, query_mask, ref_images, ref_labels)``."""
    n_refs = len(batch[0]["references"])
    out = {
        "query_image": torch.stack([s["query"]["image"] for s in batch]),
        "query_label": torch.stack([s["query"]["label"] for s in batch]),
        "query_mask": torch.stack([s["query"]["mask"] for s in batch]),
        "query_name": [s["query"]["name"] for s in batch],
        "ref_images": [
            torch.stack([s["references"][i]["image"] for s in batch])
            for i in range(n_refs)
        ],
        "ref_labels": [
            torch.stack([s["references"][i]["label"] for s in batch])
            for i in range(n_refs)
        ],
        "ref_names": [
            [s["references"][i]["name"] for s in batch]
            for i in range(n_refs)
        ],
    }
    return out
