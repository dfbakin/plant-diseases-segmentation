"""PlantSeg Dataset for binary disease segmentation."""

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PlantSegDataset(Dataset):
    """Binary segmentation dataset: background (0) vs disease (1).

    Expected structure:
    - images/{split}/*.jpg
    - annotations/{split}/*.png
    - Metadatav2.csv (optional, adds Plant/Disease info)
    """

    NUM_CLASSES = 2

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / "images" / split
        self.masks_dir = self.root / "annotations" / split

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks not found: {self.masks_dir}")

        # Load optional metadata
        metadata_path = self.root / "Metadatav2.csv"
        if metadata_path.exists():
            split_name = {"train": "Training", "val": "Validation", "test": "Testing"}
            self.metadata = pd.read_csv(metadata_path)
            self.metadata = self.metadata[self.metadata["Split"] == split_name.get(split, split)]
        else:
            self.metadata = None

        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[dict]:
        samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            sample = {"image_path": img_path, "mask_path": mask_path, "name": img_path.stem}

            if self.metadata is not None:
                row = self.metadata[self.metadata["Name"] == img_path.name]
                if not row.empty:
                    sample["plant"] = row.iloc[0]["Plant"]
                    sample["disease"] = row.iloc[0]["Disease"]

            samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = cv2.cvtColor(cv2.imread(str(sample["image_path"])), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.int64)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        result = {"image": image, "mask": mask, "name": sample["name"]}
        if "plant" in sample:
            result["plant"] = sample["plant"]
        if "disease" in sample:
            result["disease"] = sample["disease"]
        return result

    def get_class_weights(self) -> np.ndarray:
        """Inverse frequency weights for imbalanced classes."""
        class_counts = np.zeros(self.NUM_CLASSES, dtype=np.float64)

        for sample in self.samples:
            mask = (cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.int64)
            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < self.NUM_CLASSES:
                    class_counts[cls] += cnt

        total = class_counts.sum()
        weights = total / (self.NUM_CLASSES * class_counts + 1e-8)
        return weights / weights.sum()

    @property
    def class_names(self) -> list[str]:
        return ["background", "disease"]

