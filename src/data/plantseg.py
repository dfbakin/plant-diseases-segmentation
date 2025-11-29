"""PlantSeg Dataset implementation."""

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PlantSegDataset(Dataset):
    """Dataset for PlantSeg segmentation masks.

    PlantSeg provides pixel-level disease annotations with the following structure:
    - images/{split}/*.jpg
    - annotations/{split}/*.png
    - Metadatav2.csv (contains Plant, Disease, Split information)

    Attributes:
        root: Path to plantsegv3 directory.
        split: One of 'train', 'val', 'test'.
        transform: Albumentations transform pipeline.
        class_mapping: Dict mapping disease names to class indices.
    """

    # Binary segmentation: background (0) vs disease (1)
    NUM_CLASSES = 2

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
    ) -> None:
        """Initialize PlantSeg dataset.

        Args:
            root: Path to plantsegv3 directory.
            split: Dataset split ('train', 'val', 'test').
            transform: Albumentations transform to apply.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Validate paths
        self.images_dir = self.root / "images" / split
        self.masks_dir = self.root / "annotations" / split

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Load metadata for additional info (plant type, disease)
        metadata_path = self.root / "Metadatav2.csv"
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
            # Filter by split
            split_name = {"train": "Training", "val": "Validation", "test": "Testing"}
            self.metadata = self.metadata[
                self.metadata["Split"] == split_name.get(split, split)
            ]
        else:
            self.metadata = None

        # Collect image-mask pairs
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[dict]:
        """Collect all valid image-mask pairs.

        Returns:
            List of dicts with 'image_path', 'mask_path', and optional metadata.
        """
        samples = []
        image_paths = sorted(self.images_dir.glob("*.jpg"))

        for img_path in image_paths:
            # Corresponding mask has .png extension
            mask_path = self.masks_dir / f"{img_path.stem}.png"

            if not mask_path.exists():
                continue

            sample = {
                "image_path": img_path,
                "mask_path": mask_path,
                "name": img_path.stem,
            }

            # Add metadata if available (Plant name and disease name)
            if self.metadata is not None:
                row = self.metadata[self.metadata["Name"] == img_path.name]
                if not row.empty:
                    sample["plant"] = row.iloc[0]["Plant"]
                    sample["disease"] = row.iloc[0]["Disease"]

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys: 'image', 'mask', 'name', and optional 'plant', 'disease' if available.
        """
        sample = self.samples[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(str(sample["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale, binarize to 0/1)
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.int64)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        result = {
            "image": image,
            "mask": mask,
            "name": sample["name"],
        }

        # Include metadata if available
        if "plant" in sample.keys():
            result["plant"] = sample["plant"]
        if "disease" in sample.keys():
            result["disease"] = sample["disease"]

        return result

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced data.

        Useful for weighted loss functions.

        Returns:
            Array of shape (num_classes,) with inverse frequency weights.
        """
        class_counts = np.zeros(self.NUM_CLASSES, dtype=np.float64)

        for sample in self.samples:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.int64)
            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < self.NUM_CLASSES:
                    class_counts[cls] += cnt

        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (self.NUM_CLASSES * class_counts + 1e-8)
        return weights / weights.sum()

    @property
    def class_names(self) -> list[str]:
        """Return class names for visualization."""
        return ["background", "disease"]

