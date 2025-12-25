"""PlantSeg Dataset for disease segmentation (binary and multiclass)."""

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# fmt: off
DISEASE_CLASSES = [
    "background", "apple black rot", "apple mosaic virus", "apple rust", "apple scab",
    "banana anthracnose", "banana black leaf streak", "banana bunchy top", "banana cigar end rot",
    "banana cordana leaf spot", "banana panama disease", "basil downy mildew", "bean halo blight",
    "bean mosaic virus", "bean rust", "bell pepper bacterial spot", "bell pepper blossom end rot",
    "bell pepper frogeye leaf spot", "bell pepper powdery mildew", "blueberry anthracnose",
    "blueberry botrytis blight", "blueberry mummy berry", "blueberry rust", "blueberry scorch",
    "broccoli alternaria leaf spot", "broccoli downy mildew", "broccoli ring spot",
    "cabbage alternaria leaf spot", "cabbage black rot", "cabbage downy mildew",
    "carrot alternaria leaf blight", "carrot cavity spot", "carrot cercospora leaf blight",
    "cauliflower alternaria leaf spot", "cauliflower bacterial soft rot", "celery anthracnose",
    "celery early blight", "cherry leaf spot", "cherry powdery mildew", "citrus canker",
    "citrus greening disease", "coffee berry blotch", "coffee black rot", "coffee brown eye spot",
    "coffee leaf rust", "corn gray leaf spot", "corn northern leaf blight", "corn rust", "corn smut",
    "cucumber angular leaf spot", "cucumber bacterial wilt", "cucumber powdery mildew",
    "eggplant cercospora leaf spot", "eggplant phomopsis fruit rot", "eggplant phytophthora blight",
    "garlic leaf blight", "garlic rust", "ginger leaf spot", "ginger sheath blight", "grape black rot",
    "grape downy mildew", "grape leaf spot", "grapevine leafroll disease", "lettuce downy mildew",
    "lettuce mosaic virus", "maple tar spot", "peach anthracnose", "peach brown rot", "peach leaf curl",
    "peach rust", "peach scab", "plum bacterial spot", "plum brown rot", "plum pocket disease",
    "plum pox virus", "plum rust", "potato early blight", "potato late blight", "raspberry fire blight",
    "raspberry gray mold", "raspberry leaf spot", "raspberry yellow rust", "rice blast",
    "rice sheath blight", "soybean bacterial blight", "soybean brown spot", "soybean downy mildew",
    "soybean frog eye leaf spot", "soybean mosaic", "soybean rust", "squash powdery mildew",
    "strawberry anthracnose", "strawberry leaf scorch", "tobacco blue mold", "tobacco brown spot",
    "tobacco frogeye leaf spot", "tobacco mosaic virus", "tomato bacterial leaf spot",
    "tomato early blight", "tomato late blight", "tomato leaf mold", "tomato mosaic virus",
    "tomato septoria leaf spot", "tomato yellow leaf curl virus",
    "wheat bacterial leaf streak (black chaff)", "wheat head scab", "wheat leaf rust",
    "wheat loose smut", "wheat powdery mildew", "wheat septoria blotch", "wheat stem rust",
    "wheat stripe rust", "zucchini bacterial wilt", "zucchini downy mildew",
    "zucchini powdery mildew", "zucchini yellow mosaic virus",
]
# fmt: on


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
            split_name = {"train": "Training", "val": "Validation", "test": "Test"}
            self.metadata = pd.read_csv(metadata_path)
            self.metadata = self.metadata[
                self.metadata["Split"] == split_name.get(split, split)
            ]
        else:
            self.metadata = None

        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[dict]:
        samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            sample = {
                "image_path": img_path,
                "mask_path": mask_path,
                "name": img_path.stem,
            }

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
        mask = (cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0).astype(
            np.int64
        )

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
            mask = (
                cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            ).astype(np.int64)
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


class PlantSegMulticlassDataset(PlantSegDataset):
    """Multiclass segmentation: 116 classes (background + 115 diseases)."""

    NUM_CLASSES = 116

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = cv2.cvtColor(cv2.imread(str(sample["image_path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE).astype(
            np.int64
        )

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
        """Inverse frequency weights for 116 classes."""
        class_counts = np.zeros(self.NUM_CLASSES, dtype=np.float64)

        for sample in self.samples:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE).astype(
                np.int64
            )
            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < self.NUM_CLASSES:
                    class_counts[cls] += cnt

        total = class_counts.sum()
        weights = total / (self.NUM_CLASSES * class_counts + 1e-8)
        return weights / weights.sum()

    @property
    def class_names(self) -> list[str]:
        return DISEASE_CLASSES
