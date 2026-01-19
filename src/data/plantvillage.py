"""PlantVillage Classification Dataset with PlantSeg-compatible class indices.

Class Index System:
- 0: healthy (all healthy samples)
- 1-115: PlantSeg diseases (from DISEASE_CLASSES[1:])
- 116-119: PlantVillage-only diseases (no PlantSeg ground truth)
"""

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.plantseg import DISEASE_CLASSES, PlantSegDataset
from src.data.plantvillage_mappings import (
    CLASSIFICATION_CLASSES,
    DISEASE_TO_CLASS_IDX,
    NUM_CLASSIFICATION_CLASSES,
    PLANTVILLAGE_FOLDER_TO_CLASS,
    PLANTVILLAGE_FOLDER_TO_PLANT,
    EXCLUDED_FOLDERS,
    PLANTVILLAGE_FOLDER_TO_PLANT,
)

class PlantVillageDataset(Dataset):
    """PlantVillage dataset for classification with PlantSeg-compatible indices.

    Returns:
        dict with keys: image, label, name, plant, disease, folder
    """

    NUM_CLASSES = NUM_CLASSIFICATION_CLASSES

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] | None = None,
        transform: Callable | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize PlantVillage dataset.

        Args:
            root: Path to PlantVillage root directory (containing class folders)
            split: Data split ("train", "val", "test"). If None, use all data.
            transform: Albumentations transform pipeline
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            seed: Random seed for reproducible splits
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        if not self.root.exists():
            raise FileNotFoundError(f"PlantVillage root not found: {self.root}")

        self.samples = self._collect_samples()
        self._class_counts: np.ndarray | None = None
    
    def _collect_samples(self) -> list[dict]:
        all_samples = []

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name in EXCLUDED_FOLDERS:
                continue
            if folder.name not in PLANTVILLAGE_FOLDER_TO_CLASS:
                continue

            class_idx = PLANTVILLAGE_FOLDER_TO_CLASS[folder.name]
            plant = PLANTVILLAGE_FOLDER_TO_PLANT[folder.name]
            disease = CLASSIFICATION_CLASSES[class_idx]

            # jpeg, jpg, png
            regex_ext_patterns = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]

            for regex_ext_pattern in regex_ext_patterns:
                for img_path in sorted(folder.glob(regex_ext_pattern)):
                    all_samples.append(
                        {
                            "image_path": img_path,
                            "label": class_idx,
                            "name": img_path.stem,
                            "plant": plant,
                            "disease": disease,
                            "folder": folder.name,
                        }
                    )

        if not all_samples:
            raise ValueError(f"No samples found in {self.root}")

        # Apply split if specified
        if self.split is not None:
            all_samples = self._apply_split(all_samples)

        return all_samples

    def _apply_split(self, samples: list[dict]) -> list[dict]:
        rng = np.random.default_rng(self.seed)

        # Group by class for stratified split
        class_to_samples: dict[int, list[dict]] = {}
        for sample in samples:
            class_idx = sample["label"]
            if class_idx not in class_to_samples:
                class_to_samples[class_idx] = []
            class_to_samples[class_idx].append(sample)

        split_samples = []
        for class_idx, class_samples in class_to_samples.items():
            n = len(class_samples)
            indices = rng.permutation(n)

            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            if self.split == "train":
                selected_indices = indices[:n_train]
            elif self.split == "val":
                selected_indices = indices[n_train : n_train + n_val]
            else:  # test
                selected_indices = indices[n_train + n_val :]

            for idx in selected_indices:
                split_samples.append(class_samples[idx])

        return split_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
    
        image = cv2.cvtColor(cv2.imread(str(sample["image_path"])), cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:  # Grayscale with channel dim
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] == 4:  # RGBA -> RGB
            image = image[:, :, :3]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return {
            "image": image,
            "label": sample["label"],
            "name": sample["name"],
            "plant": sample["plant"],
            "disease": sample["disease"],
            "has_mask": False,
        }

    def get_class_counts(self) -> np.ndarray:
        """Get sample count per class."""
        if self._class_counts is None:
            counts = np.zeros(self.NUM_CLASSES, dtype=np.int64)
            for sample in self.samples:
                counts[sample["label"]] += 1
            self._class_counts = counts
        return self._class_counts

    def get_class_weights(self) -> np.ndarray:
        """Inverse frequency weights for class imbalance.

        Returns:
            Array of shape (NUM_CLASSES,) with normalized weights.
            Classes with more samples get lower weights.
        """
        counts = self.get_class_counts().astype(np.float64)
        # Avoid division by zero for classes with no samples
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        return weights / weights.sum()

    def get_effective_class_weights(self, beta: float = 0.999) -> np.ndarray:
        """Effective number of samples weighting (Cui et al., 2019).

        Args:
            beta: Hyperparameter in [0, 1). Higher = more emphasis on rare classes.

        Returns:
            Array of shape (NUM_CLASSES,) with normalized weights.
        """
        counts = self.get_class_counts().astype(np.float64)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        return weights / weights.sum()

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler.

        Returns:
            Array of shape (len(self),) with weight for each sample.
        """
        class_weights = self.get_class_weights()
        sample_weights = np.array(
            [class_weights[sample["label"]] for sample in self.samples]
        )
        return sample_weights

    @property
    def class_names(self) -> list[str]:
        return CLASSIFICATION_CLASSES

    def get_class_distribution(self) -> pd.DataFrame:
        """Get detailed class distribution as DataFrame."""
        counts = self.get_class_counts()
        data = []
        for idx, (name, count) in enumerate(zip(CLASSIFICATION_CLASSES, counts)):
            if count > 0:
                data.append(
                    {
                        "class_idx": idx,
                        "class_name": name,
                        "count": count,
                        "percentage": count / len(self) * 100,
                    }
                )
        return pd.DataFrame(data).sort_values("count", ascending=False)


class PlantSegClassificationDataset(Dataset):
    """Wrapper to use PlantSeg dataset for classification (image-level labels)."""

    NUM_CLASSES = NUM_CLASSIFICATION_CLASSES

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        return_mask: bool = False,
    ) -> None:
        """Initialize PlantSeg classification dataset.

        Args:
            root: Path to PlantSeg root directory
            split: Data split ("train", "val", "test")
            transform: Albumentations transform pipeline
            return_mask: If True, include GT mask in output (for CAM evaluation)
        """
        self._seg_dataset = PlantSegDataset(root=root, split=split, transform=None)
        self.root = self._seg_dataset.root
        self.split = split
        self.transform = transform
        self.return_mask = return_mask

        self.samples = self._build_classification_samples()
        self._class_counts: np.ndarray | None = None

    def _build_classification_samples(self) -> list[dict]:
        """Convert segmentation samples to classification samples."""
        samples = []
        for seg_sample in self._seg_dataset.samples:
            if "disease" not in seg_sample:
                continue

            disease_name = seg_sample["disease"]
            if disease_name not in DISEASE_TO_CLASS_IDX:
                # Disease not in our class system (shouldn't happen)
                continue

            class_idx = DISEASE_TO_CLASS_IDX[disease_name]

            samples.append(
                {
                    "image_path": seg_sample["image_path"],
                    "mask_path": seg_sample["mask_path"],  # Keep for CAM evaluation
                    "label": class_idx,
                    "name": seg_sample["name"],
                    "plant": seg_sample.get("plant", "unknown"),
                    "disease": disease_name,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = cv2.cvtColor(cv2.imread(str(sample["image_path"])), cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:  # Grayscale with channel dim
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] == 4:  # RGBA -> RGB
            image = image[:, :, :3]

        if self.return_mask:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.int64)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]

            return {
                "image": image,
                "mask": mask,
                "label": sample["label"],
                "name": sample["name"],
                "plant": sample["plant"],
                "disease": sample["disease"],
                "has_mask": True,
            }
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return {
                "image": image,
                "label": sample["label"],
                "name": sample["name"],
                "plant": sample["plant"],
                "disease": sample["disease"],
                "has_mask": True,
            }

    def get_mask(self, idx: int) -> np.ndarray:
        """Get the ground truth segmentation mask for CAM evaluation.

        Returns:
            Binary mask (H, W) where 1 = disease region.
        """
        sample = self.samples[idx]
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8)

    def get_class_counts(self) -> np.ndarray:
        """Get sample count per class."""
        if self._class_counts is None:
            counts = np.zeros(self.NUM_CLASSES, dtype=np.int64)
            for sample in self.samples:
                counts[sample["label"]] += 1
            self._class_counts = counts
        return self._class_counts

    def get_class_weights(self) -> np.ndarray:
        """Inverse frequency weights for class imbalance."""
        counts = self.get_class_counts().astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        return weights / weights.sum()

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = np.array(
            [class_weights[sample["label"]] for sample in self.samples]
        )
        return sample_weights

    @property
    def class_names(self) -> list[str]:
        return CLASSIFICATION_CLASSES


def get_combined_class_weights(
    plantvillage_dataset: PlantVillageDataset,
    plantseg_dataset: PlantSegClassificationDataset,
) -> np.ndarray:
    """Compute class weights from combined datasets."""
    combined_counts = (
        plantvillage_dataset.get_class_counts()
        + plantseg_dataset.get_class_counts()
    ).astype(np.float64)

    combined_counts = np.maximum(combined_counts, 1.0)
    weights = 1.0 / combined_counts
    return weights / weights.sum()


def get_combined_sample_weights(
    plantvillage_dataset: PlantVillageDataset,
    plantseg_dataset: PlantSegClassificationDataset,
) -> np.ndarray:
    """Compute per-sample weights for combined datasets."""
    class_weights = get_combined_class_weights(plantvillage_dataset, plantseg_dataset)

    pv_weights = np.array(
        [class_weights[s["label"]] for s in plantvillage_dataset.samples]
    )
    ps_weights = np.array(
        [class_weights[s["label"]] for s in plantseg_dataset.samples]
    )

    return np.concatenate([pv_weights, ps_weights])
