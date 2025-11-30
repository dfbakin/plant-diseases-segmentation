"""Lightning DataModule for PlantSeg dataset."""

from pathlib import Path
from typing import Literal

import lightning as L
from torch.utils.data import DataLoader

from src.data.plantseg import PlantSegDataset
from src.data.transforms import get_train_transforms, get_val_transforms


class PlantSegDataModule(L.LightningDataModule):
    """Lightning DataModule for PlantSeg dataset.

    Handles train/val/test splits with proper transforms and dataloaders.

    Attributes:
        root: Path to plantsegv3 directory.
        image_size: Target image size for transforms.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize DataModule.

        Args:
            root: Path to plantsegv3 directory.
            image_size: Target image size.
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            pin_memory: Whether to pin memory for GPU transfer.
        """
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = mean
        self.std = std

        # Will be set in setup()
        self.train_dataset: PlantSegDataset | None = None
        self.val_dataset: PlantSegDataset | None = None
        self.test_dataset: PlantSegDataset | None = None

        self.save_hyperparameters(ignore=["root"])

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        """Setup datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        train_transform = get_train_transforms(image_size=self.image_size, mean=self.mean, std=self.std)
        val_transform = get_val_transforms(image_size=self.image_size, mean=self.mean, std=self.std)

        if stage == "fit" or stage is None:
            self.train_dataset = PlantSegDataset(
                root=self.root,
                split="train",
                transform=train_transform,
            )
            self.val_dataset = PlantSegDataset(
                root=self.root,
                split="val",
                transform=val_transform,
            )

        if stage == "validate":
            self.val_dataset = PlantSegDataset(
                root=self.root,
                split="val",
                transform=val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = PlantSegDataset(
                root=self.root,
                split="test",
                transform=val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        """Return number of segmentation classes."""
        return PlantSegDataset.NUM_CLASSES

