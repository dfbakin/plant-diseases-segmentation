"""Lightning DataModule for PlantSeg dataset."""

from pathlib import Path
from typing import Literal

import albumentations as A
import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.data.plantseg import PlantSegDataset, PlantSegMulticlassDataset
from src.data.transforms import get_val_transforms


class PlantSegDataModule(L.LightningDataModule):
    """Handles train/val/test dataloaders with augmentation transforms."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        train_transform: A.Compose | None = None,
        multiclass: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = mean
        self.std = std
        self.train_transform = train_transform
        self.multiclass = multiclass

        self._dataset_cls = PlantSegMulticlassDataset if multiclass else PlantSegDataset

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.save_hyperparameters(ignore=["root", "train_transform"])

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        if self.train_transform is not None:
            train_transform = self.train_transform
        else:
            from src.data.transforms import get_train_transforms

            train_transform = get_train_transforms(self.image_size, self.mean, self.std)

        val_transform = get_val_transforms(self.image_size, self.mean, self.std)

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset_cls(self.root, "train", train_transform)
            self.val_dataset = self._dataset_cls(self.root, "val", val_transform)

        if stage == "validate":
            self.val_dataset = self._dataset_cls(self.root, "val", val_transform)

        if stage == "test" or stage is None:
            self.test_dataset = self._dataset_cls(self.root, "test", val_transform)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        return self._dataset_cls.NUM_CLASSES

