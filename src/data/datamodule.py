"""Lightning DataModule for PlantSeg dataset."""

from pathlib import Path
from typing import Literal

import lightning as L
from torch.utils.data import DataLoader

from src.data.plantseg import PlantSegDataset
from src.data.transforms import get_train_transforms, get_val_transforms


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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = mean
        self.std = std

        self.train_dataset: PlantSegDataset | None = None
        self.val_dataset: PlantSegDataset | None = None
        self.test_dataset: PlantSegDataset | None = None

        self.save_hyperparameters(ignore=["root"])

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        train_transform = get_train_transforms(self.image_size, self.mean, self.std)
        val_transform = get_val_transforms(self.image_size, self.mean, self.std)

        if stage == "fit" or stage is None:
            self.train_dataset = PlantSegDataset(self.root, "train", train_transform)
            self.val_dataset = PlantSegDataset(self.root, "val", val_transform)

        if stage == "validate":
            self.val_dataset = PlantSegDataset(self.root, "val", val_transform)

        if stage == "test" or stage is None:
            self.test_dataset = PlantSegDataset(self.root, "test", val_transform)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    @property
    def num_classes(self) -> int:
        return PlantSegDataset.NUM_CLASSES

