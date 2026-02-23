"""Dataset-agnostic WSSS dataset: images + pseudo/GT segmentation masks.

Derives image list from mask_dir glob. Works with any dataset whose
pseudo masks were produced by the CAM refinement pipeline.
"""

from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class WSSDataset(Dataset):
    """Images + segmentation masks for WSSS training.

    Args:
        image_dir: Directory containing source images.
        mask_dir: Directory containing .png masks (pseudo or GT).
        image_ext: Image file extension (e.g. ".jpg", ".png").
        image_size: Resize target for both images and masks.
        transform: Optional albumentations pipeline (must handle both image and mask).
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_ext: str = ".jpg",
        image_size: int = 512,
        transform: A.Compose | None = None,
        is_train: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_ext = image_ext

        self.names = sorted(f.stem for f in self.mask_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No .png masks found in {self.mask_dir}")

        if transform is not None:
            self.transform = transform
        else:
            steps = [A.Resize(image_size, image_size)]
            if is_train:
                steps.append(A.HorizontalFlip(p=0.5))
            steps.extend([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            self.transform = A.Compose(steps)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        img = np.array(
            Image.open(self.image_dir / f"{name}{self.image_ext}").convert("RGB")
        )
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        augmented = self.transform(image=img, mask=mask)
        image = augmented["image"]
        mask_tensor = augmented["mask"].long().unsqueeze(0)  # (1, H, W)

        return {"image": image, "mask": mask_tensor, "name": name}
