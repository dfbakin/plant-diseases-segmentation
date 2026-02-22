"""Affinity label dataset for PSA training.

Reads la_crf / ha_crf probability maps, derives pixel-pair affinity labels
(background-positive, foreground-positive, negative) for training the
AffinityNet.

Ported from MCTformer/psa/voc12/data.py (VOC12AffDataset +
ExtractAffinityLabelInRadius).
"""

from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset


class ExtractAffinityLabelInRadius:
    """Extract bg/fg/neg affinity labels for pixel pairs within a radius."""

    def __init__(self, cropsize: int, radius: int = 5):
        self.radius = radius
        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))
        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius - 1
        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor

    def __call__(self, label: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels_from = label[: -self.radius_floor, self.radius_floor : -self.radius_floor]
        labels_from = labels_from.reshape(-1)

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[
                dy : dy + self.crop_height,
                self.radius_floor + dx : self.radius_floor + dx + self.crop_width,
            ]
            labels_to = labels_to.reshape(-1)
            valid_pair = np.logical_and(labels_to < 255, labels_from < 255)
            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity = np.equal(bc_labels_from, concat_labels_to)
        bg_pos = np.logical_and(pos_affinity, np.equal(bc_labels_from, 0)).astype(np.float32)
        fg_pos = np.logical_and(
            np.logical_and(pos_affinity, np.not_equal(bc_labels_from, 0)), concat_valid_pair
        ).astype(np.float32)
        neg = np.logical_and(np.logical_not(pos_affinity), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos), torch.from_numpy(fg_pos), torch.from_numpy(neg)


class VOCAffDataset(Dataset):
    """VOC affinity dataset for PSA training using la_crf + ha_crf probability maps."""

    def __init__(
        self,
        voc_root: str | Path,
        la_crf_dir: str | Path,
        ha_crf_dir: str | Path,
        split: str = "train_aug_id",
        cropsize: int = 448,
        radius: int = 5,
        normalize_fn=None,
    ):
        self.voc_root = Path(voc_root)
        self.la_crf_dir = Path(la_crf_dir)
        self.ha_crf_dir = Path(ha_crf_dir)
        self.normalize_fn = normalize_fn

        split_file = self.voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        self.names = split_file.read_text().strip().splitlines()
        self.cropsize = cropsize
        self.extract_aff = ExtractAffinityLabelInRadius(cropsize=cropsize // 8, radius=radius)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx].strip()
        img = PIL.Image.open(self.voc_root / "JPEGImages" / f"{name}.jpg").convert("RGB")

        la_probs = np.load(str(self.la_crf_dir / f"{name}.npy"))
        ha_probs = np.load(str(self.ha_crf_dir / f"{name}.npy"))

        # Stack la + ha probabilities, then crop jointly with image
        label = np.concatenate([la_probs, ha_probs], axis=0)  # (2*C, H, W)
        label = np.transpose(label, (1, 2, 0))  # (H, W, 2*C)

        img = np.array(img)
        img, label = _random_crop(img, label, self.cropsize)
        img, label = _random_hflip(img, label)

        # Apply normalization
        if self.normalize_fn:
            img = self.normalize_fn(img)
        else:
            img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)

        # Derive affinity labels from la/ha argmax
        label = np.transpose(label, (2, 0, 1))  # (2*C, H, W)
        num_cls = label.shape[0] // 2
        la_part = label[:num_cls]
        ha_part = label[num_cls:]

        no_score_region = np.max(np.concatenate([la_part, ha_part], axis=0), axis=0) < 1e-5
        la_label = np.argmax(la_part, axis=0).astype(np.uint8)
        ha_label = np.argmax(ha_part, axis=0).astype(np.uint8)

        # Combine: la foreground is trusted fg, ha background is trusted bg
        combined = la_label.copy()
        combined[la_label == 0] = 255
        combined[ha_label == 0] = 0
        combined[no_score_region] = 255

        # Downsample to feature map size (stride 8) then extract affinity labels
        from PIL import Image as PILImage

        combined_pil = PILImage.fromarray(combined)
        feat_h, feat_w = self.cropsize // 8, self.cropsize // 8
        combined_ds = np.array(combined_pil.resize((feat_w, feat_h), resample=PILImage.NEAREST))

        bg_pos, fg_pos, neg = self.extract_aff(combined_ds)

        return torch.from_numpy(img).float(), (bg_pos, fg_pos, neg)


def _random_crop(img: np.ndarray, label: np.ndarray, cropsize: int):
    h, w = img.shape[:2]
    if h < cropsize or w < cropsize:
        pad_h = max(cropsize - h, 0)
        pad_w = max(cropsize - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        label = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        h, w = img.shape[:2]

    top = np.random.randint(0, h - cropsize + 1)
    left = np.random.randint(0, w - cropsize + 1)
    img = img[top : top + cropsize, left : left + cropsize]
    label = label[top : top + cropsize, left : left + cropsize]
    return img, label


def _random_hflip(img: np.ndarray, label: np.ndarray):
    if np.random.random() < 0.5:
        img = np.flip(img, axis=1).copy()
        label = np.flip(label, axis=1).copy()
    return img, label
