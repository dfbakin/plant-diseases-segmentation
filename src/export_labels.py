"""Export image-level labels to a universal .npy dict for CAM generation.

Output format: {image_name: multi_hot_label_array} saved as .npy.
Works with VOC masks, PlantSeg metadata, PlantVillage folders,
or MCTformer classification predictions.

Example:
    python src/export_labels.py mode=voc_masks voc_root=data/VOC2012
    python src/export_labels.py mode=plantseg root=data/plantsegv3
    python src/export_labels.py mode=plantseg_wsss root=data/plantsegv3 pv_split=train
    python src/export_labels.py mode=plantvillage root=data/plant-village
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class ExportLabelsConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    mode: str = "voc_masks"
    output: str = "outputs/labels/labels.npy"

    # VOC mode
    voc_root: str = "data/VOC2012"
    split: str = "train_aug_id"
    num_classes: int = 20

    # PlantSeg / PlantVillage mode
    root: str = ""
    pv_split: str = "train"


cs = ConfigStore.instance()
cs.store(name="export_labels_config", node=ExportLabelsConfig)


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]


def export_voc_masks(cfg: ExportLabelsConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    from PIL import Image

    voc_root = Path(cfg.voc_root)
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{cfg.split}.txt"
    names = split_file.read_text().strip().splitlines()
    mask_dir = voc_root / "SegmentationClassAug"

    labels = {}
    for name in tqdm(names, desc="VOC masks"):
        name = name.strip()
        mask = np.array(Image.open(mask_dir / f"{name}.png"))
        label = np.zeros(cfg.num_classes, dtype=np.float32)
        for cls_idx in np.unique(mask):
            if 0 < cls_idx < 255:
                label[cls_idx - 1] = 1.0
        labels[name] = label
    return labels, VOC_CLASSES


def export_plantseg(cfg: ExportLabelsConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    from src.data.plantvillage import PlantSegClassificationDataset
    from src.data.plantvillage_mappings import CLASSIFICATION_CLASSES, NUM_CLASSIFICATION_CLASSES

    ds = PlantSegClassificationDataset(root=cfg.root, split=cfg.pv_split)
    labels = {}
    for sample in tqdm(ds.samples, desc="PlantSeg"):
        label = np.zeros(NUM_CLASSIFICATION_CLASSES, dtype=np.float32)
        label[sample["label"]] = 1.0
        labels[sample["name"]] = label
    return labels, CLASSIFICATION_CLASSES


def export_plantseg_wsss(cfg: ExportLabelsConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    """Export labels from PlantSeg GT masks using DISEASE_CLASSES.

    Uses 115 foreground disease classes (DISEASE_CLASSES[1:]) so that
    segmentation class indices match the GT annotations directly.
    Healthy-only images (mask contains only background) are skipped.
    """
    from PIL import Image as PILImage

    from src.data.plantseg import DISEASE_CLASSES, PlantSegMulticlassDataset

    num_fg = len(DISEASE_CLASSES) - 1  # 115 diseases (exclude background)
    class_names = DISEASE_CLASSES[1:]

    ds = PlantSegMulticlassDataset(root=cfg.root, split=cfg.pv_split, transform=None)

    labels: dict[str, np.ndarray] = {}
    skipped = 0
    for sample in tqdm(ds.samples, desc="PlantSeg WSSS"):
        mask = np.array(PILImage.open(sample["mask_path"]))
        present = set(np.unique(mask)) - {0, 255}
        if not present:
            skipped += 1
            continue

        label = np.zeros(num_fg, dtype=np.float32)
        for cls_idx in present:
            if 1 <= cls_idx <= num_fg:
                label[cls_idx - 1] = 1.0
        labels[sample["name"]] = label

    log.info(f"PlantSeg WSSS: {len(labels)} images with diseases, {skipped} healthy-only skipped")
    return labels, class_names


def export_plantvillage(cfg: ExportLabelsConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    from src.data.plantvillage import PlantVillageDataset
    from src.data.plantvillage_mappings import CLASSIFICATION_CLASSES, NUM_CLASSIFICATION_CLASSES

    ds = PlantVillageDataset(root=cfg.root, split=cfg.pv_split)
    labels = {}
    for sample in tqdm(ds.samples, desc="PlantVillage"):
        label = np.zeros(NUM_CLASSIFICATION_CLASSES, dtype=np.float32)
        label[sample["label"]] = 1.0
        labels[sample["name"]] = label
    return labels, CLASSIFICATION_CLASSES


EXPORTERS = {
    "voc_masks": export_voc_masks,
    "plantseg": export_plantseg,
    "plantseg_wsss": export_plantseg_wsss,
    "plantvillage": export_plantvillage,
}


def export_labels(cfg: ExportLabelsConfig) -> None:
    if cfg.mode not in EXPORTERS:
        raise ValueError(f"Unknown mode: {cfg.mode}. Choose from {list(EXPORTERS.keys())}")

    labels, class_names = EXPORTERS[cfg.mode](cfg)

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), labels)

    class_names_path = output.parent / "class_names.txt"
    class_names_path.write_text("\n".join(class_names) + "\n")

    sample_name = next(iter(labels))
    log.info(
        f"Exported {len(labels)} labels to {output} "
        f"(num_classes={len(labels[sample_name])}, sample={sample_name})"
    )
    log.info(f"Exported {len(class_names)} class names to {class_names_path}")


@hydra.main(version_base=None, config_name="export_labels_config")
def main(cfg: DictConfig) -> None:
    export_labels(cfg)


if __name__ == "__main__":
    main()
