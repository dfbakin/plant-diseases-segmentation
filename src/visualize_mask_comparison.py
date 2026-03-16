"""Side-by-side comparison of masks from multiple directories.

Generates grid figures: Image | GT | Dir1 | Dir2 | ...
with semi-transparent colored overlays for easy visual comparison.

Example:
    python src/visualize_mask_comparison.py \
        image_dir=data/plantsegv3/images/train \
        gt_dir=data/plantsegv3/masks/train \
        mask_dirs='[
            {path: outputs/plantseg_binary/pseudo_masks, label: PSA+RW},
            {path: outputs/plantseg_binary/weakclip_masks_fast, label: WeakCLIP},
            {path: outputs/plantseg_binary/sam_refined/A1, label: SAM-A1}
        ]' \
        output_dir=outputs/visualizations/sam_comparison \
        num_samples=20
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class MaskDirEntry:
    path: str = ""
    label: str = ""


@dataclass
class VisMaskCompareConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    image_dir: str = ""
    image_ext: str = ".jpg"
    gt_dir: str = ""
    mask_dirs: list[Any] = field(default_factory=list)
    output_dir: str = "outputs/visualizations/mask_comparison"

    num_samples: int = 20
    seed: int = 42
    figsize_per_col: float = 3.0
    alpha: float = 0.45


cs = ConfigStore.instance()
cs.store(name="vis_mask_compare_config", node=VisMaskCompareConfig)


# Perceptually distinct colors for up to 10 mask directories
OVERLAY_COLORS = [
    (0.85, 0.20, 0.20),  # red
    (0.20, 0.70, 0.30),  # green
    (0.20, 0.40, 0.85),  # blue
    (0.90, 0.60, 0.10),  # orange
    (0.60, 0.20, 0.80),  # purple
    (0.10, 0.80, 0.80),  # cyan
    (0.85, 0.45, 0.65),  # pink
    (0.50, 0.50, 0.50),  # grey
    (0.65, 0.85, 0.20),  # lime
    (0.90, 0.90, 0.20),  # yellow
]

FOREGROUND_COLOR = np.array([1.0, 0.25, 0.25])
GT_COLOR = np.array([0.15, 0.75, 0.30])


def _overlay_mask(
    image_np: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray | tuple,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colored mask overlay onto an image."""
    color = np.asarray(color, dtype=np.float32)
    img_f = image_np.astype(np.float32) / 255.0
    overlay = img_f.copy()
    fg = mask > 0
    overlay[fg] = overlay[fg] * (1 - alpha) + color * alpha
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def visualize_mask_comparison(cfg: VisMaskCompareConfig) -> None:
    if not cfg.image_dir:
        raise ValueError("image_dir is required")
    if not cfg.mask_dirs:
        raise ValueError("mask_dirs is required (list of {path, label})")

    image_dir = Path(cfg.image_dir)
    gt_dir = Path(cfg.gt_dir) if cfg.gt_dir else None
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse mask_dirs
    mask_dirs: list[tuple[Path, str]] = []
    for entry in cfg.mask_dirs:
        if isinstance(entry, (dict, DictConfig)):
            mask_dirs.append((Path(entry["path"]), entry.get("label", entry["path"])))
        else:
            mask_dirs.append((Path(str(entry)), str(entry)))

    # Find common images across all mask directories
    available = None
    for md, _ in mask_dirs:
        stems = {f.stem for f in md.glob("*.png")}
        available = stems if available is None else available & stems
    if gt_dir:
        gt_stems = {f.stem for f in gt_dir.glob("*.png")}
        available = available & gt_stems if available else gt_stems

    if not available:
        log.error("No common images found across mask directories")
        return

    rng = np.random.default_rng(cfg.seed)
    all_names = sorted(available)
    n_samples = min(cfg.num_samples, len(all_names))
    selected = rng.choice(all_names, size=n_samples, replace=False)
    selected.sort()

    has_gt = gt_dir is not None
    n_cols = 1 + int(has_gt) + len(mask_dirs)
    col_labels = ["Image"]
    if has_gt:
        col_labels.append("GT")
    for _, label in mask_dirs:
        col_labels.append(label)

    log.info(
        f"Generating {n_samples} comparison figures, "
        f"columns: {col_labels}, output: {output_dir}"
    )

    for name in tqdm(selected, desc="Visualizing"):
        img_path = image_dir / f"{name}{cfg.image_ext}"
        if not img_path.exists():
            log.warning(f"Image not found: {img_path}")
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(cfg.figsize_per_col * n_cols, cfg.figsize_per_col),
        )
        if n_cols == 1:
            axes = [axes]

        col = 0

        # Original image
        axes[col].imshow(img_np)
        axes[col].set_title("Image", fontsize=9)
        axes[col].axis("off")
        col += 1

        # Ground truth
        if has_gt:
            gt_mask = np.array(Image.open(gt_dir / f"{name}.png"))
            overlay = _overlay_mask(img_np, gt_mask, GT_COLOR, cfg.alpha)
            axes[col].imshow(overlay)
            axes[col].set_title("GT", fontsize=9)
            axes[col].axis("off")
            col += 1

        # Mask directories
        for md, label in mask_dirs:
            mask_path = md / f"{name}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                overlay = _overlay_mask(img_np, mask, FOREGROUND_COLOR, cfg.alpha)
                axes[col].imshow(overlay)
            else:
                axes[col].imshow(img_np)
                axes[col].text(
                    0.5, 0.5, "N/A", transform=axes[col].transAxes,
                    ha="center", va="center", fontsize=14, color="red",
                )
            axes[col].set_title(label, fontsize=9)
            axes[col].axis("off")
            col += 1

        plt.tight_layout(pad=0.5)
        fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info(f"Saved {n_samples} comparison figures to {output_dir}")


@hydra.main(version_base=None, config_name="vis_mask_compare_config")
def main(cfg: DictConfig) -> None:
    visualize_mask_comparison(cfg)


if __name__ == "__main__":
    main()
