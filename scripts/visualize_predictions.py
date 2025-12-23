"""Interactive visualization of predictions vs ground truth.

Usage:
    # Interactive mode with keyboard navigation
    python scripts/visualize_predictions.py \
        --predictions outputs/predictions/segformer_val

    # Start from specific index
    python scripts/visualize_predictions.py \
        --predictions outputs/predictions/segformer_val \
        --start-idx 100

    # Generate static grid comparison
    python scripts/visualize_predictions.py \
        --predictions outputs/predictions/segformer_val \
        --grid outputs/predictions/grid.png

Controls:
    - Right Arrow / D / N: Next image
    - Left Arrow / A / P: Previous image
    - Q / Escape: Quit
    - S: Save current visualization
    - R: Toggle show raw masks (without overlay)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# ruff: noqa
matplotlib.use("TkAgg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Colors for overlays
COLOR_PREDICTION = (255, 0, 255)  # Magenta
COLOR_GROUND_TRUTH = (0, 255, 255)  # Cyan
OVERLAY_ALPHA = 0.6


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Create image with colored mask overlay."""
    overlay = image.copy()
    if mask.max() > 0:
        overlay[mask > 0] = (
            overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
    return overlay


def load_sample(metadata: dict, name: str) -> dict:
    """Load images and masks for a sample."""
    meta = metadata[name]

    image = cv2.cvtColor(cv2.imread(meta["image_path"]), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    gt_mask = (cv2.imread(meta["gt_mask_path"], cv2.IMREAD_GRAYSCALE) > 0).astype(
        np.uint8
    )

    pred_mask_raw = cv2.imread(meta["prediction_path"], cv2.IMREAD_GRAYSCALE)
    pred_mask = (pred_mask_raw > 0).astype(np.uint8)

    # Resize prediction to match original image size if needed
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return {"name": name, "image": image, "gt_mask": gt_mask, "pred_mask": pred_mask}


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Compute IoU, Dice, and accuracy for a sample."""
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    intersection = np.logical_and(gt_flat == 1, pred_flat == 1).sum()
    union = np.logical_or(gt_flat == 1, pred_flat == 1).sum()

    return {
        "iou": intersection / (union + 1e-8),
        "dice": 2 * intersection / (gt_flat.sum() + pred_flat.sum() + 1e-8),
        "accuracy": (gt_flat == pred_flat).mean(),
    }


class PredictionVisualizer:
    """Interactive visualizer for prediction vs ground truth comparison."""

    def __init__(self, predictions_dir: Path, start_idx: int = 0):
        self.predictions_dir = Path(predictions_dir)
        metadata_path = self.predictions_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.sample_names = list(self.metadata.keys())
        self.current_idx = min(start_idx, len(self.sample_names) - 1)
        self.show_raw = False

        print(f"Loaded {len(self.sample_names)} samples")
        print("Controls: ←/→ to navigate, Q to quit, S to save, R to toggle raw masks")

    def run_interactive(self):
        """Run interactive visualization."""
        print(f"Matplotlib backend: {matplotlib.get_backend()}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.15)
        fig.canvas.manager.set_window_title("Prediction Visualizer")

        def update_display():
            sample = load_sample(self.metadata, self.sample_names[self.current_idx])
            metrics = compute_metrics(sample["gt_mask"], sample["pred_mask"])

            for ax in axes:
                ax.clear()

            axes[0].imshow(sample["image"])
            axes[0].set_title(f"Original\n{sample['name'][:30]}")
            axes[0].axis("off")

            if self.show_raw:
                axes[1].imshow(sample["pred_mask"], cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Prediction (raw)")
                axes[2].imshow(sample["gt_mask"], cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Ground Truth (raw)")
            else:
                axes[1].imshow(
                    create_overlay(
                        sample["image"], sample["pred_mask"], COLOR_PREDICTION
                    )
                )
                axes[1].set_title(f"Prediction (magenta)\nIoU: {metrics['iou']:.3f}")
                axes[2].imshow(
                    create_overlay(
                        sample["image"], sample["gt_mask"], COLOR_GROUND_TRUTH
                    )
                )
                axes[2].set_title(f"Ground Truth (cyan)\nDice: {metrics['dice']:.3f}")

            axes[1].axis("off")
            axes[2].axis("off")

            fig.suptitle(
                f"Sample {self.current_idx + 1}/{len(self.sample_names)} | Acc: {metrics['accuracy']:.3f}",
                fontsize=12,
            )
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key in ["right", "d", "n"]:
                self.current_idx = (self.current_idx + 1) % len(self.sample_names)
                update_display()
            elif event.key in ["left", "a", "p"]:
                self.current_idx = (self.current_idx - 1) % len(self.sample_names)
                update_display()
            elif event.key in ["q", "escape"]:
                plt.close(fig)
            elif event.key == "s":
                save_path = (
                    self.predictions_dir
                    / f"viz_{self.sample_names[self.current_idx]}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved: {save_path}")
            elif event.key == "r":
                self.show_raw = not self.show_raw
                update_display()

        fig.canvas.mpl_connect("key_press_event", on_key)

        # Navigation buttons
        btn_prev = Button(plt.axes([0.3, 0.02, 0.1, 0.05]), "← Prev")
        btn_next = Button(plt.axes([0.6, 0.02, 0.1, 0.05]), "Next →")

        btn_prev.on_clicked(
            lambda _: (
                setattr(
                    self, "current_idx", (self.current_idx - 1) % len(self.sample_names)
                ),
                update_display(),
            )
        )
        btn_next.on_clicked(
            lambda _: (
                setattr(
                    self, "current_idx", (self.current_idx + 1) % len(self.sample_names)
                ),
                update_display(),
            )
        )

        update_display()
        print("Window opened. Use arrow keys or buttons to navigate. Press Q to quit.")
        plt.show(block=True)


def create_comparison_grid(
    predictions_dir: Path,
    output_path: Path,
    num_samples: int = 16,
    seed: int = 42,
):
    """Create a static grid comparison of multiple samples."""
    metadata_path = predictions_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    sample_names = list(metadata.keys())
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(sample_names), size=min(num_samples, len(sample_names)), replace=False
    )

    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 3, figsize=(4 * cols * 3, 4 * rows))
    axes = axes.reshape(rows, cols * 3)

    for i, idx in enumerate(indices):
        row, col_base = i // cols, (i % cols) * 3
        sample = load_sample(metadata, sample_names[idx])

        axes[row, col_base].imshow(sample["image"])
        axes[row, col_base].set_title(f"{sample['name'][:15]}...", fontsize=8)
        axes[row, col_base].axis("off")

        axes[row, col_base + 1].imshow(
            create_overlay(sample["image"], sample["pred_mask"], COLOR_PREDICTION)
        )
        axes[row, col_base + 1].set_title("Pred (magenta)", fontsize=8)
        axes[row, col_base + 1].axis("off")

        axes[row, col_base + 2].imshow(
            create_overlay(sample["image"], sample["gt_mask"], COLOR_GROUND_TRUTH)
        )
        axes[row, col_base + 2].set_title("GT (cyan)", fontsize=8)
        axes[row, col_base + 2].axis("off")

    # Hide unused axes
    for i in range(len(indices), rows * cols):
        row, col_base = i // cols, (i % cols) * 3
        for j in range(3):
            axes[row, col_base + j].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison grid to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions interactively")
    parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions directory"
    )
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Starting sample index"
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Generate static grid and save to this path",
    )
    parser.add_argument(
        "--grid-samples", type=int, default=16, help="Number of samples in grid"
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions)

    if args.grid:
        create_comparison_grid(predictions_dir, Path(args.grid), args.grid_samples)
    else:
        PredictionVisualizer(predictions_dir, args.start_idx).run_interactive()


if __name__ == "__main__":
    main()
