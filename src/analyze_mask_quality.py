"""Diagnostic analysis of pseudomask quality vs ground truth.

Computes per-image and per-disease statistics (IoU, precision, recall,
oversegmentation, undersegmentation, sparsity) and generates PDF/PNG
figures for reporting.

Example:
    python src/analyze_mask_quality.py \
        gt_dir=outputs/plantseg_binary/gt_binary_train \
        image_dir=data/plantsegv3/images/train \
        'pred_dirs=[{path: outputs/plantseg_binary/pseudo_masks_t_0.64, label: PSA+RW}]' \
        output_dir=outputs/analysis/mask_quality
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class MaskQualityConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    gt_dir: str = "outputs/plantseg_binary/gt_binary_train"
    pred_dirs: list[Any] = field(default_factory=list)
    metadata_csv: str = "data/plantsegv3/Metadatav2.csv"
    image_dir: str = "data/plantsegv3/images/train"
    image_ext: str = ".jpg"
    output_dir: str = "outputs/analysis/mask_quality"
    num_worst: int = 5
    num_best: int = 5


cs = ConfigStore.instance()
cs.store(name="mask_quality_config", node=MaskQualityConfig)

METHOD_COLORS = [
    "#2176AE",  # blue
    "#D7263D",  # red
    "#57A773",  # green
    "#F49D37",  # orange
    "#8B5CF6",  # purple
]


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _load_disease_map(csv_path: str) -> dict[str, str]:
    """Map image stem -> disease name from Metadatav2.csv."""
    mapping: dict[str, str] = {}
    path = Path(csv_path)
    if not path.exists():
        log.warning(f"Metadata CSV not found: {csv_path}")
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["Name"]).stem
            mapping[stem] = row["Disease"]
    return mapping


# ---------------------------------------------------------------------------
# Per-image metric computation
# ---------------------------------------------------------------------------

def _compute_image_stats(
    gt: np.ndarray, pred: np.ndarray
) -> dict[str, float]:
    """Compute all per-image metrics for a single binary (gt, pred) pair."""
    gt_fg = gt > 0
    pred_fg = pred > 0
    total = gt.size

    tp = int(np.sum(gt_fg & pred_fg))
    fp = int(np.sum(~gt_fg & pred_fg))
    fn = int(np.sum(gt_fg & ~pred_fg))

    gt_fg_count = int(np.sum(gt_fg))
    pred_fg_count = int(np.sum(pred_fg))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overseg = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    underseg = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    gt_labeled, gt_nc = ndimage.label(gt_fg)
    pred_labeled, pred_nc = ndimage.label(pred_fg)

    return {
        "gt_fg_frac": gt_fg_count / total,
        "pred_fg_frac": pred_fg_count / total,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "overseg_ratio": overseg,
        "underseg_ratio": underseg,
        "gt_n_components": gt_nc,
        "pred_n_components": pred_nc,
    }


def compute_per_image_stats(
    gt_dir: Path,
    pred_dir: Path,
    disease_map: dict[str, str],
) -> pd.DataFrame:
    """Compute per-image statistics for all images in pred_dir."""
    names = sorted(f.stem for f in pred_dir.glob("*.png"))
    rows = []

    for name in tqdm(names, desc=f"Analyzing {pred_dir.name}"):
        gt_path = gt_dir / f"{name}.png"
        pred_path = pred_dir / f"{name}.png"
        if not gt_path.exists():
            continue

        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))

        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )

        stats = _compute_image_stats(gt, pred)
        stats["name"] = name
        stats["disease"] = disease_map.get(name, "unknown")
        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate statistics from per-image dataframe."""
    return {
        "n_images": len(df),
        "mIoU": float(df["iou"].mean()) * 100,
        "median_iou": float(df["iou"].median()) * 100,
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_overseg_ratio": float(df["overseg_ratio"].mean()),
        "mean_underseg_ratio": float(df["underseg_ratio"].mean()),
        "mean_gt_fg_frac": float(df["gt_fg_frac"].mean()),
        "mean_pred_fg_frac": float(df["pred_fg_frac"].mean()),
        "median_gt_fg_frac": float(df["gt_fg_frac"].median()),
        "median_pred_fg_frac": float(df["pred_fg_frac"].median()),
    }


def _aggregate_per_disease(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image stats by disease category."""
    return (
        df.groupby("disease")
        .agg(
            count=("iou", "size"),
            mean_iou=("iou", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_overseg=("overseg_ratio", "mean"),
            mean_underseg=("underseg_ratio", "mean"),
            mean_gt_fg_frac=("gt_fg_frac", "mean"),
            mean_pred_fg_frac=("pred_fg_frac", "mean"),
        )
        .sort_values("mean_iou", ascending=True)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(str(output_dir / f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_iou_histogram(
    dfs: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, df) in enumerate(dfs.items()):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        ax.hist(
            df["iou"], bins=50, alpha=0.55, label=label, color=color,
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel("Per-image foreground IoU")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-image IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, "iou_histogram")


def _plot_sparsity_scatter(
    dfs: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, (label, df) in enumerate(dfs.items()):
        ax = axes[0, i]
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        ax.scatter(
            df["gt_fg_frac"], df["pred_fg_frac"],
            s=4, alpha=0.3, color=color, rasterized=True,
        )
        lim = max(df["gt_fg_frac"].max(), df["pred_fg_frac"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("GT foreground fraction")
        ax.set_ylabel("Pred foreground fraction")
        ax.set_title(f"{label}")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("GT sparsity vs Pred sparsity", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir, "sparsity_scatter")


def _plot_iou_vs_gt_sparsity(
    dfs: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, (label, df) in enumerate(dfs.items()):
        ax = axes[0, i]
        sc = ax.scatter(
            df["gt_fg_frac"], df["iou"],
            c=df["overseg_ratio"], cmap="RdYlGn_r",
            s=4, alpha=0.4, vmin=0, vmax=1, rasterized=True,
        )
        ax.set_xlabel("GT foreground fraction")
        ax.set_ylabel("IoU")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Overseg ratio", shrink=0.8)
    fig.suptitle("IoU vs GT sparsity (colored by overseg ratio)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir, "iou_vs_gt_sparsity")


def _plot_overseg_underseg(
    dfs: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, (label, df) in enumerate(dfs.items()):
        ax = axes[0, i]
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        ax.scatter(
            df["overseg_ratio"], df["underseg_ratio"],
            s=4, alpha=0.3, color=color, rasterized=True,
        )
        mean_o = df["overseg_ratio"].mean()
        mean_u = df["underseg_ratio"].mean()
        ax.axvline(mean_o, color="red", linestyle="--", alpha=0.6, label=f"mean overseg={mean_o:.2f}")
        ax.axhline(mean_u, color="blue", linestyle="--", alpha=0.6, label=f"mean underseg={mean_u:.2f}")
        ax.set_xlabel("Oversegmentation ratio (FP / pred)")
        ax.set_ylabel("Undersegmentation ratio (FN / GT)")
        ax.set_title(f"{label}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Oversegmentation vs Undersegmentation", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir, "overseg_underseg")


def _plot_worst_best_diseases(
    disease_dfs: dict[str, pd.DataFrame], output_dir: Path, n: int = 10
) -> None:
    for label, ddf in disease_dfs.items():
        safe_label = label.replace("/", "_").replace(" ", "_")
        worst = ddf.head(n).copy()
        best = ddf.tail(n).copy()
        combined = pd.concat([worst, best])

        fig, ax = plt.subplots(figsize=(8, max(4, len(combined) * 0.35)))
        colors = ["#D7263D"] * len(worst) + ["#2176AE"] * len(best)
        ax.barh(
            range(len(combined)),
            combined["mean_iou"] * 100,
            color=colors, edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(
            [f"{d} (n={c})" for d, c in zip(combined["disease"], combined["count"])],
            fontsize=7,
        )
        ax.set_xlabel("Mean IoU (%)")
        ax.set_title(f"{label}: worst {n} (red) + best {n} (blue) diseases")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
        fig.tight_layout()
        _save_fig(fig, output_dir, f"worst_best_diseases_{safe_label}")


def _plot_example_grid(
    df: pd.DataFrame,
    label: str,
    image_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    image_ext: str,
    output_dir: Path,
    kind: str,
    n: int = 5,
) -> None:
    """Save a grid of n example images: Image | GT | Pred (overlaid)."""
    if kind == "worst":
        subset = df.nsmallest(n, "iou")
    else:
        subset = df.nlargest(n, "iou")

    safe_label = label.replace("/", "_").replace(" ", "_")
    gt_color = np.array([0.15, 0.75, 0.30])
    pred_color = np.array([1.0, 0.25, 0.25])
    alpha = 0.45

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (_, rec) in enumerate(subset.iterrows()):
        name = rec["name"]
        img_path = image_dir / f"{name}{image_ext}"
        gt_path = gt_dir / f"{name}.png"
        pred_path = pred_dir / f"{name}.png"

        img_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(gt_path))
        pred_mask = np.array(Image.open(pred_path))

        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask.astype(np.uint8)).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                )
            )

        gt_overlay = _overlay(img_np, gt_mask, gt_color, alpha)
        pred_overlay = _overlay(img_np, pred_mask, pred_color, alpha)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"{name}\nIoU={rec['iou']:.3f}", fontsize=7)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title(
            f"GT (fg={rec['gt_fg_frac']:.1%})", fontsize=7
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(
            f"Pred (fg={rec['pred_fg_frac']:.1%}, P={rec['precision']:.2f}, R={rec['recall']:.2f})",
            fontsize=7,
        )
        axes[row, 2].axis("off")

    fig.suptitle(f"{label}: {kind} {n} images by IoU", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, output_dir, f"{kind}_examples_{safe_label}")


def _overlay(
    img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float
) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    out = img_f.copy()
    fg = mask > 0
    out[fg] = out[fg] * (1 - alpha) + color * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_mask_quality(cfg: MaskQualityConfig) -> None:
    if not cfg.pred_dirs:
        raise ValueError("pred_dirs is required (list of {path, label})")

    gt_dir = Path(cfg.gt_dir)
    image_dir = Path(cfg.image_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    disease_map = _load_disease_map(cfg.metadata_csv)
    log.info(f"Loaded disease metadata for {len(disease_map)} images")

    pred_dirs: list[tuple[Path, str]] = []
    for entry in cfg.pred_dirs:
        if isinstance(entry, dict) or hasattr(entry, "keys"):
            pred_dirs.append((Path(entry["path"]), entry.get("label", str(entry["path"]))))
        else:
            pred_dirs.append((Path(str(entry)), str(entry)))

    all_dfs: dict[str, pd.DataFrame] = {}
    all_disease_dfs: dict[str, pd.DataFrame] = {}

    for pred_path, label in pred_dirs:
        if not pred_path.exists():
            log.warning(f"Pred dir not found, skipping: {pred_path}")
            continue

        safe_label = label.replace("/", "_").replace(" ", "_")
        log.info(f"Computing per-image stats: {label} ({pred_path})")
        df = compute_per_image_stats(gt_dir, pred_path, disease_map)

        if df.empty:
            log.warning(f"No valid image pairs for {label}")
            continue

        # Save per-image CSV
        csv_path = output_dir / f"{safe_label}_per_image.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Saved {len(df)} rows to {csv_path}")

        # Summary JSON
        summary = _aggregate_summary(df)
        summary["label"] = label
        summary["pred_dir"] = str(pred_path)
        json_path = output_dir / f"{safe_label}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(
            f"  {label}: mIoU={summary['mIoU']:.2f}%, "
            f"precision={summary['mean_precision']:.3f}, "
            f"recall={summary['mean_recall']:.3f}, "
            f"overseg={summary['mean_overseg_ratio']:.3f}, "
            f"underseg={summary['mean_underseg_ratio']:.3f}"
        )

        # Per-disease CSV
        disease_df = _aggregate_per_disease(df)
        disease_csv = output_dir / f"{safe_label}_per_disease.csv"
        disease_df.to_csv(disease_csv, index=False)

        all_dfs[label] = df
        all_disease_dfs[label] = disease_df

        # Per-method example grids
        _plot_example_grid(
            df, label, image_dir, gt_dir, pred_path, cfg.image_ext,
            fig_dir, "worst", cfg.num_worst,
        )
        _plot_example_grid(
            df, label, image_dir, gt_dir, pred_path, cfg.image_ext,
            fig_dir, "best", cfg.num_best,
        )

    if not all_dfs:
        log.error("No valid prediction directories processed")
        return

    # Comparative figures
    log.info("Generating comparative figures...")
    _plot_iou_histogram(all_dfs, fig_dir)
    _plot_sparsity_scatter(all_dfs, fig_dir)
    _plot_iou_vs_gt_sparsity(all_dfs, fig_dir)
    _plot_overseg_underseg(all_dfs, fig_dir)
    _plot_worst_best_diseases(all_disease_dfs, fig_dir)

    log.info(f"All outputs saved to {output_dir}")


@hydra.main(version_base=None, config_name="mask_quality_config")
def main(cfg: DictConfig) -> None:
    analyze_mask_quality(cfg)


if __name__ == "__main__":
    main()
