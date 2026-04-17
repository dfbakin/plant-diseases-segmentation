"""Helpers to recover the disease class of a PlantSeg image from its filename.

Val image filenames follow the convention ``<class_name_with_underscores>_<id>``,
e.g. ``apple_scab_google_0190``. Multi-word class names (``apple scab``)
become underscore-joined (``apple_scab``). This module provides a parser
that looks up the longest matching class-name prefix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def load_class_names(path: str | Path) -> list[str]:
    """Load class names (one per line) from a text file."""
    return [l.strip() for l in open(path) if l.strip()]


def make_filename_class_resolver(class_names: list[str]) -> Callable[[str], int | None]:
    """Return a callable mapping ``image_stem -> class_index`` (or ``None``).

    Uses longest-prefix matching so that e.g. ``apple_black_rot_28`` is
    matched to ``apple black rot`` rather than the shorter ``apple``.
    """
    canonical = [(cn.replace(" ", "_"), idx) for idx, cn in enumerate(class_names)]
    canonical.sort(key=lambda x: -len(x[0]))

    def resolve(name: str) -> int | None:
        for cn, idx in canonical:
            if name == cn or name.startswith(cn + "_"):
                return idx
        return None

    return resolve


def build_class_pool_from_labels(
    label_file: str | Path,
    image_dir: str | Path,
    image_ext: str = ".jpg",
) -> dict[int, list[str]]:
    """Build ``{class_idx: [image_names]}`` from a label .npy + image dir.

    Filters to images that physically exist in *image_dir* so the names
    can be loaded directly later.
    """
    import numpy as np

    image_dir = Path(image_dir)
    labels = np.load(label_file, allow_pickle=True).item()
    pool: dict[int, list[str]] = {}
    for name, label in labels.items():
        if not (image_dir / f"{name}{image_ext}").exists():
            continue
        for cls in np.where(label > 0)[0]:
            pool.setdefault(int(cls), []).append(name)
    return pool
