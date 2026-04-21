"""Disk-backed cache for the PlantSeg per-split mask scan.

Without this cache, every probe job pays the same ~2.5-minute fixed cost:

* :class:`~src.wsss.spdnet.seg_dataset._PlantSegBase` opens every PNG in
  ``annotations/{train,val}/`` to filter out empty-foreground masks.
* :class:`~src.wsss.spdnet.dataset.SiamesePlantSegDataset` then opens the
  *same* masks again to bucket them by class for same-class reference
  sampling.

Both passes can be served from a single scan: each PlantSeg mask reveals
its filtered-or-not status *and* its set of foreground class IDs in one
``np.unique`` call. We persist the result to ``outputs/_cache/`` keyed by
``(image_dir, mask_dir, n_images, n_masks, mask_dir_mtime)`` so that the
*first* probe in an overnight run pays the scan, and all subsequent
probes load it in <1 second. Net saving on the documented overnight
budget: ~50 min off Phase 1, ~50 min off Phase 2, ~5 min off Phase 3.

The cache is invalidated automatically when masks are added/removed
(directory mtime changes) or when the file count changes -- both are
cheap to check (no PNG decode needed).

Concurrent probe jobs are safe: writes are atomic via ``Path.replace``,
and a missed-cache race only causes redundant scans, never corruption.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

CACHE_ROOT = Path("outputs/_cache")
CACHE_VERSION = 1


def _cache_key(
    image_dir: Path,
    mask_dir: Path,
    num_classes: int,
) -> dict:
    """Lightweight fingerprint detecting mask-set changes without PNG decode."""
    img_files = list(image_dir.glob("*.jpg"))
    mask_files = list(mask_dir.glob("*.png"))
    return {
        "version": CACHE_VERSION,
        "image_dir": str(image_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "num_classes": int(num_classes),
        "n_images": len(img_files),
        "n_masks": len(mask_files),
        "mask_dir_mtime": float(mask_dir.stat().st_mtime),
    }


def _cache_path(mask_dir: Path) -> Path:
    """Disk location for the cache JSON of a particular mask directory."""
    safe_tag = str(mask_dir.resolve()).replace("/", "_").strip("_")
    return CACHE_ROOT / f"plantseg_split_{safe_tag}.json"


def scan_or_load_split(
    image_dir: Path,
    mask_dir: Path,
    num_classes: int,
) -> tuple[list[str], dict[int, list[int]]]:
    """Return ``(filtered_names, class_to_indices)`` for one PlantSeg split.

    First call decodes every mask once (~1 min for the 7.9 k train
    images on a warm filesystem cache) and writes the result to
    ``outputs/_cache/``. Subsequent calls with the same fingerprint load
    from disk in well under a second.

    Args:
        image_dir: Directory of ``*.jpg`` source images.
        mask_dir: Sibling directory of ``*.png`` segmentation masks.
        num_classes: Number of foreground classes; only ``cls_idx`` in
            ``[1, num_classes]`` is recorded (PlantSeg uses 0=BG, 255=ignore).

    Returns:
        names: image stems whose mask contains at least one foreground
            class -- order matches a fresh ``sorted(image_dir.glob('*.jpg'))``.
        class_to_indices: mapping from 0-indexed class id to the list of
            positions in ``names`` whose mask contains that class.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    key = _cache_key(image_dir, mask_dir, num_classes)
    cache_path = _cache_path(mask_dir)

    if cache_path.exists():
        try:
            with cache_path.open() as fh:
                cached = json.load(fh)
            if cached.get("key") == key:
                names = list(cached["names"])
                class_to_indices = {
                    int(cls): list(idxs)
                    for cls, idxs in cached["class_to_indices"].items()
                }
                return names, class_to_indices
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    names: list[str] = []
    class_to_indices: dict[int, list[int]] = {}

    for img_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path))
        fg_classes = set(np.unique(mask).tolist()) - {0, 255}
        if not fg_classes:
            continue
        idx = len(names)
        names.append(img_path.stem)
        for cls_idx in fg_classes:
            if 1 <= cls_idx <= num_classes:
                class_to_indices.setdefault(int(cls_idx) - 1, []).append(idx)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as fh:
        json.dump(
            {
                "key": key,
                "names": names,
                "class_to_indices": {
                    str(cls): idxs for cls, idxs in class_to_indices.items()
                },
            },
            fh,
        )
    tmp_path.replace(cache_path)

    return names, class_to_indices


def filter_class_index_to_subset(
    class_to_indices: dict[int, list[int]],
    valid_indices: range | set[int],
) -> dict[int, list[int]]:
    """Drop entries that point past a truncated (e.g. ``limit``-ed) name list.

    Used by ``_PlantSegBase`` when the caller passes ``limit=N``: the
    cache is global to the split (covers all 7916 train images) but we
    only retain the first ``N``, so any class index pointing at
    position >= ``N`` would crash ``self.base[idx]`` later.
    """
    valid = valid_indices if isinstance(valid_indices, set) else set(valid_indices)
    return {
        cls: [i for i in idxs if i in valid]
        for cls, idxs in class_to_indices.items()
    }
