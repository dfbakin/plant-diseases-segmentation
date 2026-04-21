"""Unit tests for the PlantSeg split-index cache.

Run:
    .venv/bin/pytest tests/test_split_index_cache.py -v

Each test builds a tiny synthetic split (a handful of 16x16 PNG masks)
inside ``tmp_path``, then exercises one cache property in isolation.
This isolates the tests from the real PlantSeg dataset and lets them
run on CI in milliseconds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.wsss.spdnet import _split_index_cache as cache_mod
from src.wsss.spdnet._split_index_cache import (
    filter_class_index_to_subset,
    scan_or_load_split,
)


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------

def _write_pair(image_dir: Path, mask_dir: Path, stem: str, classes: list[int]) -> None:
    """Write a tiny synthetic (jpg, png) pair where the mask carries `classes`."""
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(img).save(image_dir / f"{stem}.jpg")

    mask = np.zeros((16, 16), dtype=np.uint8)
    for i, cls in enumerate(classes):
        x0 = (i * 4) % 16
        mask[x0 : x0 + 4, x0 : x0 + 4] = cls
    Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")


def _empty_pair(image_dir: Path, mask_dir: Path, stem: str) -> None:
    """Write a (jpg, png) pair where the mask has only background (=0)."""
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(img).save(image_dir / f"{stem}.jpg")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(
        mask_dir / f"{stem}.png",
    )


@pytest.fixture(autouse=True)
def _isolate_cache_root(tmp_path, monkeypatch):
    """Redirect the cache to a per-test tmp dir so tests never collide."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", cache_dir)
    return cache_dir


@pytest.fixture
def synthetic_split(tmp_path):
    """Build a 5-image split with mixed foreground classes plus 1 empty mask."""
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    _write_pair(image_dir, mask_dir, "a", classes=[1, 2])
    _write_pair(image_dir, mask_dir, "b", classes=[2])
    _write_pair(image_dir, mask_dir, "c", classes=[3, 5])
    _write_pair(image_dir, mask_dir, "d", classes=[1, 5])
    _empty_pair(image_dir, mask_dir, "e")
    return image_dir, mask_dir


# ----------------------------------------------------------------------------
# (a) Cold scan correctness
# ----------------------------------------------------------------------------

class TestColdScan:
    def test_filters_empty_masks(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        names, _ = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        assert "e" not in names, "empty-foreground mask must be filtered out"
        assert names == ["a", "b", "c", "d"]

    def test_class_index_correctness(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        _, cls_idx = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        # 0-indexed: cls 1 -> idx 0, cls 5 -> idx 4
        assert sorted(cls_idx[0]) == [0, 3], "class 1 lives in 'a' (0) and 'd' (3)"
        assert sorted(cls_idx[1]) == [0, 1], "class 2 lives in 'a' (0) and 'b' (1)"
        assert sorted(cls_idx[2]) == [2], "class 3 lives only in 'c' (2)"
        assert sorted(cls_idx[4]) == [2, 3], "class 5 lives in 'c' (2) and 'd' (3)"

    def test_respects_num_classes_upper_bound(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        # Class 5 in mask, but num_classes=3 -> drop class 5 from index
        _, cls_idx = scan_or_load_split(image_dir, mask_dir, num_classes=3)
        assert 4 not in cls_idx, "class 5 must be dropped when num_classes=3"
        # 'c' has classes {3, 5}; only class 3 (idx 2) survives
        assert sorted(cls_idx[2]) == [2]


# ----------------------------------------------------------------------------
# (b) Cache hit on second call
# ----------------------------------------------------------------------------

class TestCacheHit:
    def test_cache_file_created(self, synthetic_split, _isolate_cache_root):
        image_dir, mask_dir = synthetic_split
        scan_or_load_split(image_dir, mask_dir, num_classes=10)
        cache_files = list(_isolate_cache_root.glob("plantseg_split_*.json"))
        assert len(cache_files) == 1, f"expected 1 cache file, got {cache_files}"

    def test_warm_call_does_not_decode_pngs(self, synthetic_split, monkeypatch):
        image_dir, mask_dir = synthetic_split
        names1, idx1 = scan_or_load_split(image_dir, mask_dir, num_classes=10)

        n_decodes = 0
        original_open = Image.open

        def counted_open(*args, **kwargs):
            nonlocal n_decodes
            n_decodes += 1
            return original_open(*args, **kwargs)

        monkeypatch.setattr(Image, "open", counted_open)
        names2, idx2 = scan_or_load_split(image_dir, mask_dir, num_classes=10)

        assert n_decodes == 0, f"warm cache must not decode PNGs (saw {n_decodes})"
        assert names1 == names2
        assert idx1 == idx2

    def test_warm_call_is_fast(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        scan_or_load_split(image_dir, mask_dir, num_classes=10)
        t0 = time.perf_counter()
        for _ in range(50):
            scan_or_load_split(image_dir, mask_dir, num_classes=10)
        avg_ms = (time.perf_counter() - t0) / 50 * 1000
        assert avg_ms < 50, f"warm cache load took {avg_ms:.1f}ms, expected <50ms"


# ----------------------------------------------------------------------------
# (c) Cache invalidation
# ----------------------------------------------------------------------------

class TestInvalidation:
    def test_invalidates_on_added_file(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        names1, idx1 = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        # Add a new (image, mask) pair -- the count fingerprint changes
        _write_pair(image_dir, mask_dir, "f", classes=[7])
        names2, idx2 = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        assert "f" in names2
        assert names2 != names1
        assert 6 in idx2

    def test_invalidates_on_num_classes_change(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        _, idx_full = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        _, idx_three = scan_or_load_split(image_dir, mask_dir, num_classes=3)
        # Class 5 (idx 4) present at num_classes=10, absent at num_classes=3
        assert 4 in idx_full
        assert 4 not in idx_three

    def test_invalidates_on_mtime_change(self, synthetic_split):
        image_dir, mask_dir = synthetic_split
        scan_or_load_split(image_dir, mask_dir, num_classes=10)

        future = time.time() + 3600
        os.utime(mask_dir, (future, future))

        n_decodes = 0
        original_open = Image.open

        def counted_open(*args, **kwargs):
            nonlocal n_decodes
            n_decodes += 1
            return original_open(*args, **kwargs)

        from unittest.mock import patch
        with patch.object(Image, "open", side_effect=counted_open):
            scan_or_load_split(image_dir, mask_dir, num_classes=10)
        assert n_decodes >= 4, "expected re-scan to decode the 4 non-empty masks"

    def test_corrupt_cache_falls_back_to_rescan(self, synthetic_split, _isolate_cache_root):
        image_dir, mask_dir = synthetic_split
        scan_or_load_split(image_dir, mask_dir, num_classes=10)
        cache_file = next(_isolate_cache_root.glob("plantseg_split_*.json"))
        cache_file.write_text("{ this is not valid json")
        names, idx = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        assert names == ["a", "b", "c", "d"]
        assert idx[0] == [0, 3]
        # Cache should now be valid again
        with cache_file.open() as fh:
            assert json.load(fh)["names"] == names


# ----------------------------------------------------------------------------
# (d) ``filter_class_index_to_subset`` correctness
# ----------------------------------------------------------------------------

class TestFilterSubset:
    def test_drops_indices_outside_range(self):
        idx = {0: [1, 5, 9], 1: [2, 3], 2: [11]}
        out = filter_class_index_to_subset(idx, range(5))
        assert out[0] == [1]
        assert out[1] == [2, 3]
        assert out[2] == []

    def test_accepts_set_input(self):
        idx = {0: [1, 5, 9]}
        out = filter_class_index_to_subset(idx, {1, 9})
        assert out[0] == [1, 9]


# ----------------------------------------------------------------------------
# (e) Integration: _PlantSegBase + SiamesePlantSegDataset use the cache
# ----------------------------------------------------------------------------

class TestDatasetIntegration:
    def test_plantseg_base_caches_class_indices(self, tmp_path, monkeypatch):
        """``_PlantSegBase`` exposes the class index for the wrapper to consume."""
        from src.wsss.spdnet.seg_dataset import _PlantSegBase

        root = tmp_path / "ds"
        image_dir = root / "images" / "train"
        mask_dir = root / "annotations" / "train"
        _write_pair(image_dir, mask_dir, "a", classes=[1, 2])
        _write_pair(image_dir, mask_dir, "b", classes=[2])
        _write_pair(image_dir, mask_dir, "c", classes=[3])

        ds = _PlantSegBase(root=root, split="train", image_size=16)
        assert ds.names == ["a", "b", "c"]
        assert hasattr(ds, "_cached_class_to_indices")
        assert ds._cached_class_to_indices[0] == [0]      # cls 1 -> 'a'
        assert ds._cached_class_to_indices[1] == [0, 1]   # cls 2 -> 'a','b'
        assert ds._cached_class_to_indices[2] == [2]      # cls 3 -> 'c'

    def test_limit_filters_cached_indices(self, tmp_path):
        from src.wsss.spdnet.seg_dataset import _PlantSegBase

        root = tmp_path / "ds"
        image_dir = root / "images" / "train"
        mask_dir = root / "annotations" / "train"
        for i, cls in enumerate([1, 1, 2, 2, 3]):
            _write_pair(image_dir, mask_dir, f"x{i}", classes=[cls])

        ds = _PlantSegBase(root=root, split="train", image_size=16, limit=3)
        assert ds.names == ["x0", "x1", "x2"]
        # Class 2 had {2, 3} pre-limit; after limit=3 only idx 2 survives
        assert ds._cached_class_to_indices[1] == [2]
        # Class 3 had {4} pre-limit; after limit=3 nothing survives
        assert ds._cached_class_to_indices.get(2, []) == []

    def test_siamese_consumes_cached_index(self, tmp_path):
        """``SiamesePlantSegDataset._build_index`` must use the cache, not re-scan."""
        from unittest.mock import patch

        from src.wsss.spdnet.dataset import SiamesePlantSegDataset
        from src.wsss.spdnet.seg_dataset import _PlantSegBase

        root = tmp_path / "ds"
        image_dir = root / "images" / "train"
        mask_dir = root / "annotations" / "train"
        for i, cls in enumerate([1, 1, 2, 3, 3]):
            _write_pair(image_dir, mask_dir, f"x{i}", classes=[cls])

        base = _PlantSegBase(root=root, split="train", image_size=16)

        n_decodes = 0
        original_open = Image.open

        def counted_open(*args, **kwargs):
            nonlocal n_decodes
            n_decodes += 1
            return original_open(*args, **kwargs)

        with patch.object(Image, "open", side_effect=counted_open):
            siamese = SiamesePlantSegDataset(base_dataset=base, num_references=1)

        assert n_decodes == 0, (
            f"_build_index should consume cached index without PNG decodes; "
            f"saw {n_decodes}"
        )
        assert sorted(siamese.class_to_indices[0]) == [0, 1]
        assert sorted(siamese.class_to_indices[1]) == [2]
        assert sorted(siamese.class_to_indices[2]) == [3, 4]

    def test_siamese_fallback_when_no_cache_attr(self, tmp_path):
        """Legacy datasets without _cached_class_to_indices still work via the
        original per-mask decoder path."""
        from src.wsss.spdnet.dataset import SiamesePlantSegDataset

        # Minimal duck-typed base lacking the cache attr
        class FakeBase:
            def __init__(self, root: Path) -> None:
                self.root = root
                self.image_dir = root / "images" / "train"
                self.mask_dir = root / "annotations" / "train"
                self.num_classes = 10
                self.names = sorted(p.stem for p in self.image_dir.glob("*.jpg"))
                self._pv_samples: list = []

            def __len__(self) -> int:
                return len(self.names)

            def __getitem__(self, idx: int) -> dict:
                import torch
                return {"image": torch.zeros(3, 16, 16),
                        "label": torch.zeros(self.num_classes),
                        "name": self.names[idx]}

        root = tmp_path / "ds"
        image_dir = root / "images" / "train"
        mask_dir = root / "annotations" / "train"
        _write_pair(image_dir, mask_dir, "a", classes=[1])
        _write_pair(image_dir, mask_dir, "b", classes=[2])

        base = FakeBase(root)
        assert not hasattr(base, "_cached_class_to_indices")
        siamese = SiamesePlantSegDataset(base_dataset=base, num_references=1)
        assert siamese.class_to_indices[0] == [0]
        assert siamese.class_to_indices[1] == [1]


# ----------------------------------------------------------------------------
# (f) Atomic-write safety
# ----------------------------------------------------------------------------

class TestAtomicWrite:
    def test_no_orphan_tmp_files_after_normal_call(self, synthetic_split, _isolate_cache_root):
        image_dir, mask_dir = synthetic_split
        scan_or_load_split(image_dir, mask_dir, num_classes=10)
        tmp_files = list(_isolate_cache_root.glob("*.tmp.*"))
        assert tmp_files == [], f"orphan tmp files: {tmp_files}"

    def test_concurrent_simulation(self, synthetic_split):
        """Two back-to-back scans must not corrupt the cache file."""
        image_dir, mask_dir = synthetic_split
        a_names, a_idx = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        b_names, b_idx = scan_or_load_split(image_dir, mask_dir, num_classes=10)
        assert a_names == b_names
        assert a_idx == b_idx
