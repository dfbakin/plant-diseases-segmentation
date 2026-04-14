"""Siamese dataset wrapper: returns (query, reference) pairs of the same class.

Wraps ``PlantSegMCTformerDataset`` and pre-builds a per-class index for O(1)
pair sampling at training time.  Supports returning N references per query.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.voc_classification import PlantSegMCTformerDataset


class SiamesePlantSegDataset(Dataset):
    """Returns paired samples that share at least one disease class.

    Each ``__getitem__`` returns::

        {"query": {image, label, name}, "references": [{image, label, name}, ...]}

    Reference images are randomly sampled from the same class as one of
    the query's active labels.  Self-pairing is avoided when possible.
    """

    def __init__(
        self,
        base_dataset: PlantSegMCTformerDataset,
        num_references: int = 1,
    ) -> None:
        self.base = base_dataset
        self.num_references = num_references
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        self._build_index()

    def _build_index(self) -> None:
        """Build class -> [sample indices] mapping from labels only (no images)."""
        num_classes = self.base.num_classes
        n_plantseg = len(self.base.names)

        for idx, name in enumerate(self.base.names):
            mask_path = self.base.mask_dir / f"{name}.png"
            mask = np.array(Image.open(mask_path))
            for cls_idx in np.unique(mask):
                if 1 <= cls_idx <= num_classes:
                    self.class_to_indices[cls_idx - 1].append(idx)

        for pv_idx, (_path, label) in enumerate(self.base._pv_samples):
            idx = n_plantseg + pv_idx
            active = label.nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(active, int):
                active = [active]
            for cls in active:
                self.class_to_indices[cls].append(idx)

    def __len__(self) -> int:
        return len(self.base)

    def _sample_reference(self, idx: int, cls: int) -> int:
        candidates = self.class_to_indices[cls]
        ref_idx = random.choice(candidates)
        while ref_idx == idx and len(candidates) > 1:
            ref_idx = random.choice(candidates)
        return ref_idx

    def __getitem__(self, idx: int) -> dict:
        query = self.base[idx]
        active = query["label"].nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(active, int):
            active = [active]

        cls = random.choice(active)

        refs = []
        seen: set[int] = {idx}
        for _ in range(self.num_references):
            ref_idx = self._sample_reference(idx, cls)
            attempts = 0
            while ref_idx in seen and len(self.class_to_indices[cls]) > len(seen) and attempts < 10:
                ref_idx = self._sample_reference(idx, cls)
                attempts += 1
            seen.add(ref_idx)
            refs.append(self.base[ref_idx])

        return {"query": query, "references": refs}


def siamese_collate_fn(batch: list[dict]) -> dict:
    """Collate paired samples into batched tensors.

    Returns::

        {
            "query_image": (B, 3, H, W),
            "query_label": (B, C),
            "query_name":  [str, ...],
            "ref_images":  list of N tensors, each (B, 3, H, W),
            "ref_labels":  list of N tensors, each (B, C),
            "ref_names":   list of N lists of str,
        }
    """
    n_refs = len(batch[0]["references"])
    return {
        "query_image": torch.stack([s["query"]["image"] for s in batch]),
        "query_label": torch.stack([s["query"]["label"] for s in batch]),
        "query_name": [s["query"]["name"] for s in batch],
        "ref_images": [
            torch.stack([s["references"][i]["image"] for s in batch])
            for i in range(n_refs)
        ],
        "ref_labels": [
            torch.stack([s["references"][i]["label"] for s in batch])
            for i in range(n_refs)
        ],
        "ref_names": [
            [s["references"][i]["name"] for s in batch]
            for i in range(n_refs)
        ],
    }
