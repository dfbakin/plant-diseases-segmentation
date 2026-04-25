"""Training-time localization quality probe for SPDNet.

Cheap "mAP up, IoU down" detector. See RESEARCH_CONTEXT.md §5.11 and the
auxiliary-spatial-losses plan for the contract. Logs three scalars every
``every_n_epochs`` validation epochs:

* ``val/cam_iou_best``     -- max over a 21-point threshold sweep.
* ``val/cam_iou_best_thr`` -- argmax. Sharpness diagnostic: a low optimal
  threshold means diffuse activations; high means concentrated peaks.
* ``val/cam_iou_auc``      -- trapezoidal AUC of ``IoU(tau)`` over
  ``tau in [0, 1]``, normalised to [0, 1]. Threshold-agnostic summary.

Why bother? Phase F's frozen-probe pipeline is the ground truth for
localization, but it's expensive (~minutes per checkpoint). This online
metric is ~30s/eval and detects the exact "mAP up, IoU down" failure that
burned ~7 h on ``spdnet_spatial_n1_ps_pv``. Disable via the kill switch
``losses.online_loc_eval_enabled = false`` if it becomes distracting.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps

from src.wsss.spdnet._split_index_cache import scan_or_load_split

if TYPE_CHECKING:  # pragma: no cover
    from src.wsss.spdnet.model import SPDNet


# ---------------------------------------------------------------------------
# 21-point threshold sweep, defined as a tensor constant for reproducibility.
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: torch.Tensor = torch.linspace(0.0, 1.0, 21)


# ---------------------------------------------------------------------------
# Pure functions: sweep + summary. No model, no I/O. Easy to test on synthetic
# CAMs.
# ---------------------------------------------------------------------------


def compute_iou_sweep(
    cams: torch.Tensor,
    masks: torch.Tensor,
    thresholds: torch.Tensor = DEFAULT_THRESHOLDS,
) -> torch.Tensor:
    """Mean per-image IoU at each threshold.

    Args:
        cams: ``(N, H, W)`` per-image CAM, expected min-max normalised into
            ``[0, 1]`` (per image). Larger value = more disease.
        masks: ``(N, H, W)`` binary GT mask at the same resolution as ``cams``.
            Values must be in ``{0, 1}``.
        thresholds: ``(T,)`` thresholds in ``[0, 1]``, ascending. Defaults to
            ``DEFAULT_THRESHOLDS`` (21 points, step 0.05).

    Returns:
        ``(T,)`` tensor with the mean per-image IoU at each threshold.
    """
    if cams.shape != masks.shape:
        raise ValueError(
            f"cams shape {tuple(cams.shape)} and masks shape "
            f"{tuple(masks.shape)} must agree"
        )
    if cams.dim() != 3:
        raise ValueError(
            f"cams/masks must be (N, H, W); got cams shape {tuple(cams.shape)}"
        )

    masks_b = (masks > 0).float()
    T_n = thresholds.numel()
    ious_per_thr = torch.zeros(T_n, device=cams.device, dtype=torch.float32)
    for ti in range(T_n):
        tau = thresholds[ti].item()
        pred = (cams >= tau).float()
        inter = (pred * masks_b).sum(dim=(1, 2))
        union = ((pred + masks_b) > 0).float().sum(dim=(1, 2))
        # If both pred and GT are empty for an image, treat that image as a
        # perfect match (IoU=1). This avoids penalising the trivial "no
        # disease" case and matches the convention used by Phase F probes.
        iou = torch.where(
            union > 0,
            inter / union.clamp_min(1.0),
            torch.ones_like(union),
        )
        ious_per_thr[ti] = iou.mean()
    return ious_per_thr


def summarize_iou_sweep(
    ious_per_thr: torch.Tensor,
    thresholds: torch.Tensor = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """``cam_iou_best``, ``cam_iou_best_thr``, ``cam_iou_auc``.

    AUC is the trapezoidal integral of ``IoU(tau)`` over ``tau in [0, 1]``,
    so it's already normalised to ``[0, 1]`` (since IoU itself is).

    Tie-break for ``cam_iou_best_thr``: if several thresholds are tied at
    the maximum IoU, return the HIGHEST one. This matches the diagnostic
    intent of the metric -- a sharper localization is preferred when both
    are equally accurate -- and gives the right signal to the
    "best_thr stuck near 0 -> diffuse activation" rule of thumb.
    """
    if ious_per_thr.shape != thresholds.shape:
        raise ValueError(
            f"ious_per_thr shape {tuple(ious_per_thr.shape)} and thresholds "
            f"shape {tuple(thresholds.shape)} must agree"
        )
    if thresholds.numel() < 2:
        raise ValueError("need >= 2 thresholds for AUC")

    best_iou = float(ious_per_thr.max().item())
    # Highest threshold whose IoU equals the best.
    eps = 1e-9
    tie_mask = (ious_per_thr - best_iou).abs() <= eps
    best_idx = int(torch.nonzero(tie_mask).max().item())
    auc = float(torch.trapz(ious_per_thr, thresholds).item())
    return {
        "cam_iou_best": float(ious_per_thr[best_idx].item()),
        "cam_iou_best_thr": float(thresholds[best_idx].item()),
        "cam_iou_auc": auc,
    }


# ---------------------------------------------------------------------------
# Subset selection: deterministic, hash-stable across machines.
# ---------------------------------------------------------------------------


def _stable_sort_key(name: str, seed: int) -> int:
    """Hash-based sort key that's stable across Python interpreters and
    OSes (unlike ``hash()`` which uses PYTHONHASHSEED)."""
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def select_deterministic_subset(
    candidate_names: Iterable[str],
    subset_size: int,
    seed: int = 1234,
) -> list[str]:
    """Pick ``subset_size`` names from ``candidate_names`` deterministically.

    Sorts by ``_stable_sort_key(name, seed)`` ascending and takes the
    prefix; same seed -> same subset across processes/machines.
    """
    names = list(candidate_names)
    names.sort(key=lambda n: _stable_sort_key(n, seed))
    return names[:subset_size]


def first_per_class_references(
    train_class_to_indices: dict[int, list[int]],
    train_names: list[str],
    num_classes: int,
) -> dict[int, str]:
    """For each class, return the alphabetically-first train image stem
    whose mask contains that class.

    Classes with zero positive train samples are silently dropped from the
    output (so the caller can decide how to handle them).
    """
    out: dict[int, str] = {}
    for c in range(num_classes):
        idxs = train_class_to_indices.get(c, [])
        if not idxs:
            continue
        # Alphabetically-first stem -> bit-stable across processes.
        names_c = sorted(train_names[i] for i in idxs)
        out[c] = names_c[0]
    return out


# ---------------------------------------------------------------------------
# OnlineCAMIoU: owns subset, references, transforms, GT masks, and the eval
# loop. Lightning module calls ``should_run(epoch)`` then ``evaluate(model,
# device)``.
# ---------------------------------------------------------------------------


def _build_eval_transform(image_size: int) -> A.Compose:
    """Same normalization as the training pipeline; deterministic resize."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _open_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.array(im)


def _open_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        return (np.array(im) > 0).astype(np.uint8)


class OnlineCAMIoU:
    """Lightweight online localization quality probe.

    Holds, after construction:

    * ``self.query_names``: ``subset_size`` validation stems (deterministic).
    * ``self.query_images``: ``(N, 3, H, W)`` normalized query tensors.
    * ``self.query_masks``: ``(N, H, W)`` binary GT masks at training res.
    * ``self.query_labels``: ``(N, C)`` multi-label one-hot.
    * ``self.ref_images``: ``(N, 3, H, W)`` per-query reference tensor (one
      same-class reference per query, picked deterministically as the
      alphabetically-first train image of the query's first active class).

    Memory: 100 images at 448x448x3 float32 ~ 230 MB total (queries + refs +
    masks). Acceptable for a single GPU process.

    Args:
        plantseg_root: PlantSeg dataset root (``data/plantsegv3``).
        gt_binary_dir: Directory of binary GT masks
            (``outputs/plantseg_binary_mc115/gt_binary_val``).
        num_classes: Number of foreground classes (115 for PlantSeg).
        subset_size: How many val images to evaluate every K epochs (100).
        seed: Subset selection seed (1234).
        every_n_epochs: Run cadence (1 = every epoch).
        image_size: Resize to this size for the model forward (and for IoU
            computation; GT is also resized here).
        thresholds: Threshold sweep; defaults to
            :data:`DEFAULT_THRESHOLDS` (21 points).
        enabled: Master kill switch. If ``False``, ``should_run`` always
            returns ``False`` and ``evaluate`` returns an empty dict.
        eval_batch_size: Forward this many query+reference pairs per batch.
    """

    def __init__(
        self,
        plantseg_root: str | Path,
        gt_binary_dir: str | Path,
        num_classes: int = 115,
        subset_size: int = 100,
        seed: int = 1234,
        every_n_epochs: int = 1,
        image_size: int = 448,
        thresholds: torch.Tensor | None = None,
        enabled: bool = True,
        eval_batch_size: int = 8,
    ) -> None:
        self.enabled = enabled
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.subset_size = int(subset_size)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.eval_batch_size = int(eval_batch_size)
        self.thresholds = (
            thresholds.clone() if thresholds is not None else DEFAULT_THRESHOLDS.clone()
        )
        if self.thresholds.numel() < 2:
            raise ValueError("need >= 2 thresholds for AUC")

        if not self.enabled:
            # Skip all I/O so the kill switch is truly free.
            self.query_names = []
            return

        plantseg_root = Path(plantseg_root)
        gt_binary_dir = Path(gt_binary_dir)
        val_image_dir = plantseg_root / "images" / "val"
        val_mask_dir = plantseg_root / "annotations" / "val"
        train_image_dir = plantseg_root / "images" / "train"
        train_mask_dir = plantseg_root / "annotations" / "train"

        for p in (val_image_dir, val_mask_dir, train_image_dir, train_mask_dir):
            if not p.exists():
                raise FileNotFoundError(f"OnlineCAMIoU: missing required dir {p}")
        if not gt_binary_dir.exists():
            raise FileNotFoundError(
                f"OnlineCAMIoU: GT binary mask dir missing: {gt_binary_dir}. "
                "Run `dvc pull outputs/plantseg_binary_mc115/gt_binary_val.dvc` first."
            )

        # 1) Val candidates: only stems with a binary GT mask on disk.
        val_names, _ = scan_or_load_split(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            num_classes=self.num_classes,
        )
        candidates = [n for n in val_names if (gt_binary_dir / f"{n}.png").exists()]
        if not candidates:
            raise FileNotFoundError(
                f"OnlineCAMIoU: no candidate val stems with GT in {gt_binary_dir}"
            )
        self.query_names = select_deterministic_subset(
            candidates, self.subset_size, self.seed,
        )

        # 2) Train references: first stem per class.
        train_names, train_class_to_indices = scan_or_load_split(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            num_classes=self.num_classes,
        )
        ref_lookup = first_per_class_references(
            train_class_to_indices, train_names, self.num_classes,
        )
        if not ref_lookup:
            raise RuntimeError(
                "OnlineCAMIoU: no train references available -- check num_classes"
            )

        # 3) Pre-load + transform the whole subset (queries + refs + masks)
        # once. This is intentional: the subset is small (100 images) and
        # avoiding I/O every epoch saves ~10 s per eval.
        transform = _build_eval_transform(self.image_size)

        query_images: list[torch.Tensor] = []
        query_masks: list[torch.Tensor] = []
        query_labels: list[torch.Tensor] = []
        ref_images: list[torch.Tensor] = []

        for stem in self.query_names:
            img = _open_rgb(val_image_dir / f"{stem}.jpg")
            mc_mask = _open_rgb(val_mask_dir / f"{stem}.png")[..., 0]
            gt_binary = _open_binary_mask(gt_binary_dir / f"{stem}.png")

            # Multi-label from the multi-class mask, dropping bg=0 / ignore=255.
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            for cls_idx in np.unique(mc_mask):
                if 1 <= cls_idx <= self.num_classes:
                    label[int(cls_idx) - 1] = 1.0
            query_labels.append(label)

            # Resize image+GT mask jointly so they're aligned at training res.
            tfm = A.Compose([
                A.Resize(self.image_size, self.image_size, interpolation=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], additional_targets={"mask": "mask"})
            out = tfm(image=img, mask=gt_binary)
            query_images.append(out["image"])
            mask_t = out["mask"]
            if isinstance(mask_t, np.ndarray):
                mask_t = torch.from_numpy(mask_t)
            query_masks.append(mask_t.float())

            # Pick the first active class as the reference key. This is
            # deterministic given the labels, so the metric is repeatable.
            active = torch.nonzero(label, as_tuple=False).squeeze(-1)
            ref_class = int(active.min().item()) if active.numel() > 0 else 0
            ref_stem = ref_lookup.get(ref_class)
            if ref_stem is None:
                # No train reference for this class -- pick any class that has
                # one. Falls back deterministically to the smallest available
                # class index.
                ref_stem = ref_lookup[min(ref_lookup)]
            ref_img = _open_rgb(train_image_dir / f"{ref_stem}.jpg")
            ref_out = transform(image=ref_img)
            ref_images.append(ref_out["image"])

        self.query_images = torch.stack(query_images)              # (N, 3, H, W)
        self.query_masks = torch.stack(query_masks)                # (N, H, W)
        self.query_labels = torch.stack(query_labels)              # (N, C)
        self.ref_images = torch.stack(ref_images)                  # (N, 3, H, W)

    # ---------------------- public API ----------------------

    def should_run(self, epoch: int) -> bool:
        """``True`` iff enabled AND ``(epoch + 1) % every_n_epochs == 0``.

        Cadence is keyed off ``epoch + 1`` so that the FIRST evaluated epoch
        (epoch=0 with every_n_epochs=1) is included; useful for debugging
        smoke runs.
        """
        if not self.enabled:
            return False
        return (epoch + 1) % self.every_n_epochs == 0

    @torch.no_grad()
    def evaluate(
        self,
        model: "SPDNet",
        device: torch.device,
    ) -> dict[str, float]:
        """Run the full subset through the model and return the three
        scalars ``cam_iou_best``, ``cam_iou_best_thr``, ``cam_iou_auc``.

        Returns an empty dict if ``self.enabled`` is False.
        """
        if not self.enabled or not self.query_names:
            return {}

        was_training = model.training
        model.eval()
        try:
            cams = self._compute_cams(model, device)            # (N, H, W) on CPU
        finally:
            if was_training:
                model.train()

        masks = self.query_masks                                # (N, H, W) on CPU
        ious_per_thr = compute_iou_sweep(cams, masks, self.thresholds)
        return summarize_iou_sweep(ious_per_thr, self.thresholds)

    # ---------------------- internals ----------------------

    @torch.no_grad()
    def _compute_cams(
        self, model: "SPDNet", device: torch.device,
    ) -> torch.Tensor:
        """Forward the subset, build per-image CAM, normalise to [0, 1].

        Returns:
            ``(N, H, W)`` float32 CPU tensor of per-image CAMs in ``[0, 1]``.
        """
        N = self.query_images.shape[0]
        H = W = self.image_size
        all_cams = torch.zeros(N, H, W, dtype=torch.float32)

        for start in range(0, N, self.eval_batch_size):
            stop = min(start + self.eval_batch_size, N)
            q = self.query_images[start:stop].to(device, non_blocking=True)
            r = self.ref_images[start:stop].to(device, non_blocking=True)
            labels_b = self.query_labels[start:stop].to(device, non_blocking=True)

            feats = model.extract_merged_features(q, [r])
            fused = feats["fused"]                              # (B, C_in, H', W')
            cls_w = model.classifier.weight                     # (C, C_in)
            S = torch.einsum("nc,bchw->bnhw", cls_w, fused)     # (B, C, H', W')

            # Mask out unlabelled classes with -inf so max() picks only from
            # active ones; then take argmax over labelled classes.
            label_mask = labels_b.bool().unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            S_masked = S.masked_fill(~label_mask, float("-inf"))
            cam = S_masked.max(dim=1).values                    # (B, H', W')

            # Resize to training res for IoU vs GT.
            cam_full = F.interpolate(
                cam.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(1)                                        # (B, H, W)

            # Per-image min-max normalise to [0, 1].
            cam_full = cam_full.float()
            mn = cam_full.amin(dim=(1, 2), keepdim=True)
            mx = cam_full.amax(dim=(1, 2), keepdim=True)
            cam_norm = (cam_full - mn) / (mx - mn + 1e-8)

            all_cams[start:stop] = cam_norm.detach().cpu()
        return all_cams
