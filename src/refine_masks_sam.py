"""Refine pseudomasks using SAM1 (Segment Anything Model).

Prompts SAM1 with existing pseudomasks (and optionally CAM-derived points)
to produce boundary-refined segmentation masks.

Prompt modes:
    mask_only       - feed pseudomask as dense prompt
    mask_and_points - pseudomask + positive/negative points from CAM
    box_only        - bounding box from mask (SAM segments freely inside)
    box_and_points  - bounding box from mask + points from CAM
    points_only     - positive/negative points from CAM only

Mask selection strategies (when multimask_output=True, SAM returns 3 masks):
    best_iou        - pick mask with highest predicted IoU score (default)
    smallest_area   - pick mask with smallest foreground area

Binary and multiclass masks are supported. For multiclass, SAM is
prompted per-class and results are merged by IoU confidence.

Example:
    python src/refine_masks_sam.py \
        image_dir=data/plantsegv3/images/train \
        mask_dir=outputs/plantseg_binary/pseudo_masks \
        output_dir=outputs/plantseg_binary/sam_refined/A1 \
        num_classes=2 prompt_mode=mask_only mask_selection=smallest_area
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from transformers import SamModel, SamProcessor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SAMRefineConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    image_dir: str = ""
    image_ext: str = ".jpg"
    mask_dir: str = ""
    cam_dir: str = ""
    output_dir: str = "outputs/sam_refined"

    labels_file: str = ""
    mask_ext: str = ".png"

    model_name: str = "facebook/sam-vit-huge"

    prompt_mode: str = "mask_only"
    mask_selection: str = "best_iou"
    num_pos_points: int = 3
    num_neg_points: int = 3
    pos_quantile: float = 0.95
    neg_quantile: float = 0.05
    point_min_distance: int = 16

    num_classes: int = 2
    batch_size: int = 8
    min_component_size: int = 0
    multimask_output: bool = True

    device: str = "cuda"


cs = ConfigStore.instance()
cs.store(name="sam_refine_config", node=SAMRefineConfig)


# ---------------------------------------------------------------------------
# Helper: connected-component filtering
# ---------------------------------------------------------------------------

def filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than *min_size* pixels."""
    if min_size <= 0:
        return mask
    out = mask.copy()
    labeled, n = ndimage.label(mask > 0)
    for c in range(1, n + 1):
        if (labeled == c).sum() < min_size:
            out[labeled == c] = 0
    return out


# ---------------------------------------------------------------------------
# Helper: point sampling from CAM
# ---------------------------------------------------------------------------

def sample_points_from_cam(
    cam: np.ndarray,
    mask: np.ndarray,
    num_pos: int = 3,
    num_neg: int = 3,
    pos_quantile: float = 0.95,
    neg_quantile: float = 0.05,
    min_distance: int = 16,
) -> tuple[list[list[int]], list[int]]:
    """Sample spatially diverse positive/negative points from a continuous CAM.

    Returns (points, labels) where each point is [x, y] in image coordinates
    and labels are 1 (positive) or 0 (negative).
    """
    points: list[list[int]] = []
    labels: list[int] = []

    h, w = cam.shape

    # --- Positive points: high activation inside the mask ---
    pos_thresh = np.quantile(cam, pos_quantile)
    pos_candidates = np.argwhere((cam >= pos_thresh) & (mask > 0))  # (N, 2) as (row, col)

    if len(pos_candidates) > 0:
        selected_pos = _farthest_point_sample(pos_candidates, num_pos, min_distance)
        for r, c in selected_pos:
            points.append([int(c), int(r)])  # SAM expects [x, y]
            labels.append(1)

    # --- Negative points: low activation in background ---
    neg_thresh = np.quantile(cam, neg_quantile)
    neg_candidates = np.argwhere((cam <= neg_thresh) & (mask == 0))

    if len(neg_candidates) > 0:
        selected_neg = _farthest_point_sample(neg_candidates, num_neg, min_distance)
        for r, c in selected_neg:
            points.append([int(c), int(r)])
            labels.append(0)

    return points, labels


def _farthest_point_sample(
    candidates: np.ndarray, k: int, min_distance: int
) -> np.ndarray:
    """Greedy farthest-point sampling for spatial diversity.

    *candidates*: (N, 2) array of (row, col) coordinates.
    Returns up to *k* points.
    """
    if k <= 0:
        return np.empty((0, 2), dtype=candidates.dtype)
    if len(candidates) <= k:
        return candidates

    rng = np.random.default_rng(42)
    first_idx = rng.integers(0, len(candidates))
    selected = [candidates[first_idx]]

    for _ in range(k - 1):
        dists = np.min(
            [np.sum((candidates - s) ** 2, axis=1) for s in selected], axis=0
        )
        best_idx = np.argmax(dists)
        if dists[best_idx] < min_distance ** 2:
            break
        selected.append(candidates[best_idx])

    return np.array(selected)


# ---------------------------------------------------------------------------
# Helper: mask → logits for SAM input_masks
# ---------------------------------------------------------------------------

def mask_to_logits(mask: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """Convert a binary mask to low-resolution logits for SAM's input_masks.

    Returns tensor of shape (1, target_size, target_size).
    """
    resized = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (target_size, target_size), Image.NEAREST
        )
    )
    logits = (resized.astype(np.float32) * 2.0 - 1.0) * 6.0
    return torch.from_numpy(logits).unsqueeze(0)


# ---------------------------------------------------------------------------
# Helper: mask → bounding box
# ---------------------------------------------------------------------------

def mask_to_bbox(
    mask: np.ndarray, padding_frac: float = 0.03
) -> list[int] | None:
    """Compute padded bounding box [x_min, y_min, x_max, y_max] from a binary mask.

    Returns None if the mask is empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = mask.shape
    pad_h = int(h * padding_frac)
    pad_w = int(w * padding_frac)

    return [
        max(0, int(cmin) - pad_w),
        max(0, int(rmin) - pad_h),
        min(w, int(cmax) + pad_w + 1),
        min(h, int(rmax) + pad_h + 1),
    ]


# ---------------------------------------------------------------------------
# Core: per-image SAM refinement
# ---------------------------------------------------------------------------

def _build_prompts_for_class(
    binary_mask: np.ndarray,
    cam: np.ndarray | None,
    prompt_mode: str,
    cfg: SAMRefineConfig,
) -> dict[str, Any]:
    """Assemble SAM prompts for a single foreground class.

    Returns dict with optional keys: input_points, input_labels,
    input_boxes, input_masks (all as nested lists / tensors).
    """
    prompts: dict[str, Any] = {}

    if prompt_mode in ("mask_only", "mask_and_points"):
        prompts["input_masks"] = mask_to_logits(binary_mask)

    if prompt_mode in ("box_only", "box_and_points"):
        bbox = mask_to_bbox(binary_mask, padding_frac=0.03)
        if bbox is not None:
            prompts["input_boxes"] = [[bbox]]

    if prompt_mode in ("mask_and_points", "box_and_points", "points_only"):
        if cam is not None:
            pts, lbls = sample_points_from_cam(
                cam, binary_mask,
                num_pos=cfg.num_pos_points,
                num_neg=cfg.num_neg_points,
                pos_quantile=cfg.pos_quantile,
                neg_quantile=cfg.neg_quantile,
                min_distance=cfg.point_min_distance,
            )
            if pts:
                prompts["input_points"] = [[pts]]
                prompts["input_labels"] = [[lbls]]
        elif prompt_mode == "points_only":
            log.warning("points_only mode requires cam_dir; skipping image")

    return prompts


def _select_mask(
    all_masks: np.ndarray,
    iou_scores: torch.Tensor,
    strategy: str,
) -> tuple[int, float]:
    """Choose which of SAM's output masks to keep.

    Args:
        all_masks: (N, H, W) boolean/uint8 array at original resolution.
        iou_scores: (N,) tensor of predicted IoU scores.
        strategy: ``"best_iou"`` or ``"smallest_area"``.

    Returns:
        (selected_index, iou_score_of_selected).
    """
    if strategy == "smallest_area":
        areas = np.array([m.sum() for m in all_masks])
        nonempty = areas > 0
        if nonempty.any():
            idx = int(np.where(nonempty, areas, areas.max() + 1).argmin())
        else:
            idx = int(torch.argmax(iou_scores).item())
        return idx, float(iou_scores[idx].item())

    # default: best_iou
    idx = int(torch.argmax(iou_scores).item())
    return idx, float(iou_scores[idx].item())


def _refine_single_class(
    model: SamModel,
    processor: SamProcessor,
    image_embedding: torch.Tensor,
    original_size: torch.Tensor,
    reshaped_size: torch.Tensor,
    pil_image: Image.Image,
    binary_mask: np.ndarray,
    cam: np.ndarray | None,
    prompt_mode: str,
    cfg: SAMRefineConfig,
    device: torch.device,
) -> tuple[np.ndarray, float] | None:
    """Run SAM on a single class, return (refined_binary_mask, iou_score) or None."""
    prompts = _build_prompts_for_class(binary_mask, cam, prompt_mode, cfg)
    if not prompts:
        return None

    proc_kwargs: dict[str, Any] = {}
    if "input_points" in prompts:
        proc_inputs = processor(
            images=pil_image,
            input_points=prompts["input_points"],
            input_labels=prompts["input_labels"],
            return_tensors="pt",
        )
        proc_kwargs["input_points"] = proc_inputs["input_points"].to(device)
        proc_kwargs["input_labels"] = proc_inputs["input_labels"].to(device)

    if "input_boxes" in prompts:
        proc_inputs = processor(
            images=pil_image,
            input_boxes=prompts["input_boxes"],
            return_tensors="pt",
        )
        proc_kwargs["input_boxes"] = proc_inputs["input_boxes"].to(device)

    if "input_masks" in prompts:
        proc_kwargs["input_masks"] = prompts["input_masks"].unsqueeze(0).to(device)

    outputs = model(
        image_embeddings=image_embedding.unsqueeze(0),
        original_sizes=original_size.unsqueeze(0),
        reshaped_input_sizes=reshaped_size.unsqueeze(0),
        multimask_output=cfg.multimask_output,
        **proc_kwargs,
    )

    iou_scores = outputs.iou_scores[0, 0]

    # Upscale all candidate masks to original resolution, then select
    all_masks_tensor = processor.post_process_masks(
        outputs.pred_masks,
        original_size.unsqueeze(0),
        reshaped_size.unsqueeze(0),
    )
    all_masks_np = all_masks_tensor[0][0].cpu().numpy().astype(np.uint8)  # (N, H, W)

    sel_idx, sel_score = _select_mask(all_masks_np, iou_scores, cfg.mask_selection)
    return all_masks_np[sel_idx], sel_score


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_masks_sam(cfg: SAMRefineConfig) -> None:
    _VALID_MODES = ("mask_only", "mask_and_points", "box_only", "box_and_points", "points_only")
    _VALID_SELECTIONS = ("best_iou", "smallest_area")

    if not cfg.image_dir:
        raise ValueError("image_dir is required")
    if not cfg.mask_dir:
        raise ValueError("mask_dir is required")
    if cfg.prompt_mode not in _VALID_MODES:
        raise ValueError(f"Unknown prompt_mode: {cfg.prompt_mode}. Must be one of {_VALID_MODES}")
    if cfg.mask_selection not in _VALID_SELECTIONS:
        raise ValueError(f"Unknown mask_selection: {cfg.mask_selection}. Must be one of {_VALID_SELECTIONS}")
    if cfg.prompt_mode in ("mask_and_points", "box_and_points", "points_only") and not cfg.cam_dir:
        raise ValueError(f"cam_dir is required for prompt_mode={cfg.prompt_mode}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    log.info(f"Loading SAM model: {cfg.model_name}")
    processor = SamProcessor.from_pretrained(cfg.model_name)
    model = SamModel.from_pretrained(cfg.model_name).to(device).eval()

    image_dir = Path(cfg.image_dir)
    mask_dir = Path(cfg.mask_dir)
    cam_dir = Path(cfg.cam_dir) if cfg.cam_dir else None
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_ext = cfg.mask_ext
    names = sorted(f.stem for f in mask_dir.glob(f"*{mask_ext}"))
    if not names:
        log.error(f"No {mask_ext} masks found in {mask_dir}")
        return

    labels_dict = None
    if cfg.labels_file:
        labels_dict = np.load(cfg.labels_file, allow_pickle=True).item()
        log.info(f"Loaded image-level labels for {len(labels_dict)} images")

    log.info(
        f"SAM refinement: {len(names)} images, model={cfg.model_name}, "
        f"mode={cfg.prompt_mode}, selection={cfg.mask_selection}, "
        f"num_classes={cfg.num_classes}, batch_size={cfg.batch_size}, "
        f"min_component_size={cfg.min_component_size}"
    )

    success = 0
    batch_size = cfg.batch_size

    for batch_start in tqdm(range(0, len(names), batch_size), desc="SAM refine"):
        batch_names = names[batch_start : batch_start + batch_size]
        batch_images: list[Image.Image] = []
        batch_valid: list[bool] = []

        for name in batch_names:
            img_path = image_dir / f"{name}{cfg.image_ext}"
            if img_path.exists():
                batch_images.append(Image.open(img_path).convert("RGB"))
                batch_valid.append(True)
            else:
                batch_images.append(Image.new("RGB", (64, 64)))
                batch_valid.append(False)

        # Batch encode images
        proc_inputs = processor(images=batch_images, return_tensors="pt").to(device)
        image_embeddings = model.get_image_embeddings(proc_inputs["pixel_values"])

        for i, name in enumerate(batch_names):
            if not batch_valid[i]:
                log.warning(f"Image not found: {name}{cfg.image_ext}")
                continue

            pil_img = batch_images[i]
            img_emb = image_embeddings[i]
            orig_size = proc_inputs["original_sizes"][i]
            reshaped_size = proc_inputs["reshaped_input_sizes"][i]

            pred_mask_path = mask_dir / f"{name}{mask_ext}"
            if mask_ext == ".npy":
                pred_mask = np.load(str(pred_mask_path))
            else:
                pred_mask = np.array(Image.open(pred_mask_path))

            if cfg.num_classes == 2:
                # Binary: single foreground class
                fg_mask = (pred_mask == 1).astype(np.uint8)
                fg_mask = filter_small_components(fg_mask, cfg.min_component_size)

                if not fg_mask.any():
                    Image.fromarray(np.zeros_like(pred_mask)).save(
                        str(output_dir / f"{name}.png")
                    )
                    success += 1
                    continue

                cam = None
                if cam_dir is not None:
                    cam_path = cam_dir / f"{name}.npy"
                    if cam_path.exists():
                        cam_data = np.load(str(cam_path), allow_pickle=True).item()
                        cam = cam_data.get(0)

                result = _refine_single_class(
                    model, processor, img_emb, orig_size, reshaped_size,
                    pil_img, fg_mask, cam, cfg.prompt_mode, cfg, device,
                )

                if result is not None:
                    refined, _ = result
                    out = np.where(refined > 0, 1, 0).astype(np.uint8)
                else:
                    out = fg_mask

                Image.fromarray(out).save(str(output_dir / f"{name}.png"))
                success += 1

            else:
                # Multiclass: per-class refinement
                present_classes = set(np.unique(pred_mask)) - {0, 255}
                if labels_dict is not None and name in labels_dict:
                    valid_fg = set(np.where(labels_dict[name] > 0)[0] + 1)
                    present_classes &= valid_fg

                class_masks: list[tuple[int, np.ndarray, float]] = []

                for cls_id in sorted(present_classes):
                    cls_mask = (pred_mask == cls_id).astype(np.uint8)
                    cls_mask = filter_small_components(cls_mask, cfg.min_component_size)
                    if not cls_mask.any():
                        continue

                    cam = None
                    if cam_dir is not None:
                        cam_path = cam_dir / f"{name}.npy"
                        if cam_path.exists():
                            cam_data = np.load(str(cam_path), allow_pickle=True).item()
                            cam = cam_data.get(cls_id - 1)

                    result = _refine_single_class(
                        model, processor, img_emb, orig_size, reshaped_size,
                        pil_img, cls_mask, cam, cfg.prompt_mode, cfg, device,
                    )

                    if result is not None:
                        refined, score = result
                        class_masks.append((cls_id, refined, score))
                    else:
                        class_masks.append((cls_id, cls_mask, 0.0))

                # Merge: resolve conflicts by IoU confidence
                h, w = pred_mask.shape
                out = np.zeros((h, w), dtype=np.uint8)
                confidence = np.full((h, w), -1.0, dtype=np.float32)

                for cls_id, refined, score in class_masks:
                    better = (refined > 0) & (score > confidence)
                    out[better] = cls_id
                    confidence[better] = score

                Image.fromarray(out).save(str(output_dir / f"{name}.png"))
                success += 1

    log.info(f"Saved {success}/{len(names)} refined masks to {output_dir}")


@hydra.main(version_base=None, config_name="sam_refine_config")
def main(cfg: DictConfig) -> None:
    refine_masks_sam(cfg)


if __name__ == "__main__":
    main()
