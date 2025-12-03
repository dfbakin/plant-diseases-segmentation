"""Pretrained weight loading for MSCAN/SegNeXt.

Loads official mmsegmentation ImageNet-1K pretrained weights.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


MSCANVariant = Literal["tiny", "small", "base", "large"]

MSCAN_CHECKPOINT_NAMES: dict[MSCANVariant, str] = {
    "tiny": "mscan_t.pth",
    "small": "mscan_s.pth",
    "base": "mscan_b.pth",
    "large": "mscan_l.pth",
}

DEFAULT_PRETRAINED_DIR = Path("pretrained/mscan")


def _extract_encoder_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract encoder weights, filtering out classification head."""
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k: v for k, v in state_dict.items() if not k.startswith("head.")}


def load_pretrained_mscan(
    model: nn.Module,
    variant: MSCANVariant,
    checkpoint_path: str | Path | None = None,
    pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load pretrained MSCAN weights into model's encoder.

    Returns (loaded_keys, missing_keys). Raises FileNotFoundError if checkpoint missing.
    """
    if variant not in MSCAN_CHECKPOINT_NAMES:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(MSCAN_CHECKPOINT_NAMES.keys())}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(pretrained_dir) / MSCAN_CHECKPOINT_NAMES[variant]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Download MSCAN-{variant.upper()} weights.")

    logger.info(f"Loading pretrained MSCAN-{variant.upper()} from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained_state_dict = _extract_encoder_state_dict(checkpoint)

    encoder = model.encoder if hasattr(model, "encoder") else model
    model_state_dict = encoder.state_dict()

    pretrained_keys = set(pretrained_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    loadable_keys = pretrained_keys & model_keys
    missing_keys = list(model_keys - pretrained_keys)
    unexpected_keys = list(pretrained_keys - model_keys)

    # Check shapes
    shape_mismatches = [
        f"{k}: pretrained={pretrained_state_dict[k].shape}, model={model_state_dict[k].shape}"
        for k in loadable_keys if pretrained_state_dict[k].shape != model_state_dict[k].shape
    ]
    if shape_mismatches:
        msg = "Shape mismatches:\n" + "\n".join(shape_mismatches)
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    param_keys = [k for k in loadable_keys if "running" not in k and "num_batches" not in k]
    logger.info(f"Loading {len(param_keys)} params, {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")

    critical_missing = [k for k in missing_keys if "running" not in k and "num_batches" not in k]
    if critical_missing:
        msg = f"Missing keys: {critical_missing}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    load_result = encoder.load_state_dict(pretrained_state_dict, strict=False)
    return list(loadable_keys), load_result.missing_keys


def get_pretrained_checkpoint_path(variant: MSCANVariant, pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR) -> Path:
    if variant not in MSCAN_CHECKPOINT_NAMES:
        raise ValueError(f"Unknown variant: {variant}")
    return Path(pretrained_dir) / MSCAN_CHECKPOINT_NAMES[variant]


def check_pretrained_available(variant: MSCANVariant, pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR) -> bool:
    try:
        return get_pretrained_checkpoint_path(variant, pretrained_dir).exists()
    except ValueError:
        return False

