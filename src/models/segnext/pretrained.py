"""Pretrained weight loading utilities for MSCAN/SegNeXt.

Handles loading of official mmsegmentation ImageNet-1K pretrained weights
into the pure PyTorch MSCAN implementation.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


MSCANVariant = Literal["tiny", "small", "base", "large"]

# Mapping from variant to checkpoint filename
MSCAN_CHECKPOINT_NAMES: dict[MSCANVariant, str] = {
    "tiny": "mscan_t.pth",
    "small": "mscan_s.pth",
    "base": "mscan_b.pth",
    "large": "mscan_l.pth",
}

# Default pretrained weights directory (relative to project root)
DEFAULT_PRETRAINED_DIR = Path("pretrained/mscan")


def _extract_encoder_state_dict(
    checkpoint: dict,
) -> dict[str, torch.Tensor]:
    """Extract encoder weights from mmseg checkpoint.

    The mmseg checkpoint contains:
    - 'state_dict' wrapper with model weights
    - 'head.*' keys for ImageNet classification (not needed)
    - All other keys are for MSCAN encoder

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        Filtered state dict containing only encoder weights.
    """
    # Extract state_dict if wrapped
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Filter out classification head (not needed for segmentation)
    encoder_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("head.")
    }

    return encoder_state_dict


def load_pretrained_mscan(
    model: nn.Module,
    variant: MSCANVariant,
    checkpoint_path: str | Path | None = None,
    pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load pretrained MSCAN weights into model's encoder.

    Loads official mmsegmentation ImageNet-1K pretrained weights.
    The checkpoint keys match our implementation directly.

    Args:
        model: SegNeXt model with MSCAN encoder, or MSCAN model directly.
        variant: MSCAN variant ('tiny', 'small', 'base', 'large').
        checkpoint_path: Optional explicit path to checkpoint file.
            If None, looks in pretrained_dir.
        pretrained_dir: Directory containing pretrained checkpoints.
        strict: If True, raises error on missing/unexpected keys.

    Returns:
        Tuple of (loaded_keys, missing_keys).

    Raises:
        ValueError: If variant is unknown.
        FileNotFoundError: If checkpoint file not found.
        RuntimeError: If strict=True and keys don't match.
    """
    if variant not in MSCAN_CHECKPOINT_NAMES:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Available: {list(MSCAN_CHECKPOINT_NAMES.keys())}"
        )

    # Determine checkpoint path
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = Path(pretrained_dir) / MSCAN_CHECKPOINT_NAMES[variant]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}\n"
            f"Please download MSCAN-{variant.upper()} weights to {ckpt_path}"
        )

    logger.info(f"Loading pretrained MSCAN-{variant.upper()} from {ckpt_path}")

    # Load checkpoint (weights_only=False because mmseg includes metadata)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract encoder weights
    pretrained_state_dict = _extract_encoder_state_dict(checkpoint)

    # Get target encoder
    if hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        encoder = model

    model_state_dict = encoder.state_dict()

    # Compare keys (excluding running stats buffers for logging)
    pretrained_keys = set(pretrained_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    # Keys that will be loaded
    loadable_keys = pretrained_keys & model_keys
    missing_keys = list(model_keys - pretrained_keys)
    unexpected_keys = list(pretrained_keys - model_keys)

    # Check shapes match
    shape_mismatches = []
    for key in loadable_keys:
        if pretrained_state_dict[key].shape != model_state_dict[key].shape:
            shape_mismatches.append(
                f"{key}: pretrained={pretrained_state_dict[key].shape}, "
                f"model={model_state_dict[key].shape}"
            )

    if shape_mismatches:
        msg = f"Shape mismatches found:\n" + "\n".join(shape_mismatches)
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    # Log loading info
    param_keys = [k for k in loadable_keys
                  if "running" not in k and "num_batches" not in k]
    logger.info(
        f"Loading {len(param_keys)} parameter tensors, "
        f"{len(missing_keys)} missing, {len(unexpected_keys)} unexpected"
    )

    if missing_keys:
        # Filter out running stats from missing (these are expected)
        critical_missing = [k for k in missing_keys
                           if "running" not in k and "num_batches" not in k]
        if critical_missing:
            msg = f"Missing keys in pretrained: {critical_missing}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    # Load weights
    load_result = encoder.load_state_dict(pretrained_state_dict, strict=False)

    return list(loadable_keys), load_result.missing_keys


def get_pretrained_checkpoint_path(
    variant: MSCANVariant,
    pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
) -> Path:
    """Get path to pretrained checkpoint file.

    Args:
        variant: MSCAN variant.
        pretrained_dir: Directory containing pretrained checkpoints.

    Returns:
        Path to checkpoint file.

    Raises:
        ValueError: If variant is unknown.
    """
    if variant not in MSCAN_CHECKPOINT_NAMES:
        raise ValueError(f"Unknown variant: {variant}")

    return Path(pretrained_dir) / MSCAN_CHECKPOINT_NAMES[variant]


def check_pretrained_available(
    variant: MSCANVariant,
    pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
) -> bool:
    """Check if pretrained weights are available for a variant.

    Args:
        variant: MSCAN variant.
        pretrained_dir: Directory to check.

    Returns:
        True if checkpoint exists, False otherwise.
    """
    try:
        path = get_pretrained_checkpoint_path(variant, pretrained_dir)
        return path.exists()
    except ValueError:
        return False

