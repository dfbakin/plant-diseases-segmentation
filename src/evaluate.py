"""Evaluation and prediction generation script.

Examples:
    # Evaluate on test set
    python src/evaluate.py +checkpoint_path=outputs/.../best.ckpt model=segformer

    # Generate predictions on validation set
    python src/evaluate.py +checkpoint_path=outputs/.../best.ckpt model=segformer \
        +save_predictions=true +predictions_dir=outputs/predictions/segformer_val \
        data.batch_size=1 +split=val
"""

import json
import logging
from pathlib import Path

import cv2
import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.conf.config import Config, register_configs
from src.data import PlantSegDataModule
from src.data.plantseg import PlantSegDataset
from src.data.transforms import get_val_transforms
from src.models import SegmentationModule, create_model

log = logging.getLogger(__name__)

# Register structured configs before hydra.main
register_configs()


def evaluate(cfg: Config) -> dict[str, float]:
    """Evaluate checkpoint on test set. Returns test metrics dict."""
    L.seed_everything(cfg.experiment.seed, workers=True)

    datamodule = PlantSegDataModule(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        mean=list(cfg.data.normalization.mean),
        std=list(cfg.data.normalization.std),
    )

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided via +checkpoint_path=...")

    log.info(f"Loading checkpoint: {checkpoint_path}")

    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        encoder_name=getattr(cfg.model, "encoder_name", "resnet50") or "resnet50",
        encoder_weights=None,
        variant=getattr(cfg.model, "variant", None),
        pretrained=False,
    )

    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path, model=model_backbone
    )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )

    log.info("Running test evaluation...")
    results = trainer.test(module, datamodule=datamodule)
    return results[0] if results else {}


def generate_predictions(cfg: Config) -> None:
    """Generate and save predictions for a dataset split."""
    L.seed_everything(cfg.experiment.seed, workers=True)

    checkpoint_path = cfg.checkpoint_path
    predictions_dir = Path(cfg.predictions_dir)
    split = cfg.split

    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided via +checkpoint_path=...")

    predictions_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = predictions_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    log.info(f"Loading checkpoint: {checkpoint_path}")
    log.info(f"Saving predictions to: {predictions_dir}")
    log.info(f"Split: {split}")

    # Determine device
    if cfg.trainer.accelerator == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info(f"Using device: {device}")

    # Create model
    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        encoder_name=getattr(cfg.model, "encoder_name", "resnet50") or "resnet50",
        encoder_weights=None,
        variant=getattr(cfg.model, "variant", None),
        pretrained=False,
    )

    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path,
        model=model_backbone,
        map_location=device,
    )
    module.eval()
    module.to(device)

    # Create dataset
    transform = get_val_transforms(
        image_size=cfg.data.image_size,
        mean=tuple(cfg.data.normalization.mean),
        std=tuple(cfg.data.normalization.std),
    )

    dataset = PlantSegDataset(
        root=cfg.data.root,
        split=split,
        transform=transform,
    )
    log.info(f"Loaded {len(dataset)} samples from {split} split")

    # Use DataLoader with specified batch size (typically 1)
    from torch.utils.data import DataLoader

    batch_size = cfg.data.batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    metadata = {}

    module.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(data_loader, desc="Generating predictions")
        ):
            images = batch["image"].to(device)
            names = batch["name"]

            # Forward pass
            logits = module(images)
            pred_masks = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            # For each image in batch, save prediction and metadata
            for i in range(images.size(0)):
                name = names[i]
                pred_mask = pred_masks[i]
                index = batch_idx * batch_size + i
                mask_path = masks_dir / f"{name}.png"
                cv2.imwrite(str(mask_path), pred_mask * 255)

                # Store metadata
                metadata[name] = {
                    "prediction_path": str(mask_path),
                    "image_path": str(dataset.samples[index]["image_path"]),
                    "gt_mask_path": str(dataset.samples[index]["mask_path"]),
                }

    # Save metadata
    metadata_path = predictions_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Saved {len(metadata)} predictions to {predictions_dir}")
    log.info(f"Metadata saved to {metadata_path}")


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    save_predictions = cfg.get("save_predictions", False)

    if save_predictions:
        generate_predictions(cfg)  # type: ignore[arg-type]
    else:
        results = evaluate(cfg)  # type: ignore[arg-type]
        log.info(f"Test Results:\n{OmegaConf.to_yaml(results)}")


if __name__ == "__main__":
    main()
