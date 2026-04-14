"""Generate ADPL-CAMs from a trained SPDNet checkpoint.

Uses reference-guided token fusion for improved disease localization.
Supports multi-scale, flip augmentation, binary aggregation, and
optional threshold sweep evaluation.

Example:
    python src/generate_spdnet_cams.py \
        checkpoint=outputs/spdnet_plantseg/.../checkpoints/last.ckpt \
        label_file=outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy \
        image_dir=data/plantsegv3/images/val
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

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.spdnet.cam_generator import generate_all_cams, load_spdnet_from_checkpoint

log = logging.getLogger(__name__)


@dataclass
class GenSPDNetCAMConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    checkpoint: str = ""
    image_dir: str = "data/plantsegv3/images/val"
    image_ext: str = ".jpg"
    label_file: str = "outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy"
    output_dir: str = "outputs/spdnet_plantseg/cams/cam_npy_val"

    num_classes: int = 115
    fpn_channels: int = 256
    mse_reduction: int = 4
    input_size: int = 448
    max_size: int = 0
    scales: list[float] = field(default_factory=lambda: [1.0, 0.75, 1.25])
    num_ref_images: int = 1
    binary_aggregate: str = "max"

    gt_dir: str = "outputs/plantseg_binary_mc115/gt_binary_val"
    eval_threshold_sweep: bool = True
    eval_sweep_samples: int = 0
    eval_optimize_metric: str = "disease_iou"


cs = ConfigStore.instance()
cs.store(name="gen_spdnet_cam_config", node=GenSPDNetCAMConfig)


def generate_cams(cfg: GenSPDNetCAMConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading SPDNet from {cfg.checkpoint}")
    model = load_spdnet_from_checkpoint(
        cfg.checkpoint, cfg.num_classes, cfg.fpn_channels, cfg.mse_reduction
    ).to(device)
    model.eval()

    labels = np.load(cfg.label_file, allow_pickle=True).item()
    log.info(f"Generating CAMs for {len(labels)} images")

    processed = generate_all_cams(
        model=model,
        label_dict=labels,
        image_dir=Path(cfg.image_dir),
        output_dir=Path(cfg.output_dir),
        image_ext=cfg.image_ext,
        scales=list(cfg.scales),
        max_size=cfg.max_size,
        input_size=cfg.input_size,
        num_ref_images=cfg.num_ref_images,
        binary_aggregate=cfg.binary_aggregate,
        device=device,
    )

    if cfg.eval_threshold_sweep and cfg.gt_dir:
        eval_num_cls = 2 if cfg.binary_aggregate else cfg.num_classes + 1
        log.info(
            f"Running threshold sweep (optimize={cfg.eval_optimize_metric}, "
            f"num_cls={eval_num_cls})..."
        )
        result = evaluate_cam_threshold_sweep(
            predict_dir=cfg.output_dir,
            gt_dir=cfg.gt_dir,
            name_list=processed,
            num_cls=eval_num_cls,
            max_samples=cfg.eval_sweep_samples,
            optimize_metric=cfg.eval_optimize_metric,
        )
        best_all = result.get("result_at_best", {})
        parts = [f"{k}={v:.2f}%" for k, v in best_all.items()]
        log.info(
            f"Best threshold={result['best_threshold']:.2f}  " + "  ".join(parts)
        )


@hydra.main(version_base=None, config_name="gen_spdnet_cam_config")
def main(cfg: DictConfig) -> None:
    generate_cams(cfg)


if __name__ == "__main__":
    main()
