"""Hydra configuration for the SPDNet localization-capacity probe."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SPDNetProbeModelConfig:
    fpn_channels: int = 256
    num_classes: int = 115
    head_hidden_dim: int = 64
    target_size_h: int = 448
    target_size_w: int = 448
    position: str = "P3_query_merged"
    freeze_backbone: bool = True
    seg_loss_weight: float = 1.0
    cls_loss_weight: float = 0.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    head_lr: float = 1e-3
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4


@dataclass
class SPDNetProbeDataConfig:
    root: str = "data/plantsegv3"
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    num_references: int = 1
    train_aug: bool = True
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None


@dataclass
class SPDNetProbeTrainerConfig:
    # Phase 1 default. Bumped from 5 -> 20 because the first frozen run
    # (token_n1_heavy/P1_layer4) was visibly under-converged at epoch 5
    # (train/seg_loss still descending, val/IoU@0.5 still rising). Phases
    # 2 and 3 override this from their orchestrator scripts.
    max_epochs: int = 20
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0


@dataclass
class SPDNetProbeConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    experiment_name: str = "spdnet_probe"
    seed: int = 42
    phase: str = "phase1"  # informational tag only

    # Source SPDNet checkpoint to wrap
    checkpoint: str = "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"
    ckpt_tag: str = "token_n1_heavy"

    # Probe + dataset + trainer settings
    model: SPDNetProbeModelConfig = field(default_factory=SPDNetProbeModelConfig)
    data: SPDNetProbeDataConfig = field(default_factory=SPDNetProbeDataConfig)
    trainer: SPDNetProbeTrainerConfig = field(default_factory=SPDNetProbeTrainerConfig)

    # Output / MLflow
    run_name: Optional[str] = None
    output_dir: str = "outputs/spdnet_plantseg/seg_probe_${phase}/${ckpt_tag}/${model.position}"
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "spdnet_seg_probe"

    resume_if_exists: bool = True
