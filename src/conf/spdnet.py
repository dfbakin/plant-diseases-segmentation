"""Hydra configuration dataclasses for SPDNet Siamese training."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SPDNetModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    fpn_channels: int = 256
    num_classes: int = 115
    input_size: int = 448
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    mse_reduction: int = 4
    fusion_mode: str = "token"


@dataclass
class SPDNetDataConfig:
    root: str = "data/plantsegv3"
    pv_root: str = "data/plant-village"
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    include_plantvillage: bool = False
    num_references: int = 1
    augmentation: str = "heavy"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class SPDNetTrainerConfig:
    max_epochs: int = 80
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-5


@dataclass
class SPDNetSpatialLossesConfig:
    """Auxiliary spatial losses + EMA teacher + online localization metric.

    Defaults reproduce the "no aux losses, no online metric" classification
    baseline (every loss weight == 0, online metric kill-switched off);
    enable individually for the experiments described in the plan
    (`spdnet_auxiliary_spatial_losses_*.plan.md`).

    Field groups:

    * Equivariance loss (``lambda_eq``, ``equivariance_transforms``).
    * Patch-contrastive loss (``lambda_con``, ``con_*``, ``con_position``).
    * Self-distillation (``lambda_distill``, ``distill_*``, ``ema_alpha``).
    * Online localization metric (``online_loc_*``).
    """

    # ----- Equivariance loss -----
    lambda_eq: float = 0.0
    # Allowed transform IDs from src.wsss.spdnet.equivariance_transforms:
    # 0=identity, 1=hflip, 2=rot90, 3=rot180, 4=rot270.
    equivariance_transforms: tuple[int, ...] = (1, 2, 3, 4)

    # ----- Patch contrastive loss -----
    lambda_con: float = 0.0
    con_top_K: int = 8
    con_M_negatives: int = 16
    con_temperature: float = 0.07
    con_projection_dim: int = 128
    # Where to take the patch features from. Currently only "P3_query_merged"
    # is implemented (matches the spec).
    con_position: str = "P3_query_merged"
    # Linear warmup for ``lambda_con``. The effective weight used in
    # ``training_step`` is scheduled as::
    #
    #   e < con_warmup_start_epoch                          -> 0
    #   start <= e < start + con_warmup_epochs              -> lambda_con * (e - start) / ramp
    #   e >= start + con_warmup_epochs                      -> lambda_con
    #
    # Defaults ``start=0, ramp=0`` reproduce the original no-warmup
    # behaviour (lambda_con applied in full from epoch 0 onward). A typical
    # "classifier first, then contrastive" recipe uses ``start=14, ramp=7``.
    con_warmup_start_epoch: int = 0
    con_warmup_epochs: int = 0

    # ----- Self-distillation -----
    lambda_distill: float = 0.0
    ema_alpha: float = 0.999
    distill_T_teacher: float = 0.04
    distill_T_student: float = 0.1
    distill_center_beta: float = 0.9
    distill_warmup_epochs: int = 10

    # ----- Online localization metric -----
    online_loc_eval_enabled: bool = True
    online_loc_eval_subset_size: int = 100
    online_loc_eval_seed: int = 1234
    online_loc_eval_every_n_epochs: int = 1
    online_loc_eval_batch_size: int = 8
    online_loc_gt_binary_dir: str = "outputs/plantseg_binary_mc115/gt_binary_val"


@dataclass
class SPDNetConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    experiment_name: str = "spdnet_plantseg"
    seed: int = 42

    model: SPDNetModelConfig = field(default_factory=SPDNetModelConfig)
    data: SPDNetDataConfig = field(default_factory=SPDNetDataConfig)
    trainer: SPDNetTrainerConfig = field(default_factory=SPDNetTrainerConfig)
    losses: SPDNetSpatialLossesConfig = field(
        default_factory=SPDNetSpatialLossesConfig,
    )

    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "${experiment_name}"
    output_dir: str = "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
