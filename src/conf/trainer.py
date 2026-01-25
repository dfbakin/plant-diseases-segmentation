from dataclasses import dataclass, field


@dataclass
class CheckpointConfig:
    monitor: str = "val/miou"
    mode: str = "max"
    save_top_k: int = 2
    save_last: bool = True


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val/miou"
    patience: int = 15
    mode: str = "max"
    min_epochs: int = 20


@dataclass
class TrainerConfig:
    max_epochs: int = 60
    min_epochs: int = 10

    accelerator: str = "auto"  # auto, gpu, cpu, mps
    devices: str = "auto"  # "auto" detects available GPUs, or set to specific number
    strategy: str = "auto"  # "auto" picks ddp for multi-gpu, or "ddp"
    precision: str = "32"  # 16-mixed, bf16-mixed, 32

    accumulate_grad_batches: int = 1

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    log_every_n_steps: int = 50

    val_check_interval: float = 1.0  # Run validation every epoch
    check_val_every_n_epoch: int = 1

    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"

    deterministic: bool = False  # Set true for full reproducibility (slower)
