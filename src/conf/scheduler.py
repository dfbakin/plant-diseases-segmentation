"""Learning rate scheduler configurations."""

from dataclasses import dataclass
from typing import Any

import torch.optim.lr_scheduler as lr_scheduler


@dataclass
class SchedulerConfig:
    name: str = "cosine"


@dataclass
class ConstantSchedulerConfig(SchedulerConfig):
    name: str = "constant"


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    name: str = "cosine"
    eta_min: float = 1e-6


@dataclass
class StepSchedulerConfig(SchedulerConfig):
    name: str = "step"
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class CyclicSchedulerConfig(SchedulerConfig):
    name: str = "cyclic"
    base_lr: float = 1e-6  # min LR
    max_lr_factor: float = 1.0  # max_lr = learning_rate * max_lr_factor
    step_size_up: int = 10  # epochs to go from base to max
    mode: str = "triangular2"  # triangular, triangular2, exp_range
    gamma: float = 0.99  # decay factor for exp_range mode


@dataclass
class OneCycleSchedulerConfig(SchedulerConfig):
    name: str = "one_cycle"
    max_lr_factor: float = 10.0  # peak LR = learning_rate * max_lr_factor
    pct_start: float = 0.3  # fraction of training to increase LR
    div_factor: float = 25.0  # initial_lr = max_lr / div_factor
    final_div_factor: float = 1e4  # final_lr = max_lr / final_div_factor


@dataclass
class PlateauSchedulerConfig(SchedulerConfig):
    name: str = "plateau"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6


def create_scheduler(
    optimizer: Any,
    config: SchedulerConfig,
    max_epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
) -> dict[str, Any] | None:
    """Create scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
        max_epochs: Total training epochs
        steps_per_epoch: Number of batches per epoch
        learning_rate: Base learning rate from optimizer

    Returns:
        Lightning scheduler dict or None for constant LR
    """
    name = config.name.lower()

    if name == "constant":
        return None

    if name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=config.eta_min,
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    if name == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    if name == "cyclic":
        max_lr = learning_rate * config.max_lr_factor
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.base_lr,
            max_lr=max_lr,
            step_size_up=config.step_size_up * steps_per_epoch,
            mode=config.mode,
            gamma=config.gamma if config.mode == "exp_range" else 1.0,
            cycle_momentum=False,
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    if name == "one_cycle":
        max_lr = learning_rate * config.max_lr_factor
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    if name == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/miou",
        }

    raise ValueError(f"Unknown scheduler: {name}")
