"""Train PSA affinity network on CRF-refined pseudo labels.

Inputs: la_crf/ and ha_crf/ directories from apply_crf.py.
Output: trained affinity network checkpoint.

Example:
    python src/train_psa.py la_crf_dir=outputs/cams/la_crf ha_crf_dir=outputs/cams/ha_crf
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.wsss.refinement.aff_dataset import VOCAffDataset
from src.wsss.refinement.affinity_net import AffinityNet
from src.wsss.refinement.resnet38d import Normalize

log = logging.getLogger(__name__)


class PolyOptimizer(torch.optim.SGD):
    """SGD with polynomial learning rate decay."""

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.global_step = 0
        self.max_step = max_step
        self.initial_lrs = [pg["lr"] for pg in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** 0.9
            for i, pg in enumerate(self.param_groups):
                pg["lr"] = self.initial_lrs[i] * lr_mult
        super().step(closure)
        self.global_step += 1


@dataclass
class PSATrainConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    voc_root: str = "data/VOC2012"
    la_crf_dir: str = "outputs/cams/la_crf"
    ha_crf_dir: str = "outputs/cams/ha_crf"
    split: str = "train_aug_id"

    backbone_weights: str = "pretrained/res38_cls.pth"
    output_path: str = "outputs/psa/psa_aff.pth"

    batch_size: int = 8
    max_epochs: int = 5
    lr: float = 0.01
    weight_decay: float = 5e-4
    num_workers: int = 8
    cropsize: int = 448


cs = ConfigStore.instance()
cs.store(name="psa_train_config", node=PSATrainConfig)


def train_psa(cfg: PSATrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AffinityNet(predefined_featuresize=cfg.cropsize // 8)
    sd = torch.load(cfg.backbone_weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.train()
    log.info(f"Loaded backbone from {cfg.backbone_weights}")

    normalize_fn = Normalize()
    dataset = VOCAffDataset(
        voc_root=cfg.voc_root,
        la_crf_dir=cfg.la_crf_dir,
        ha_crf_dir=cfg.ha_crf_dir,
        split=cfg.split,
        cropsize=cfg.cropsize,
        normalize_fn=normalize_fn,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    max_step = len(dataset) // cfg.batch_size * cfg.max_epochs
    log.info(f"Dataset: {len(dataset)} images, max_step={max_step}")

    param_groups = model.get_parameter_groups()
    optimizer = PolyOptimizer(
        [
            {"params": param_groups[0], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": param_groups[1], "lr": 2 * cfg.lr, "weight_decay": 0},
            {"params": param_groups[2], "lr": 10 * cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": param_groups[3], "lr": 20 * cfg.lr, "weight_decay": 0},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_step=max_step,
    )

    for epoch in range(cfg.max_epochs):
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"PSA Epoch {epoch + 1}/{cfg.max_epochs}")
        for img, (bg_label, fg_label, neg_label) in pbar:
            img = img.to(device)
            bg_label = bg_label.to(device)
            fg_label = fg_label.to(device)
            neg_label = neg_label.to(device)

            aff = model(img).view(img.size(0), -1)
            bg_label = bg_label.view(img.size(0), -1)
            fg_label = fg_label.view(img.size(0), -1)
            neg_label = neg_label.view(img.size(0), -1)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(-bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(-fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(-neg_label * torch.log(1.0 + 1e-5 - aff)) / neg_count

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = running_loss / len(loader)
        log.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))
    log.info(f"Saved affinity net to {output_path}")


@hydra.main(version_base=None, config_name="psa_train_config")
def main(cfg: DictConfig) -> None:
    train_psa(cfg)


if __name__ == "__main__":
    main()
