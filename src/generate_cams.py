"""Generate CAMs from a trained MCTformer checkpoint.

Dataset-agnostic: reads image_dir + label_file (.npy dict).
Use src/export_labels.py to create label files from any dataset.

Example:
    python src/generate_cams.py checkpoint=... label_file=outputs/labels/voc.npy
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

from src.wsss.mctformer.evaluation import evaluate_cam_threshold_sweep
from src.wsss.mctformer.model import create_mctformer_v2

log = logging.getLogger(__name__)


@dataclass
class GenCAMConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    checkpoint: str = ""
    image_dir: str = "data/VOC2012/JPEGImages"
    image_ext: str = ".jpg"
    label_file: str = ""
    output_dir: str = "outputs/cams/cam_npy"

    num_classes: int = 20
    input_size: int = 448
    max_size: int = 0
    patch_size: int = 16
    scales: list[float] = field(default_factory=lambda: [1.0, 0.75, 1.25])
    n_layers: int = 3
    attention_type: str = "fused"
    patch_attn_refine: bool = True

    binary_aggregate: str = ""

    gt_dir: str = ""
    eval_threshold_sweep: bool = False
    eval_sweep_samples: int = 0


cs = ConfigStore.instance()
cs.store(name="gen_cam_config", node=GenCAMConfig)


def load_model(checkpoint: str, num_classes: int, input_size: int) -> torch.nn.Module:
    model = create_mctformer_v2(num_classes=num_classes, pretrained=False, input_size=input_size)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    return model


def build_val_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


@torch.no_grad()
def generate_cam_single(
    model: torch.nn.Module,
    image_list: list[torch.Tensor],
    target: torch.Tensor,
    cfg: GenCAMConfig,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Generate fused multi-scale CAM for one image.

    When ``cfg.binary_aggregate`` is set (``"max"`` or ``"mean"``), all class
    CAMs are computed without GT masking and aggregated into a single binary
    CAM stored as ``{0: cam}``.
    """
    aggregate = cfg.binary_aggregate
    w_orig = image_list[0].shape[2]
    h_orig = image_list[0].shape[3]

    cam_list = []
    for s, images in enumerate(image_list):
        images = images.to(device)
        w = images.shape[2] - images.shape[2] % cfg.patch_size
        h = images.shape[3] - images.shape[3] % cfg.patch_size
        w_feat = w // cfg.patch_size
        h_feat = h // cfg.patch_size

        _out, cls_att, patch_att = model(
            images, return_att=True, n_layers=cfg.n_layers, attention_type=cfg.attention_type
        )
        patch_att = torch.sum(patch_att, dim=0)

        if cfg.patch_attn_refine:
            cls_att = torch.matmul(
                patch_att.unsqueeze(1),
                cls_att.view(cls_att.shape[0], cls_att.shape[1], -1, 1),
            ).reshape(cls_att.shape[0], cls_att.shape[1], w_feat, h_feat)

        cls_att = F.interpolate(
            cls_att, size=(w_orig, h_orig), mode="bilinear", align_corners=False
        )
        cls_att = cls_att[0].cpu().numpy()

        if not aggregate:
            cls_att = cls_att * target.view(cfg.num_classes, 1, 1).cpu().numpy()

        if s % 2 == 1:
            cls_att = np.flip(cls_att, axis=-1)
        cam_list.append(cls_att)

    sum_cam = np.sum(cam_list, axis=0)

    if aggregate:
        if aggregate == "max":
            merged = np.max(sum_cam, axis=0)
        elif aggregate == "mean":
            merged = np.mean(sum_cam, axis=0)
        else:
            raise ValueError(f"Unknown binary_aggregate: {aggregate!r}. Use 'max' or 'mean'.")
        merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-8)
        return {0: merged.astype(np.float32)}

    cam_dict: dict[int, np.ndarray] = {}
    for c in range(cfg.num_classes):
        if target[c] > 0:
            cls_cam = sum_cam[c]
            cls_cam = (cls_cam - cls_cam.min()) / (cls_cam.max() - cls_cam.min() + 1e-8)
            cam_dict[c] = cls_cam.copy()
    return cam_dict


def generate_cams(cfg: GenCAMConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading model from {cfg.checkpoint}")
    model = load_model(cfg.checkpoint, cfg.num_classes, cfg.input_size).to(device)
    model.eval()

    if cfg.binary_aggregate and cfg.binary_aggregate not in ("max", "mean"):
        raise ValueError(f"binary_aggregate must be 'max', 'mean', or '' (disabled), got {cfg.binary_aggregate!r}")

    labels = np.load(cfg.label_file, allow_pickle=True).item()
    names = list(labels.keys())
    agg_msg = f", binary_aggregate={cfg.binary_aggregate}" if cfg.binary_aggregate else ""
    log.info(f"Generating CAMs for {len(names)} images, scales={cfg.scales}{agg_msg}")

    image_dir = Path(cfg.image_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tfm = build_val_transform()

    for name in tqdm(names, desc="Generating CAMs"):
        img_path = image_dir / f"{name}{cfg.image_ext}"
        if not img_path.exists():
            continue

        img_pil = PIL.Image.open(img_path).convert("RGB")
        label = torch.from_numpy(labels[name]).float()

        max_long = cfg.max_size if cfg.max_size > 0 else int(cfg.input_size * 1.75)
        long_side = max(img_pil.size)
        if long_side > max_long:
            ratio = max_long / long_side
            img_pil = img_pil.resize(
                (round(img_pil.width * ratio), round(img_pil.height * ratio)),
                resample=PIL.Image.BICUBIC,
            )

        image_list = []
        for s in cfg.scales:
            tw = round(img_pil.size[0] * s)
            th = round(img_pil.size[1] * s)
            scaled_long = max(tw, th)
            if scaled_long > max_long:
                r = max_long / scaled_long
                tw, th = round(tw * r), round(th * r)
            s_img = img_pil.resize((tw, th), resample=PIL.Image.BICUBIC)
            t_img = tfm(s_img).unsqueeze(0)
            image_list.append(t_img)
            image_list.append(torch.flip(t_img, [-1]))

        cam_dict = generate_cam_single(model, image_list, label, cfg, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if cam_dict:
            np.save(str(output_dir / f"{name}.npy"), cam_dict)

    log.info(f"Saved CAMs to {output_dir}")

    if cfg.eval_threshold_sweep and cfg.gt_dir:
        eval_num_cls = 2 if cfg.binary_aggregate else cfg.num_classes + 1
        log.info(
            f"Running threshold sweep for mIoU evaluation (num_cls={eval_num_cls})..."
        )
        result = evaluate_cam_threshold_sweep(
            predict_dir=str(output_dir),
            gt_dir=cfg.gt_dir,
            name_list=names,
            num_cls=eval_num_cls,
            max_samples=cfg.eval_sweep_samples,
        )
        log.info(
            f"Best mIoU: {result['best_miou']:.2f}% at threshold={result['best_threshold']:.2f}"
        )


@hydra.main(version_base=None, config_name="gen_cam_config")
def main(cfg: DictConfig) -> None:
    generate_cams(cfg)


if __name__ == "__main__":
    main()
