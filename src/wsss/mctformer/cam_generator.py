"""Attention-based CAM generation for MCTformer (from engine.py)."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.wsss.mctformer.model import MCTformerPlus

log = logging.getLogger(__name__)


class MCTformerCAMGenerator:
    """Generate class activation maps from a trained MCTformer model.

    Args:
        model: Trained MCTformerPlus model
        device: Torch device
        n_layers: Number of last attention layers to average
        attention_type: 'fused', 'patchcam', or 'mct'
        patch_size: Patch size used by the model
        patch_attn_refine: Whether to refine CAMs with patch attention
    """

    def __init__(
        self,
        model: MCTformerPlus,
        device: torch.device,
        n_layers: int = 3,
        attention_type: str = "fused",
        patch_size: int = 16,
        patch_attn_refine: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.patch_size = patch_size
        self.patch_attn_refine = patch_attn_refine
        self.model.eval()

    @torch.no_grad()
    def generate_cam_single(
        self,
        image_list: list[torch.Tensor],
        target: torch.Tensor,
        num_classes: int,
    ) -> dict[int, np.ndarray]:
        """Generate CAM for a single image (possibly multi-scale).

        Args:
            image_list: List of scaled image tensors [(1, 3, H_s, W_s), ...].
                Even indices are original, odd indices are horizontally flipped.
            target: Multi-hot label vector (num_classes,)
            num_classes: Number of classes

        Returns:
            Dictionary mapping class_idx -> (H, W) normalized CAM heatmap
        """
        images1 = image_list[0].to(self.device)
        w_orig = images1.shape[2]
        h_orig = images1.shape[3]

        cam_list = []
        for s in range(len(image_list)):
            images = image_list[s].to(self.device)
            w = images.shape[2] - images.shape[2] % self.patch_size
            h = images.shape[3] - images.shape[3] % self.patch_size
            w_featmap = w // self.patch_size
            h_featmap = h // self.patch_size

            _output, cls_attentions, patch_attn = self.model(
                images,
                return_att=True,
                n_layers=self.n_layers,
                attention_type=self.attention_type,
            )
            patch_attn = torch.sum(patch_attn, dim=0)

            if self.patch_attn_refine:
                cls_attentions = torch.matmul(
                    patch_attn.unsqueeze(1),
                    cls_attentions.view(cls_attentions.shape[0], cls_attentions.shape[1], -1, 1),
                ).reshape(
                    cls_attentions.shape[0],
                    cls_attentions.shape[1],
                    w_featmap,
                    h_featmap,
                )

            cls_attentions = F.interpolate(
                cls_attentions,
                size=(w_orig, h_orig),
                mode="bilinear",
                align_corners=False,
            )[0]
            cls_attentions = (
                cls_attentions.cpu().numpy() * target.view(num_classes, 1, 1).cpu().numpy()
            )

            if s % 2 == 1:
                cls_attentions = np.flip(cls_attentions, axis=-1)
            cam_list.append(cls_attentions)

        sum_cam = np.sum(cam_list, axis=0)

        cam_dict: dict[int, np.ndarray] = {}
        for cls_ind in range(num_classes):
            if target[cls_ind] > 0:
                cls_cam = sum_cam[cls_ind]
                cls_cam = (cls_cam - cls_cam.min()) / (cls_cam.max() - cls_cam.min() + 1e-8)
                cam_dict[cls_ind] = cls_cam

        return cam_dict

    def generate_cams_dataset(
        self,
        data_loader: torch.utils.data.DataLoader,
        img_names: list[str],
        num_classes: int,
        output_dir: str | Path,
        visualize: bool = False,
        vis_dir: str | Path | None = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Generate CAMs for an entire dataset and save as .npy files.

        Args:
            data_loader: DataLoader yielding (image_list, target) tuples.
                image_list is a list of scaled tensors (for multi-scale).
            img_names: List of image names (same order as data_loader).
            num_classes: Number of classes.
            output_dir: Directory to save .npy CAM files.
            visualize: If True, save CAM visualizations.
            vis_dir: Directory for visualizations.
            mean: ImageNet mean used for denormalization.
            std: ImageNet std used for denormalization.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if visualize and vis_dir is not None:
            vis_dir = Path(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)

        index = 0
        for image_list, target in tqdm(data_loader, desc="Generating CAMs"):
            img_name = img_names[index]
            index += 1

            cam_dict = self.generate_cam_single(image_list, target[0], num_classes)

            if cam_dict:
                np.save(str(output_dir / f"{img_name}.npy"), cam_dict)

                if visualize and vis_dir is not None:
                    img_tensor = image_list[0][0]
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = np.zeros_like(img_np)
                    for c in range(3):
                        img_np[:, :, c] = (img_tensor[c].cpu().numpy() * std[c] + mean[c]) * 255.0
                    img_np = img_np.clip(0, 255).astype(np.uint8)

                    for cls_ind, cam in cam_dict.items():
                        _save_cam_visualization(img_np, cam, vis_dir / f"{img_name}_{cls_ind}.png")

        log.info(f"Generated CAMs for {index} images -> {output_dir}")


def _save_cam_visualization(img: np.ndarray, cam: np.ndarray, save_path: Path) -> None:
    """Overlay CAM heatmap on image and save."""
    img_float = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    combined = heatmap + img_float
    combined = combined / np.max(combined)
    combined = np.uint8(255 * combined)
    cv2.imwrite(str(save_path), combined)
