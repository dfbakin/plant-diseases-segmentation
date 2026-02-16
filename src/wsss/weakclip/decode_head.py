"""FPN decode head with seeding loss for WeakCLIP (from FPNHeadDGCN dgcn_lite)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wsss.weakclip.losses import cues_from_pseudo_mask, seeding_loss, stable_softmax


class FPNDecodeHead(nn.Module):
    """Multi-scale FPN decode head with seeding loss."""

    def __init__(
        self,
        in_channels: list[int] | None = None,
        channels: int = 256,
        num_classes: int = 21,
        feature_strides: list[int] | None = None,
        feature_size: tuple[int, int] = (64, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [256, 256, 256, 256]
        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]

        self.num_classes = num_classes
        self.feature_size = feature_size
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i, (in_ch, stride) in enumerate(zip(in_channels, feature_strides)):
            head_layers = []
            ratio = stride // feature_strides[0]
            num_ups = max(0, int(torch.tensor(ratio).log2().item()))
            curr_ch = in_ch
            for _ in range(num_ups):
                head_layers.extend(
                    [
                        nn.Conv2d(curr_ch, channels, 3, padding=1),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ]
                )
                curr_ch = channels
            if not head_layers:
                head_layers.append(nn.Conv2d(curr_ch, channels, 3, padding=1))
            self.scale_heads.append(nn.Sequential(*head_layers))

        self.dropout = nn.Dropout2d(dropout)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        output = self.scale_heads[0](inputs[0])
        for i in range(1, len(self.feature_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](inputs[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        output = self.dropout(output)
        return self.cls_seg(output)

    def compute_loss(
        self,
        logits: torch.Tensor,
        gt_semantic_seg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seg_logits = F.interpolate(
            logits,
            size=self.feature_size,
            mode="bilinear",
            align_corners=False,
        )
        cues = cues_from_pseudo_mask(
            gt_semantic_seg,
            self.num_classes,
            self.feature_size,
        )
        probs = stable_softmax(seg_logits)
        loss = seeding_loss(probs, cues)
        return {"loss_seeding": loss}
