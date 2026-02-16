"""Feature Pyramid Network neck for WeakCLIP (replaces mmseg FPN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network neck."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        num_outs: int = 4,
    ) -> None:
        super().__init__()
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs
