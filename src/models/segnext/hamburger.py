"""Hamburger decoder head for SegNeXt.

LightHamHead uses NMF matrix decomposition for global context modeling.
Reference: https://arxiv.org/abs/2109.04553 (HamNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        num_groups: int = 32,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=not norm)
        ]
        if norm:
            layers.append(nn.GroupNorm(num_groups, out_channels))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization for 2D features.

    Performs NMF-based decomposition for global context modeling.
    Iteratively refines bases and coefficients to reconstruct input.
    """

    def __init__(
        self,
        spatial: bool = True,
        s: int = 1,
        d: int = 512,
        r: int = 64,
        train_steps: int = 6,
        eval_steps: int = 7,
        inv_t: int = 100,
        eta: float = 0.9,
        rand_init: bool = True,
    ) -> None:
        super().__init__()
        self.spatial = spatial
        self.S = s
        self.D = d
        self.R = r
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = 1  # NMF uses inv_t=1
        self.eta = eta
        self.rand_init = rand_init

    def _build_bases(self, b: int, s: int, d: int, r: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bases = torch.rand((b * s, d, r), device=device, dtype=torch.float32)
        return F.normalize(bases, dim=1).to(dtype)

    def local_step(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One NMF update: refine coefficients then bases."""
        # Update coefficients: (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        # Update bases: (B*S, D, N) @ (B*S, N, R) -> (B*S, D, R)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def local_inference(self, x: torch.Tensor, bases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NMF iterations to convergence."""
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """Final coefficient refinement."""
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """NMF decomposition and reconstruction: (B, C, H, W) -> (B, C, H, W)."""
        b, c, h, w = x.shape
        input_dtype = x.dtype

        # Reshape to (B*S, D, N)
        if self.spatial:
            d, n = c // self.S, h * w
            x = x.view(b * self.S, d, n)
        else:
            d, n = h * w, c // self.S
            x = x.view(b * self.S, n, d).transpose(1, 2)

        bases = self._build_bases(b, self.S, d, self.R, x.device, input_dtype)
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)

        # Reconstruct: (B*S, D, R) @ (B*S, N, R)^T -> (B*S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # Reshape back
        if self.spatial:
            x = x.view(b, c, h, w)
        else:
            x = x.transpose(1, 2).view(b, c, h, w)

        return x


class Hamburger(nn.Module):
    """Global context module using NMF decomposition."""

    def __init__(
        self,
        ham_channels: int = 512,
        md_r: int = 64,
        train_steps: int = 6,
        eval_steps: int = 7,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1, bias=True)
        self.ham = NMF2D(spatial=True, s=1, d=ham_channels, r=md_r, train_steps=train_steps, eval_steps=eval_steps)
        self.ham_out = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, 1, bias=False),
            nn.GroupNorm(num_groups, ham_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enjoy = F.relu(self.ham_in(x), inplace=True)
        enjoy = self.ham_out(self.ham(enjoy))
        return F.relu(x + enjoy, inplace=True)


class LightHamHead(nn.Module):
    """Decoder that fuses multi-scale features with Hamburger global context."""

    def __init__(
        self,
        in_channels: list[int],
        in_index: list[int],
        ham_channels: int = 512,
        channels: int = 512,
        num_classes: int = 2,
        dropout_ratio: float = 0.1,
        md_r: int = 64,
        train_steps: int = 6,
        eval_steps: int = 7,
        num_groups: int = 32,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.ham_channels = ham_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.squeeze = ConvBNReLU(sum(in_channels), ham_channels, kernel_size=1)
        self.hamburger = Hamburger(ham_channels, md_r, train_steps, eval_steps, num_groups)
        self.align = ConvBNReLU(ham_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _transform_inputs(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [inputs[i] for i in self.in_index]

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features and produce segmentation logits."""
        inputs = self._transform_inputs(inputs)

        # Resize all to first feature's spatial size
        target_size = inputs[0].shape[2:]
        resized = [
            F.interpolate(feat, size=target_size, mode="bilinear", align_corners=self.align_corners)
            if feat.shape[2:] != target_size else feat
            for feat in inputs
        ]

        x = self.squeeze(torch.cat(resized, dim=1))
        x = self.hamburger(x)
        x = self.align(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)


