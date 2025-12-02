"""Hamburger decoder head for SegNeXt.

Pure PyTorch implementation of LightHamHead using NMF matrix decomposition.
Reference: https://arxiv.org/abs/2109.04553 (HamNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGNReLU(nn.Module):
    """Convolution + GroupNorm + ReLU block."""

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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not norm,
            )
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

    This module performs NMF-based matrix decomposition for
    global context modeling in the Hamburger module.

    Attributes:
        S: Number of spatial splits.
        D: Feature dimension.
        R: Rank of decomposition (number of bases).
        train_steps: Number of NMF iterations during training.
        eval_steps: Number of NMF iterations during evaluation.
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
        """Initialize NMF2D module.

        Args:
            spatial: Whether to decompose spatially.
            s: Number of splits.
            d: Feature dimension.
            r: Rank of decomposition.
            train_steps: NMF iterations during training.
            eval_steps: NMF iterations during evaluation.
            inv_t: Inverse temperature for softmax.
            eta: Momentum for bases update.
            rand_init: Whether to use random initialization.
        """
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

    def _build_bases(
        self, b: int, s: int, d: int, r: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        bases = torch.rand((b * s, d, r), device=device, dtype=torch.float32)
        bases = F.normalize(bases, dim=1)
        return bases.to(dtype)

    def local_step(
        self,
        x: torch.Tensor,
        bases: torch.Tensor,
        coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one NMF update step.

        Args:
            x: Input features (B*S, D, N).
            bases: Current bases (B*S, D, R).
            coef: Current coefficients (B*S, N, R).

        Returns:
            Updated bases and coefficients.
        """
        # Update coefficients
        # (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B*S, N, R) @ [(B*S, D, R)^T @ (B*S, D, R)] -> (B*S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        # Update bases
        # (B*S, D, N) @ (B*S, N, R) -> (B*S, D, R)
        numerator = torch.bmm(x, coef)
        # (B*S, D, R) @ [(B*S, N, R)^T @ (B*S, N, R)] -> (B*S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def local_inference(
        self, x: torch.Tensor, bases: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NMF iterations.

        Args:
            x: Input features (B*S, D, N).
            bases: Initial bases (B*S, D, R).

        Returns:
            Final bases and coefficients.
        """
        # Initialize coefficients
        # (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(
        self,
        x: torch.Tensor,
        bases: torch.Tensor,
        coef: torch.Tensor,
    ) -> torch.Tensor:
        """Compute final coefficients.

        Args:
            x: Input features (B*S, D, N).
            bases: Final bases (B*S, D, R).
            coef: Current coefficients (B*S, N, R).

        Returns:
            Updated coefficients.
        """
        # (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B*S, N, R) @ (B*S, D, R)^T @ (B*S, D, R) -> (B*S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        input_dtype = x.dtype

        # Reshape: (B, C, H, W) -> (B*S, D, N)
        if self.spatial:
            d = c // self.S
            n = h * w
            x = x.view(b * self.S, d, n)
        else:
            d = h * w
            n = c // self.S
            x = x.view(b * self.S, n, d).transpose(1, 2)

        # Build bases with matching dtype
        bases = self._build_bases(b, self.S, d, self.R, x.device, input_dtype)

        # Run NMF
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)

        # Reconstruct: (B*S, D, R) @ (B*S, N, R)^T -> (B*S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # Reshape back: (B*S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(b, c, h, w)
        else:
            x = x.transpose(1, 2).view(b, c, h, w)

        return x


class Hamburger(nn.Module):
    """Hamburger module for global context modeling.

    Uses NMF-based matrix decomposition for efficient global reasoning.
    """

    def __init__(
        self,
        ham_channels: int = 512,
        md_r: int = 64,
        train_steps: int = 6,
        eval_steps: int = 7,
        num_groups: int = 32,
    ) -> None:
        """Initialize Hamburger module.

        Args:
            ham_channels: Number of input/output channels.
            md_r: Rank of NMF decomposition.
            train_steps: NMF iterations during training.
            eval_steps: NMF iterations during evaluation.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1, bias=True)

        self.ham = NMF2D(
            spatial=True,
            s=1,
            d=ham_channels,
            r=md_r,
            train_steps=train_steps,
            eval_steps=eval_steps,
        )

        self.ham_out = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, 1, bias=False),
            nn.GroupNorm(num_groups, ham_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham


class LightHamHead(nn.Module):
    """Lightweight Hamburger Head for semantic segmentation.

    A decoder that fuses multi-scale features and applies Hamburger
    module for global context modeling.

    Attributes:
        in_channels: List of input channel dimensions from encoder.
        in_index: Indices of encoder stages to use.
        ham_channels: Channel dimension for Hamburger module.
        channels: Output channels before classification.
        num_classes: Number of segmentation classes.
    """

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
        """Initialize LightHamHead.

        Args:
            in_channels: Channel dimensions for each input feature map.
            in_index: Indices of encoder features to use.
            ham_channels: Channels for Hamburger module.
            channels: Output channels before classification.
            num_classes: Number of output classes.
            dropout_ratio: Dropout ratio before classification.
            md_r: Rank of NMF decomposition.
            train_steps: NMF iterations during training.
            eval_steps: NMF iterations during evaluation.
            num_groups: Number of groups for GroupNorm.
            align_corners: align_corners for F.interpolate.
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.ham_channels = ham_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        # Squeeze: concat all features and reduce channels
        self.squeeze = ConvBNReLU(
            sum(in_channels),
            ham_channels,
            kernel_size=1,
        )

        # Hamburger module for global context
        self.hamburger = Hamburger(
            ham_channels=ham_channels,
            md_r=md_r,
            train_steps=train_steps,
            eval_steps=eval_steps,
            num_groups=num_groups,
        )

        # Align: reduce to output channels
        self.align = ConvBNReLU(
            ham_channels,
            channels,
            kernel_size=1,
        )

        # Classification head
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _transform_inputs(
        self, inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Select and transform encoder features.

        Args:
            inputs: List of all encoder feature maps.

        Returns:
            Selected feature maps based on in_index.
        """
        return [inputs[i] for i in self.in_index]

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: List of encoder feature maps.

        Returns:
            Segmentation logits of shape (B, num_classes, H', W').
        """
        # Select features
        inputs = self._transform_inputs(inputs)

        # Resize all to first feature's size and concatenate
        target_size = inputs[0].shape[2:]
        resized = [
            F.interpolate(
                feat,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            if feat.shape[2:] != target_size
            else feat
            for feat in inputs
        ]

        x = torch.cat(resized, dim=1)

        # Squeeze
        x = self.squeeze(x)

        # Hamburger for global context
        x = self.hamburger(x)

        # Align
        x = self.align(x)

        # Classification
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.conv_seg(x)

        return output


