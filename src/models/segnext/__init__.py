"""SegNeXt model components.

This module provides a pure PyTorch implementation of SegNeXt,
a semantic segmentation architecture using:
- MSCAN (Multi-Scale Convolutional Attention Network) encoder
- LightHamHead (Hamburger-style) decoder based on NMF matrix decomposition

Reference: https://arxiv.org/abs/2209.08575
"""

from src.models.segnext.segnext import SegNeXt

__all__ = ["SegNeXt"]


