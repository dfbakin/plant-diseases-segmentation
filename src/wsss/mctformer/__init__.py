"""MCTformer: Multi-Class Token Transformer for WSSS.

Ported from https://github.com/xulianuwa/MCTformer (Python 3.6 / timm 0.4.12)
to modern Python 3.12 / timm >= 1.0.
"""

from src.wsss.mctformer.model import MCTformerPlus, create_mctformer_v2

__all__ = ["MCTformerPlus", "create_mctformer_v2"]
