"""WeakCLIP: CLIP-guided weakly-supervised semantic segmentation.

Extracted from https://github.com/hustvl/WeakCLIP (mmseg-based)
into pure PyTorch modules for integration with Lightning/Hydra.
"""

from src.wsss.weakclip.model import WeakCLIP
