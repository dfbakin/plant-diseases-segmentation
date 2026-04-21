#!/usr/bin/env python
"""Create a "scratch" SPDNet checkpoint for the Phase 3 from-scratch run.

Phase 3 of the localization-capacity probe asks for a from-scratch
SPDNet (random weights, except for the ImageNet-pretrained ResNet50
backbone). Our train_spdnet_probe entrypoint always loads weights from
disk via ``load_spdnet_from_checkpoint``; this helper materialises that
"fresh" model as a checkpoint file so the same entrypoint can be reused
without any branching.

Run:
    python scripts/save_scratch_spdnet.py \\
        --output outputs/spdnet_plantseg/seg_probe_phase3/scratch_init.pt \\
        --fusion-mode spatial \\
        --num-classes 115
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.spdnet.model import SPDNet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fusion-mode", choices=["token", "spatial"], default="spatial")
    ap.add_argument("--num-classes", type=int, default=115)
    ap.add_argument("--fpn-channels", type=int, default=256)
    ap.add_argument("--mse-reduction", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = SPDNet(
        num_classes=args.num_classes,
        fpn_channels=args.fpn_channels,
        mse_reduction=args.mse_reduction,
        pretrained=True,  # ImageNet ResNet50; rest is random-init
        fusion_mode=args.fusion_mode,
    )

    blob = {
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "hyper_parameters": {
            "fusion_mode": args.fusion_mode,
            "num_classes": args.num_classes,
            "fpn_channels": args.fpn_channels,
            "mse_reduction": args.mse_reduction,
            "scratch_init": True,
            "seed": args.seed,
        },
    }
    torch.save(blob, args.output)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved scratch SPDNet to {args.output}")
    print(f"  fusion_mode = {args.fusion_mode}")
    print(f"  num_classes = {args.num_classes}")
    print(f"  fpn_channels = {args.fpn_channels}")
    print(f"  total params = {n_params:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
