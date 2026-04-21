#!/usr/bin/env python
"""Module-level smoke test for the SPDNet seg-probe wrapper.

Loads the actual token and spatial SPDNet checkpoints from disk, runs
five forward passes per model, and asserts that:

  * every probe position returns a tensor of the correct shape;
  * frozen-backbone mode keeps grads on the head only;
  * loss is finite for random masks.

This is an end-to-end smoke that exercises real disk weights and is
meant to be the very last gate before launching the overnight pipeline.

Run:
    .venv/bin/python scripts/smoke_test_seg_probe.py

Exits 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint
from src.wsss.spdnet.seg_probe import (
    NEEDS_REFERENCE,
    PROBE_POSITIONS,
    SPATIAL_ONLY_POSITIONS,
    SPDNetWithProbes,
    bce_dice_loss,
    channels_for_position,
)


CKPT_TOKEN = "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"
CKPT_SPATIAL = (
    "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/"
    "epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt"
)
N_PASSES = 5
INPUT_SIZE = 448
TARGET_SIZE = (448, 448)


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _assert_shapes(feats: dict, expected_positions: tuple[str, ...], model) -> None:
    """Sanity-check the feature dict from extract_probe_features."""
    for pos in expected_positions:
        assert pos in feats, f"missing position {pos!r}; have {list(feats.keys())}"
        t = feats[pos]
        assert t.dim() == 4, f"{pos}: expected 4D, got {t.shape}"
        ch_expected = channels_for_position(model, pos)
        assert t.shape[1] == ch_expected, (
            f"{pos}: expected {ch_expected} channels, got {t.shape[1]}"
        )


def _check_grad_isolation(wrapper: SPDNetWithProbes, q, r) -> None:
    """One backward pass must leave SPDNet grads at zero in frozen mode."""
    for p in wrapper.parameters():
        if p.grad is not None:
            p.grad.zero_()
    seg_logits = wrapper(q, r, return_cls=False)
    target = torch.randint(0, 2, seg_logits.shape, dtype=torch.float32, device=seg_logits.device)
    loss = bce_dice_loss(seg_logits, target)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    loss.backward()
    backbone_nonzero = sum(
        1 for n, p in wrapper.spdnet.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert backbone_nonzero == 0, (
        f"frozen mode leaked {backbone_nonzero} backbone grads"
    )
    head_nonzero = sum(
        1 for p in wrapper.head.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert head_nonzero > 0, "head got no gradient"


def _smoke_one_ckpt(ckpt_path: str, fusion_mode_expected: str, device: torch.device) -> None:
    _section(f"Smoke: {fusion_mode_expected.upper()} ckpt -- {ckpt_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    spdnet = load_spdnet_from_checkpoint(ckpt_path, num_classes=115).to(device).eval()
    assert spdnet.fusion_mode == fusion_mode_expected, (
        f"expected fusion_mode={fusion_mode_expected!r}, got {spdnet.fusion_mode!r}"
    )

    if fusion_mode_expected == "spatial":
        positions = PROBE_POSITIONS  # all 6
    else:
        positions = tuple(p for p in PROBE_POSITIONS if p not in SPATIAL_ONLY_POSITIONS)

    print(f"  positions: {positions}")
    print(f"  fusion_mode: {spdnet.fusion_mode}")
    print(f"  device: {device}")

    q_batch = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    r_batch = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    feats = spdnet.extract_probe_features(q_batch, r_batch)
    _assert_shapes(feats, positions, spdnet)
    print(f"  feature dict has all {len(positions)} expected positions [OK]")

    feats_no_ref = spdnet.extract_probe_features(q_batch, reference=None)
    no_ref_keys = {"P1_layer4", "P2_fpn_p2", "P3_query_merged"}
    for k in no_ref_keys:
        assert k in feats_no_ref, f"missing {k} in no-ref feats"
    for k in NEEDS_REFERENCE:
        assert k not in feats_no_ref, f"unexpected {k} in no-ref feats"
    print(f"  no-reference path correctly omits P4/P5/P6 [OK]")

    for pos in positions:
        wrapper = SPDNetWithProbes(
            spdnet=load_spdnet_from_checkpoint(ckpt_path, num_classes=115).to(device).eval(),
            position=pos,
            target_size=TARGET_SIZE,
            freeze_backbone=True,
        ).to(device)
        wrapper.train()  # head in train mode for BN-less probe; backbone stays frozen

        ref_for_fwd = r_batch if wrapper.needs_reference else None

        with torch.no_grad():
            for i in range(N_PASSES):
                seg_logits, cls_logits = wrapper(q_batch, ref_for_fwd, return_cls=True)
                assert seg_logits.shape == (2, 1, *TARGET_SIZE), (
                    f"{pos}: bad seg_logits shape {seg_logits.shape}"
                )
                assert cls_logits.shape == (2, 115), (
                    f"{pos}: bad cls_logits shape {cls_logits.shape}"
                )
                assert torch.isfinite(seg_logits).all(), f"{pos}: non-finite seg logits"
                assert torch.isfinite(cls_logits).all(), f"{pos}: non-finite cls logits"
        print(f"  position={pos}: {N_PASSES} forward passes OK, "
              f"seg_logits={tuple(seg_logits.shape)}, cls_logits={tuple(cls_logits.shape)}")

        _check_grad_isolation(wrapper, q_batch, ref_for_fwd)
        print(f"  position={pos}: frozen-backbone grad isolation OK")

        del wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    _section("SPDNet seg-probe module smoke test")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  Torch:         {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU mem free:  {free // 1024**2} / {total // 1024**2} MiB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    failures: list[str] = []

    for ckpt, fusion in [(CKPT_TOKEN, "token"), (CKPT_SPATIAL, "spatial")]:
        try:
            _smoke_one_ckpt(ckpt, fusion, device)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\nFAILED on {fusion} ckpt {ckpt!r}: {e}\n{tb}")
            failures.append(f"{fusion}: {e}")

    _section("Summary")
    print(f"  total time: {time.time() - t0:.1f}s")
    if failures:
        print(f"  FAILED ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
