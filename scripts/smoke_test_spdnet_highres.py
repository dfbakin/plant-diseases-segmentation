"""896x896 SPDNet smoke test: one forward+backward + VRAM headroom check.

Purpose
-------
Before committing ~25h of compute to the Phase 5 highres training
stage, verify that:

  * SPDNet (spatial fusion) actually forwards + backwards at 896^2
    with the D4 auxiliary losses turned on,
  * peak allocated VRAM at ``batch_size=2`` stays below a configurable
    threshold (default 22 GiB on 24 GiB RTX 5090 / 4090, so
    gradient-accum at batch=4 accum=8 is safe),
  * all intermediate shapes stay consistent (no hidden ``interpolate``
    bug that only fires at 896^2).

CPU-only or missing CUDA: runs at a reduced resolution (128x128) just
to verify forward+backward shapes. VRAM check is skipped; the CI
smoke test relies on that behaviour.

Exit codes:
  0 -- forward+backward succeeded and VRAM (if measured) is under cap
  1 -- OOM, shape mismatch, or other runtime failure
  2 -- succeeded but peak VRAM exceeded the cap

Usage
-----
::

    uv run python scripts/smoke_test_spdnet_highres.py \\
        --image-size 896 --batch-size 2 --vram-cap-gib 22.0

Writes ``outputs/phase5/smoke_highres/smoke.json`` when ``--out``
is provided (Phase 5 orchestration artefact).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.wsss.spdnet.model import SPDNet
from src.wsss.spdnet.spatial_losses import (
    attention_concentration_loss,
    cam_pseudo_mask_loss,
)

NUM_CLASSES_DEFAULT = 115
DEFAULT_IMAGE_SIZE = 896
DEFAULT_BATCH_SIZE = 2
DEFAULT_VRAM_CAP_GIB = 22.0
CPU_FALLBACK_IMAGE_SIZE = 128


def _run_one_step(
    image_size: int,
    batch_size: int,
    num_classes: int,
    device: torch.device,
    use_aux: bool,
    lambda_ac: float,
    lambda_mask: float,
    ref_pool_size: int = 14,
) -> dict:
    """Run one fwd+bwd at (image_size, batch_size) and return diagnostics."""
    torch.manual_seed(0)

    peak_mib = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = SPDNet(
        num_classes=num_classes, pretrained=False, fusion_mode="spatial",
        ref_pool_size=ref_pool_size,
    ).to(device).train()

    q = torch.randn(batch_size, 3, image_size, image_size, device=device)
    r = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.zeros(batch_size, num_classes, device=device)
    labels[:, 0] = 1.0  # active class 0

    t0 = time.time()
    feats = model.extract_merged_features(q, r, return_attn=use_aux)
    fused = feats["fused"]
    pooled = fused.mean(dim=[2, 3])
    logits = model.classifier(pooled)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    comp = {"L_cls": float(loss.detach().item())}

    if use_aux:
        # L_ac: direct aux target of the D1/D4 recipe.
        if "attn_map" in feats and lambda_ac > 0:
            L_ac = attention_concentration_loss(feats["attn_map"])
            loss = loss + lambda_ac * L_ac
            comp["L_ac"] = float(L_ac.detach().item())
        # L_mask: D2 / D4 pseudo-mask MSE.
        if lambda_mask > 0:
            L_mask = cam_pseudo_mask_loss(
                p3_query=feats["query_merged"],
                p4_fused=fused,
                cls_weight=model.classifier.weight,
                labels=labels,
                alpha_pos=0.25,
                beta_neg=0.5,
                mask_combiner="intersection",
            )
            loss = loss + lambda_mask * L_mask
            comp["L_mask"] = float(L_mask.detach().item())

    loss.backward()
    fwd_bwd_s = time.time() - t0

    n_params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in model.parameters())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    info = {
        "image_size": image_size,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "device": str(device),
        "use_aux": use_aux,
        "lambda_ac": lambda_ac,
        "lambda_mask": lambda_mask,
        "loss_total": float(loss.detach().item()),
        "loss_components": comp,
        "query_merged_shape": list(feats["query_merged"].shape),
        "fused_shape": list(fused.shape),
        "logits_shape": list(logits.shape),
        "fwd_bwd_seconds": round(fwd_bwd_s, 3),
        "peak_vram_mib": round(peak_mib, 1),
        "peak_vram_gib": round(peak_mib / 1024.0, 3),
        "params_with_grad_nonzero": n_params_with_grad,
        "total_params": total_params,
    }
    del model, q, r, feats, fused, pooled, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES_DEFAULT)
    p.add_argument("--vram-cap-gib", type=float, default=DEFAULT_VRAM_CAP_GIB)
    p.add_argument("--no-aux", action="store_true",
                    help="Skip aux losses (L_ac, L_mask). Forward+backward only.")
    p.add_argument("--lambda-ac", type=float, default=0.1)
    p.add_argument("--lambda-mask", type=float, default=0.1)
    p.add_argument("--ref-pool-size", type=int, default=14,
                    help="SCA reference pool side length. Pass 0 to auto-scale "
                         "as max(14, image_size // 44) (matching train_spdnet).")
    p.add_argument("--out", type=Path, default=None,
                    help="Optional: write diagnostics to this JSON path.")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                    help="When CUDA is unavailable, silently reduce image-size "
                         "to 128 and skip the VRAM cap check. Useful for CI.")
    args = p.parse_args()

    have_cuda = torch.cuda.is_available()
    if not have_cuda and not args.allow_cpu_fallback:
        print(
            "ERROR: CUDA is not available. Highres smoke test needs a GPU. "
            "Pass --allow-cpu-fallback for a CI smoke at 128^2 (shape-only).",
            file=sys.stderr,
        )
        return 1

    if not have_cuda:
        effective_image_size = min(args.image_size, CPU_FALLBACK_IMAGE_SIZE)
        print(
            f"[smoke] CUDA unavailable; running reduced shape smoke at "
            f"{effective_image_size}^2 (cap check skipped)",
        )
    else:
        effective_image_size = args.image_size

    device = torch.device("cuda" if have_cuda else "cpu")

    rps = args.ref_pool_size
    if rps <= 0:
        rps = max(14, effective_image_size // 44)
        print(f"[smoke] ref_pool_size auto: max(14, {effective_image_size}//44) = {rps}")
    try:
        info = _run_one_step(
            image_size=effective_image_size,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            device=device,
            use_aux=not args.no_aux,
            lambda_ac=args.lambda_ac,
            lambda_mask=args.lambda_mask,
            ref_pool_size=rps,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[smoke] FAILED: OOM at {effective_image_size}^2 "
              f"batch={args.batch_size}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    print("[smoke] forward+backward OK")
    print(json.dumps(info, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(info, indent=2))
        print(f"[smoke] wrote {args.out}")

    # VRAM cap check (only when we're actually on CUDA).
    if have_cuda:
        peak_gib = info["peak_vram_gib"]
        if peak_gib > args.vram_cap_gib:
            print(
                f"[smoke] FAIL: peak VRAM {peak_gib:.2f} GiB exceeds cap "
                f"{args.vram_cap_gib:.2f} GiB at batch={args.batch_size}. "
                f"Reduce batch size or increase gradient accumulation.",
                file=sys.stderr,
            )
            return 2
        headroom = args.vram_cap_gib - peak_gib
        print(
            f"[smoke] OK: peak VRAM {peak_gib:.2f} GiB <= cap "
            f"{args.vram_cap_gib:.2f} GiB (headroom {headroom:.2f} GiB)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
