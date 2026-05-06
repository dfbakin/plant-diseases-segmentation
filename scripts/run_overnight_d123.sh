#!/usr/bin/env bash
###############################################################################
# Overnight chain: D1 -> D2 -> D3.
#
# The three diagnostic interventions proposed in RESEARCH_CONTEXT.md §5.13.7
# ("What would actually move the metric"), launched sequentially on top of
# the converged equivariance-only checkpoint.
#
# D1  (~4 h, 40 ep) : warmstart + attention-concentration regulariser
#                     (lambda_ac=0.5). Replaces L_eq. Measures whether
#                     breaking the uniform-attention fixed point alone is
#                     enough to move val/cam_iou_best. Cheapest test.
#                     -> outputs/spdnet_aux_losses/spdnet_spatial_d1_ac_warmstart_<DATE>/
#
# D2  (~4 h, 40 ep) : warmstart + pseudo-mask CAM supervision (lambda_mask=1.0,
#                     alpha_pos=0.25, beta_neg=0.5, intersection=true).
#                     Direct supervision of the active-class CAM slice
#                     against chvar-derived positives intersected with the
#                     CAM's own top-alpha. If this doesn't move
#                     val/cam_iou_best, no aux loss will.
#                     -> outputs/spdnet_aux_losses/spdnet_spatial_d2_mask_warmstart_<DATE>/
#
# D3  (~5 h, 40 ep) : D2 + L_con with union anchors
#                     (anchor_source=union_cls_chvar, lambda_con=0.5, con
#                     warmup 0..5). Stacks union-anchor contrastive learning
#                     on top of pseudo-mask supervision. Intended to run
#                     *only if D2 shows signal* -- but is cheap enough to
#                     launch unconditionally so we don't lose GPU time.
#                     -> outputs/spdnet_aux_losses/spdnet_spatial_d3_d2plus_union_warmstart_<DATE>/
#
# Sequencing rationale (same reasoning as ACF):
# * D1 has the smallest change surface -- one new loss, no data-mask
#   construction. If D1 crashes, D2 and D3 probably will too, so we learn
#   about the shared warmstart+training plumbing first.
# * D2 tests the pseudo-mask pipeline (chvar + intersection); if it NaNs
#   we'd rather catch it before also turning on L_con in D3.
# * D3 runs last so we have standalone D2 numbers even if D3 itself is
#   unstable.
#
# Idempotency: a per-phase ``.DONE`` marker under ``outputs/_d123_chain/``
# causes that phase to be skipped on re-run. ``rm outputs/_d123_chain/D1.DONE``
# to force a re-run.
#
# Monitoring: each phase tees its full stdout/stderr to
# ``logs/d123_<phase>_<timestamp>.log``. Every phase also writes MLflow
# scalars under experiment ``spdnet_aux_losses`` with the online CAM-IoU
# metric, so progress is visible in real time.
#
# Usage:
#   bash scripts/run_overnight_d123.sh                 # run D1 -> D2 -> D3
#   bash scripts/run_overnight_d123.sh --preflight-only  # validate only, no training
#   PHASES="D2 D3"  bash scripts/run_overnight_d123.sh # skip D1
#   PHASES="D1"     bash scripts/run_overnight_d123.sh # run D1 only
#   MAX_EPOCHS_D1=60 bash scripts/run_overnight_d123.sh # override per-phase length
###############################################################################

set -uo pipefail   # NOT set -e: one failed phase should not abort the chain.

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ----------------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------------

PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        -h|--help)
            sed -n '2,54p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'. Try --help." >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Inputs (override via env vars)
# ----------------------------------------------------------------------------

# Best eq-only checkpoint (val/mAP=0.8615 at epoch 72). All three D phases
# warmstart from this exact file. See run_spdnet_aux_losses_experiments.sh
# for how the '=' and '/' chars in the filename are forwarded to Hydra.
CKPT_EQ_ONLY="${CKPT_EQ_ONLY:-outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Per-phase max_epochs. Each D run warmstarts from a converged classifier
# (val/mAP ~0.86), so 40 epochs is enough to expose any localisation
# signal from the new loss; longer runs are a waste if the signal is
# absent (null-result budget: we'd rather see 3 independent 40-ep runs
# than 1 long 120-ep one). Override individually if needed.
MAX_EPOCHS_D1="${MAX_EPOCHS_D1:-40}"
MAX_EPOCHS_D2="${MAX_EPOCHS_D2:-40}"
MAX_EPOCHS_D3="${MAX_EPOCHS_D3:-40}"

# Which phases to run. Default "all". Set to e.g. "D2 D3" to skip D1.
# NOTE: use `${PHASES-...}` rather than `${PHASES:-...}` so that explicitly
# setting ``PHASES=""`` means "run nothing" instead of silently expanding to
# the default "D1 D2 D3" (which would burn hours of GPU on a typo).
PHASES="${PHASES-D1 D2 D3}"

# Run-skip markers.
MARKER_DIR="outputs/_d123_chain"
mkdir -p "$MARKER_DIR" logs

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

echo "================================================================"
echo "  D1 -> D2 -> D3 overnight chain"
echo "  Started:       $(date)"
echo "  CKPT_EQ_ONLY:  $CKPT_EQ_ONLY"
echo "  DATE_TAG:      $DATE_TAG"
echo "  phases:        ${PHASES:-<none>}"
echo "  D1 epochs:     $MAX_EPOCHS_D1"
echo "  D2 epochs:     $MAX_EPOCHS_D2"
echo "  D3 epochs:     $MAX_EPOCHS_D3"
echo "  preflight only: $PREFLIGHT_ONLY"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
echo "================================================================"

# Validate requested phase tokens early so a typo ("d1" instead of "D1")
# doesn't silently skip the whole run.
for phase in ${PHASES:-}; do
    case "$phase" in
        D1|D2|D3) : ;;
        *) echo "ERROR: PHASES contains unknown phase '$phase'. Valid: D1 D2 D3." >&2; exit 4 ;;
    esac
done

if [[ ! -f "$CKPT_EQ_ONLY" ]]; then
    echo "ERROR: eq-only checkpoint missing: $CKPT_EQ_ONLY" >&2
    exit 1
fi

# Pre-flight code sanity. Exits non-zero on any ImportError or missing
# symbol. Covers all three new loss functions + the warmup schedule field.
python - <<'PY'
import importlib, sys
from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.wsss.spdnet.lightning import SPDNetModule
from src.wsss.spdnet.spatial_losses import (
    attention_concentration_loss,
    cam_pseudo_mask_loss,
    patch_contrastive_loss,
)

# New config fields
for field in (
    "lambda_ac", "lambda_mask",
    "mask_alpha_pos", "mask_beta_neg", "mask_use_intersection",
    "mask_warmup_start_epoch", "mask_warmup_epochs",
    "con_anchor_source",
):
    assert hasattr(SPDNetSpatialLossesConfig, field), \
        f"missing SPDNetSpatialLossesConfig.{field}"

# New method on SPDNetModule
assert callable(getattr(SPDNetModule, "effective_lambda_mask", None))

# patch_contrastive_loss must accept anchor_source
import inspect
assert "anchor_source" in inspect.signature(patch_contrastive_loss).parameters

# train_spdnet module imports (warmstart plumbing)
importlib.import_module("src.train_spdnet")
print("pre-flight OK")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: pre-flight code check failed. Fix src/ before launching." >&2
    exit 2
fi

# Smoke: run the aux-loss launcher in --dry-run for each phase so Hydra
# parses the preset before we commit ~12 h of GPU time.
echo ""
echo "Pre-flight: dry-run each preset to validate Hydra overrides..."
for PRESET in d1_ac_warmstart d2_mask_warmstart d3_d2plus_union_warmstart; do
    if ! bash scripts/run_spdnet_aux_losses_experiments.sh \
        --dry-run --preset "$PRESET" \
        --from-checkpoint "$CKPT_EQ_ONLY" \
        > /tmp/dryrun_${PRESET}.log 2>&1
    then
        echo "ERROR: dry-run for '$PRESET' failed. See /tmp/dryrun_${PRESET}.log" >&2
        tail -20 /tmp/dryrun_${PRESET}.log >&2
        exit 3
    fi
    echo "  $PRESET: dry-run OK"
done

if (( PREFLIGHT_ONLY )); then
    echo ""
    echo "Pre-flight OK. --preflight-only supplied -> exiting before dispatch."
    exit 0
fi

# ----------------------------------------------------------------------------
# Phase runner helper
# ----------------------------------------------------------------------------

run_phase() {
    local name="$1"; shift
    local marker="$MARKER_DIR/${name}.DONE"
    local log_path="logs/d123_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "  [${name}]  start"
    echo "  log:        $log_path"
    echo "  marker:     $marker"
    echo "  started:    $(date)"
    echo "================================================================"

    if [[ -f "$marker" ]]; then
        echo "[${name}] marker exists -- skipping. (rm $marker to force re-run.)"
        return 0
    fi

    "$@" 2>&1 | tee "$log_path"
    local rc=${PIPESTATUS[0]}
    echo ""
    echo "[${name}] done (rc=$rc) at $(date)"
    if (( rc == 0 )); then
        touch "$marker"
    else
        echo "[${name}] FAILED -- see $log_path; not writing $marker."
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# Phase definitions
# ----------------------------------------------------------------------------

phase_D1() {
    MAX_EPOCHS="$MAX_EPOCHS_D1" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d1_ac_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

phase_D2() {
    MAX_EPOCHS="$MAX_EPOCHS_D2" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d2_mask_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

phase_D3() {
    MAX_EPOCHS="$MAX_EPOCHS_D3" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d3_d2plus_union_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
declare -A RC=( [D1]=-1 [D2]=-1 [D3]=-1 )

if [[ -z "${PHASES// }" ]]; then
    echo "PHASES is empty -> nothing to dispatch. Exiting."
    exit 0
fi

for phase in ${PHASES}; do
    case "$phase" in
        D1)  run_phase D1 phase_D1; RC[D1]=$? ;;
        D2)  run_phase D2 phase_D2; RC[D2]=$? ;;
        D3)  run_phase D3 phase_D3; RC[D3]=$? ;;
    esac
done

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  D1 -> D2 -> D3 chain: wall clock $((t1 - t0))s ($(((t1 - t0) / 3600))h)"
echo "================================================================"
for p in D1 D2 D3; do
    if [[ "${RC[$p]}" == "-1" ]]; then
        printf "  %-3s  SKIPPED (not in PHASES)\n" "$p"
    else
        printf "  %-3s  rc=%d\n" "$p" "${RC[$p]}"
    fi
done
echo ""
echo "MLflow runs to inspect:"
echo "  D1: spdnet_spatial_d1_ac_warmstart_${DATE_TAG}"
echo "  D2: spdnet_spatial_d2_mask_warmstart_${DATE_TAG}"
echo "  D3: spdnet_spatial_d3_d2plus_union_warmstart_${DATE_TAG}"
echo ""
echo "Key MLflow metrics to compare against eq-only baseline:"
echo "  val/cam_iou_best      -- online CAM IoU at single best threshold"
echo "  val/cam_iou_best_thr  -- the threshold that produced the best"
echo "  val/mAP               -- classification accuracy (regression check)"
echo "  train/L_ac            -- D1 only; should trend toward -1 (peak concentration)"
echo "  train/attn_mean       -- D1 only; should trend toward 1.0 (peak concentration)"
echo "  train/L_mask          -- D2, D3; should trend toward 0 as CAM matches pseudo-mask"
echo "  train/L_con           -- D3 only; should follow usual InfoNCE trajectory"
echo ""

# Exit non-zero if ANY phase we tried to run failed.
any_fail=0
for p in D1 D2 D3; do
    if [[ "${RC[$p]}" != "-1" && "${RC[$p]}" != "0" ]]; then
        any_fail=1
    fi
done
if (( any_fail )); then
    echo "At least one phase failed -- inspect logs/d123_<phase>_*.log."
    exit 1
fi
echo "All requested phases succeeded."
exit 0
