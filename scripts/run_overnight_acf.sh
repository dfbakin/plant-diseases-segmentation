#!/usr/bin/env bash
###############################################################################
# Overnight chain: A -> C -> F.
#
# A  (~6-8 h)  : seg-probe Phase 1 (aux only) on the converged eq-only ckpt.
#                Trains 6 frozen ProbeHeads (one per position) + evaluates
#                each against 3 non-trainable baselines.
#                -> outputs/spdnet_plantseg/seg_probe_phase1/<AUX_TAG>/
#
# C  (~5-7 h)  : eq_con warmstart from the same eq-only ckpt.
#                Classifier is already converged (val/mAP ~0.86), so
#                L_con is linearly ramped in over 5 epochs from epoch 0
#                and then trained at full weight for 20 more epochs
#                (25 total). Fresh optimizer + LR scheduler.
#                -> outputs/spdnet_aux_losses/spdnet_spatial_eq_con_warmstart_<DATE>/
#
# F  (~14 h)   : eq_con from scratch with L_con warmup (start=14, ramp=7),
#                80 epochs. This is the "correct" eq_con run: lets the
#                classifier reach ~0.6 mAP on L_cls alone before L_con
#                starts shaping spatial features.
#                -> outputs/spdnet_aux_losses/spdnet_spatial_eq_con_warmup_<DATE>/
#
# Sequencing logic: A has no dependency on the new training code, so a bug
# in the warmup/warmstart wiring can only hurt C and F. C tests both the
# warmup schedule AND the +checkpoint= load; F tests only the schedule.
# If C crashes, we still learn something from A and can stop before F.
# If F crashes but C succeeded, we still have the short warmstart result.
#
# Idempotency: a per-phase ``.DONE`` marker under
# ``outputs/_acf_chain/`` causes that phase to be skipped on re-run.
# ``rm outputs/_acf_chain/<X>.DONE`` to force a re-run of phase X.
#
# Monitoring: each phase tees its full stdout/stderr to
# ``logs/acf_<phase>_<timestamp>.log`` so a partially-crashed run is
# easy to triage after the fact.
###############################################################################

set -uo pipefail   # NOT set -e: one failed phase should not abort the chain.

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ----------------------------------------------------------------------------
# Inputs (override via env vars)
# ----------------------------------------------------------------------------

# Best eq-only checkpoint (val/mAP=0.8615 at epoch 72). The path contains
# '=' and '/' chars produced by our ModelCheckpoint filename template;
# downstream launchers single-quote it before handing to Hydra.
CKPT_EQ_ONLY="${CKPT_EQ_ONLY:-outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt}"
AUX_TAG="${AUX_TAG:-spatial_eq_20260424}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Per-phase max_epochs. Defaults match the plan in the header comment.
MAX_EPOCHS_C="${MAX_EPOCHS_C:-40}"
MAX_EPOCHS_F="${MAX_EPOCHS_F:-80}"

# Run-skip markers. ``rm outputs/_acf_chain/A.DONE`` to force re-run of A.
MARKER_DIR="outputs/_acf_chain"
mkdir -p "$MARKER_DIR" logs

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

echo "================================================================"
echo "  A -> C -> F overnight chain"
echo "  Started:     $(date)"
echo "  CKPT_EQ_ONLY: $CKPT_EQ_ONLY"
echo "  AUX_TAG:      $AUX_TAG"
echo "  DATE_TAG:     $DATE_TAG"
echo "  C epochs:     $MAX_EPOCHS_C"
echo "  F epochs:     $MAX_EPOCHS_F"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
echo "================================================================"

if [[ ! -f "$CKPT_EQ_ONLY" ]]; then
    echo "ERROR: eq-only checkpoint missing: $CKPT_EQ_ONLY" >&2
    exit 1
fi

# Quick code sanity: make sure the new warmup cfg fields exist and the
# warmstart plumbing is compiled in. Exits non-zero on any ImportError.
python - <<'PY'
import importlib, sys
from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.wsss.spdnet.lightning import SPDNetModule
# New fields:
assert hasattr(SPDNetSpatialLossesConfig, "con_warmup_start_epoch")
assert hasattr(SPDNetSpatialLossesConfig, "con_warmup_epochs")
# New method:
assert callable(getattr(SPDNetModule, "effective_lambda_con", None))
# Make sure train_spdnet imports cleanly so the warmstart path is syntactically sound.
importlib.import_module("src.train_spdnet")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: preflight code check failed. Fix src/ before launching." >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Phase runner helper
# ----------------------------------------------------------------------------

run_phase() {
    local name="$1"; shift
    local marker="$MARKER_DIR/${name}.DONE"
    local log_path="logs/acf_${name}_$(date +%Y%m%d_%H%M%S).log"

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

    # The `tee` swallows the exit status; pull it out of PIPESTATUS.
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
# A: seg-probe Phase 1 (aux only) on eq-only ckpt
# ----------------------------------------------------------------------------
# We call the existing run_seg_probes_phase1.sh with AUX_ONLY=1 so the
# token/spatial baseline loops are skipped (their ckpts are gone). That
# leaves the 6 aux probes (one per spatial position) plus the Phase 1
# decision gate and SUMMARY.md.

phase_A() {
    AUX_ONLY=1 \
    CKPT_AUX="$CKPT_EQ_ONLY" \
    AUX_TAG="$AUX_TAG" \
        bash scripts/run_seg_probes_phase1.sh
}

# ----------------------------------------------------------------------------
# C: eq_con warmstart from eq-only ckpt
# ----------------------------------------------------------------------------
# Preset `eq_con_warmstart`: lambda_eq=1.0, lambda_con=0.5, con warmup
# start=0, ramp=5. MAX_EPOCHS=25 by default (short because the classifier
# is already at val/mAP ~0.86). +checkpoint= is forwarded by
# run_spdnet_aux_losses_experiments.sh to train_spdnet.py, which then
# loads weights-only via module.load_state_dict (fresh optimizer/scheduler).

phase_C() {
    MAX_EPOCHS="$MAX_EPOCHS_C" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset eq_con_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

# ----------------------------------------------------------------------------
# F: eq_con from scratch with L_con warmup
# ----------------------------------------------------------------------------
# Preset `eq_con_warmup`: lambda_eq=1.0, lambda_con=0.5, con warmup
# start=14 (where the eq-only run reaches val/mAP ~0.6), ramp=7 -> L_con
# reaches its full 0.5 weight at epoch 21. 80 epochs total.

phase_F() {
    MAX_EPOCHS="$MAX_EPOCHS_F" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset eq_con_warmup
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
rc_A=0; rc_C=0; rc_F=0

run_phase A phase_A; rc_A=$?
run_phase C phase_C; rc_C=$?
run_phase F phase_F; rc_F=$?

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  A -> C -> F chain: wall clock $((t1 - t0))s"
echo "================================================================"
printf "  %-5s  rc=%d\n" "A" "$rc_A"
printf "  %-5s  rc=%d\n" "C" "$rc_C"
printf "  %-5s  rc=%d\n" "F" "$rc_F"
echo ""
if (( rc_A == 0 && rc_C == 0 && rc_F == 0 )); then
    echo "All phases succeeded."
    exit 0
fi
echo "At least one phase failed -- inspect logs/acf_<phase>_*.log."
exit 1
