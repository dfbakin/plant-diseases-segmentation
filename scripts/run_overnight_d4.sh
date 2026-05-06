#!/usr/bin/env bash
###############################################################################
# Overnight chain: D4 ablation (4 runs).
#
# The D4 experiment proposed in `reports/notes/aux_loss_next_steps_decision.md`
# triangulates three independent hypotheses distilled from RQ1/RQ2/RQ5:
#
#   H1 -- magnitude rebalancing alone is enough to fix D1.
#   H2 -- L_marg_H beats a safely-scaled L_ac.
#   H3 -- mask_combiner="union" beats "intersection".
#   H4 -- L_mask still helps on top of L_marg_H.
#
# All four phases warmstart from the converged eq-only checkpoint
# (val/mAP=0.8615 at epoch 72). None enable L_eq, L_con, or L_dist -- the
# RQ1 measurement showed those gradients either too small to matter
# (L_eq) or severely out of scale (L_con 22x, L_dist 20x) on that ckpt.
#
# D4_main      (~4 h, 40 ep) : warmstart + L_marg_H (lambda=0.15, beta=0.25)
#                              + L_mask with mask_combiner="union"
#                              (lambda=0.10, alpha=0.25, beta_neg=0.50).
#                              No L_ac, no L_eq/L_con/L_dist. Headline D4.
#                              -> outputs/spdnet_aux_losses/spdnet_spatial_d4_main_warmstart_<DATE>/
#
# D4_attn_only (~4 h, 40 ep) : L_marg_H alone (no L_mask). Isolates the
#                              attention-shaping contribution from the
#                              pseudo-mask supervision. Lets us attribute
#                              any delta in val/cam_iou to the marginal
#                              term rather than the teacher signal.
#                              -> outputs/spdnet_aux_losses/spdnet_spatial_d4_attn_only_warmstart_<DATE>/
#
# D4_ac_safe   (~4 h, 40 ep) : Classical L_ac at the RQ1-calibrated weight
#                              (lambda_ac=0.05, 10x smaller than D1's 0.5)
#                              + L_mask(union). Tests H1 in isolation:
#                              does magnitude rebalancing alone fix the
#                              mode collapse D1 suffered, without needing
#                              the new marginal-entropy term?
#                              -> outputs/spdnet_aux_losses/spdnet_spatial_d4_ac_safe_warmstart_<DATE>/
#
# D4_int       (~4 h, 40 ep) : A/B twin of D4_main with mask_combiner
#                              flipped back to "intersection". Same
#                              lambdas as D4_main so the ONLY change is
#                              the positive-mask combiner. Tests H3.
#                              -> outputs/spdnet_aux_losses/spdnet_spatial_d4_int_warmstart_<DATE>/
#
# Sequencing rationale:
# * D4_main first -- the only preset where all new mechanics fire
#   simultaneously. If it crashes, the other three likely will too, so we
#   learn about the shared L_marg_H / union plumbing before burning more
#   hours.
# * D4_attn_only second -- same L_marg_H code path as D4_main but with
#   L_mask disabled. Narrowest change surface vs D4_main.
# * D4_ac_safe third -- exercises a DIFFERENT attention loss (the old
#   L_ac). Ordering it after the two L_marg_H runs lets us compare it
#   against their attention-dynamics snapshots directly.
# * D4_int last -- only differs from D4_main in one string hyperparameter.
#   If D4_main already failed, D4_int almost certainly will, so we run it
#   last to avoid wasting budget on a guaranteed-correlated failure.
#
# Idempotency: a per-phase ``.DONE`` marker under ``outputs/_d4_chain/``
# causes that phase to be skipped on re-run. ``rm outputs/_d4_chain/D4_main.DONE``
# to force a re-run.
#
# Monitoring: each phase tees its full stdout/stderr to
# ``logs/d4_<phase>_<timestamp>.log``. Every phase writes MLflow scalars
# under experiment ``spdnet_aux_losses`` with the online CAM-IoU metric
# and the new ``train/L_marg_H`` series, so progress is visible in real
# time.
#
# Usage:
#   bash scripts/run_overnight_d4.sh                    # run all 4 phases
#   bash scripts/run_overnight_d4.sh --preflight-only   # validate only
#   PHASES="D4_main D4_int"  bash scripts/run_overnight_d4.sh
#   PHASES="D4_main"         bash scripts/run_overnight_d4.sh
#   MAX_EPOCHS_D4_MAIN=60    bash scripts/run_overnight_d4.sh
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
            sed -n '2,70p' "$0"
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

# Best eq-only checkpoint (val/mAP=0.8615 at epoch 72). All four D4 phases
# warmstart from this exact file. See run_spdnet_aux_losses_experiments.sh
# for how the '=' and '/' chars in the filename are forwarded to Hydra.
CKPT_EQ_ONLY="${CKPT_EQ_ONLY:-outputs/spdnet_aux_losses/spdnet_spatial_eq_20260424/checkpoints/epoch=epoch=72-val_mAP=val/mAP=0.8615.ckpt}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Per-phase max_epochs. Each D4 run warmstarts from a converged classifier
# (val/mAP ~0.86), so 40 epochs is enough to expose any localisation
# signal from the new loss mix. Override individually if needed.
MAX_EPOCHS_D4_MAIN="${MAX_EPOCHS_D4_MAIN:-40}"
MAX_EPOCHS_D4_ATTN_ONLY="${MAX_EPOCHS_D4_ATTN_ONLY:-40}"
MAX_EPOCHS_D4_AC_SAFE="${MAX_EPOCHS_D4_AC_SAFE:-40}"
MAX_EPOCHS_D4_INT="${MAX_EPOCHS_D4_INT:-40}"

# Which phases to run. Default "all". Set to e.g. "D4_main D4_int" to skip
# the other two.
# NOTE: use `${PHASES-...}` rather than `${PHASES:-...}` so that explicitly
# setting ``PHASES=""`` means "run nothing" instead of silently expanding
# to the default list (which would burn hours of GPU on a typo).
PHASES="${PHASES-D4_main D4_attn_only D4_ac_safe D4_int}"

# Run-skip markers.
MARKER_DIR="outputs/_d4_chain"
mkdir -p "$MARKER_DIR" logs

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

echo "================================================================"
echo "  D4 ablation overnight chain (4 phases)"
echo "  Started:              $(date)"
echo "  CKPT_EQ_ONLY:         $CKPT_EQ_ONLY"
echo "  DATE_TAG:             $DATE_TAG"
echo "  phases:               ${PHASES:-<none>}"
echo "  D4_main epochs:       $MAX_EPOCHS_D4_MAIN"
echo "  D4_attn_only epochs:  $MAX_EPOCHS_D4_ATTN_ONLY"
echo "  D4_ac_safe epochs:    $MAX_EPOCHS_D4_AC_SAFE"
echo "  D4_int epochs:        $MAX_EPOCHS_D4_INT"
echo "  preflight only:       $PREFLIGHT_ONLY"
echo "  GPU:                  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
echo "================================================================"

# Validate requested phase tokens early so a typo ("d4_main" instead of
# "D4_main") doesn't silently skip the whole run.
for phase in ${PHASES:-}; do
    case "$phase" in
        D4_main|D4_attn_only|D4_ac_safe|D4_int) : ;;
        *) echo "ERROR: PHASES contains unknown phase '$phase'. Valid: D4_main D4_attn_only D4_ac_safe D4_int." >&2; exit 4 ;;
    esac
done

if [[ ! -f "$CKPT_EQ_ONLY" ]]; then
    echo "ERROR: eq-only checkpoint missing: $CKPT_EQ_ONLY" >&2
    exit 1
fi

# Pre-flight code sanity. Exits non-zero on any ImportError or missing
# symbol. Covers the D4 additions: the new loss function, the new config
# fields, the exposed attn_w in extract_merged_features, and the new
# mask_combiner literal.
python - <<'PY'
import importlib, inspect, sys
from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.wsss.spdnet.lightning import SPDNetModule
from src.wsss.spdnet.spatial_losses import (
    attention_concentration_loss,
    attention_marginal_entropy_loss,
    cam_pseudo_mask_loss,
    patch_contrastive_loss,
)

# D4 new config fields.
for field in (
    # carried forward from D1-D3
    "lambda_ac", "lambda_mask",
    "mask_alpha_pos", "mask_beta_neg", "mask_use_intersection",
    "mask_warmup_start_epoch", "mask_warmup_epochs",
    "con_anchor_source",
    # D4-new
    "lambda_marg_H", "marg_H_beta", "mask_combiner",
):
    assert hasattr(SPDNetSpatialLossesConfig, field), \
        f"missing SPDNetSpatialLossesConfig.{field}"

# New method on SPDNetModule (still valid from D1-D3).
assert callable(getattr(SPDNetModule, "effective_lambda_mask", None))

# cam_pseudo_mask_loss must accept mask_combiner and allow Optional legacy alias.
sig = inspect.signature(cam_pseudo_mask_loss)
assert "mask_combiner" in sig.parameters, "cam_pseudo_mask_loss missing mask_combiner"
assert "use_intersection" in sig.parameters, "cam_pseudo_mask_loss missing use_intersection alias"

# attention_marginal_entropy_loss has the expected signature.
sig_marg = inspect.signature(attention_marginal_entropy_loss)
assert "attn_w" in sig_marg.parameters, "attention_marginal_entropy_loss missing attn_w"
assert "beta" in sig_marg.parameters, "attention_marginal_entropy_loss missing beta"

# train_spdnet module imports (warmstart plumbing).
importlib.import_module("src.train_spdnet")

# Verify attn_w is plumbed through extract_merged_features by a lightweight
# forward pass. Use a tiny random-init SPDNet to avoid downloading weights.
import torch
from src.wsss.spdnet.model import SPDNet
m = SPDNet(num_classes=4, fpn_channels=16, fusion_mode="spatial", pretrained=False).eval()
with torch.no_grad():
    feats = m.extract_merged_features(
        torch.randn(1, 3, 64, 64), [torch.randn(1, 3, 64, 64)], return_attn=True,
    )
assert "attn_w" in feats, "extract_merged_features must expose attn_w when return_attn=True"
assert feats["attn_w"].dim() == 3, f"attn_w should be (B, P, N); got {feats['attn_w'].shape}"

print("pre-flight OK")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: pre-flight code check failed. Fix src/ before launching." >&2
    exit 2
fi

# Smoke: run the aux-loss launcher in --dry-run for each phase so Hydra
# parses the preset before we commit ~16 h of GPU time.
echo ""
echo "Pre-flight: dry-run each preset to validate Hydra overrides..."
for PRESET in d4_main_warmstart d4_attn_only_warmstart d4_ac_safe_warmstart d4_int_warmstart; do
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
    local log_path="logs/d4_${name}_$(date +%Y%m%d_%H%M%S).log"

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

phase_D4_main() {
    MAX_EPOCHS="$MAX_EPOCHS_D4_MAIN" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d4_main_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

phase_D4_attn_only() {
    MAX_EPOCHS="$MAX_EPOCHS_D4_ATTN_ONLY" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d4_attn_only_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

phase_D4_ac_safe() {
    MAX_EPOCHS="$MAX_EPOCHS_D4_AC_SAFE" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d4_ac_safe_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

phase_D4_int() {
    MAX_EPOCHS="$MAX_EPOCHS_D4_INT" \
    DATE_TAG="$DATE_TAG" \
        bash scripts/run_spdnet_aux_losses_experiments.sh \
            --preset d4_int_warmstart \
            --from-checkpoint "$CKPT_EQ_ONLY"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
declare -A RC=( [D4_main]=-1 [D4_attn_only]=-1 [D4_ac_safe]=-1 [D4_int]=-1 )

if [[ -z "${PHASES// }" ]]; then
    echo "PHASES is empty -> nothing to dispatch. Exiting."
    exit 0
fi

for phase in ${PHASES}; do
    case "$phase" in
        D4_main)       run_phase D4_main       phase_D4_main;       RC[D4_main]=$? ;;
        D4_attn_only)  run_phase D4_attn_only  phase_D4_attn_only;  RC[D4_attn_only]=$? ;;
        D4_ac_safe)    run_phase D4_ac_safe    phase_D4_ac_safe;    RC[D4_ac_safe]=$? ;;
        D4_int)        run_phase D4_int        phase_D4_int;        RC[D4_int]=$? ;;
    esac
done

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  D4 ablation chain: wall clock $((t1 - t0))s ($(((t1 - t0) / 3600))h)"
echo "================================================================"
for p in D4_main D4_attn_only D4_ac_safe D4_int; do
    if [[ "${RC[$p]}" == "-1" ]]; then
        printf "  %-14s  SKIPPED (not in PHASES)\n" "$p"
    else
        printf "  %-14s  rc=%d\n" "$p" "${RC[$p]}"
    fi
done
echo ""
echo "MLflow runs to inspect:"
echo "  D4_main:       spdnet_spatial_d4_main_warmstart_${DATE_TAG}"
echo "  D4_attn_only:  spdnet_spatial_d4_attn_only_warmstart_${DATE_TAG}"
echo "  D4_ac_safe:    spdnet_spatial_d4_ac_safe_warmstart_${DATE_TAG}"
echo "  D4_int:        spdnet_spatial_d4_int_warmstart_${DATE_TAG}"
echo ""
echo "Key MLflow metrics to compare against eq-only and D1/D2/D3 baselines:"
echo "  val/cam_iou_best      -- online CAM IoU at single best threshold"
echo "  val/cam_iou_best_thr  -- the threshold that produced the best"
echo "  val/cam_iou_auc       -- AUC over the threshold sweep"
echo "  val/mAP               -- classification accuracy (regression check)"
echo "  train/L_marg_H        -- D4_main, D4_attn_only, D4_int; should trend negative"
echo "  train/L_ac            -- D4_ac_safe only"
echo "  train/L_mask          -- D4_main, D4_ac_safe, D4_int"
echo ""
echo "Post-run analysis: run scripts/measure_rq4_attn_dynamics.py on each new"
echo "checkpoint and compare against the RQ4 baseline table in"
echo "reports/notes/rq4_attn_dynamics.md; write reports/notes/rq_d4_ablation_results.md."
echo ""

# Exit non-zero if ANY phase we tried to run failed.
any_fail=0
for p in D4_main D4_attn_only D4_ac_safe D4_int; do
    if [[ "${RC[$p]}" != "-1" && "${RC[$p]}" != "0" ]]; then
        any_fail=1
    fi
done
if (( any_fail )); then
    echo "At least one phase failed -- inspect logs/d4_<phase>_*.log."
    exit 1
fi
echo "All requested phases succeeded."
exit 0
