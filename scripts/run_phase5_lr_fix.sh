#!/usr/bin/env bash
###############################################################################
# Phase 5 LR/SCA scaling-trap fix verification chain.
#
# Context (RESEARCH_CONTEXT.md §5.14): the four 896² runs in
# ``outputs/phase5_highres/`` regressed both classification (val/mAP
# 0.838 vs the 448 baseline 0.888) and localisation (val/cam_iou_best
# 0.241 vs 0.262), and every aux-loss run collapsed the attention map
# within 5 epochs. §5.14.2 traced the regression to four scale traps:
#
#   Trap 1 -- ``scaled_lr = base_lr * batch_size / 256`` ignored
#             ``accumulate_grad_batches``, so any high-res config that
#             traded micro-batch for accum got HALF the appropriate LR
#             at the same effective batch (the H4 cls-only run used
#             eff_batch=30 but only the per-step batch=6 entered the
#             scaling formula, gradient-starving the classifier).
#   Trap 2 -- ``SpatialCrossAttention.ref_pool_size`` was hard-coded to
#             14 regardless of image_size. At 896² this leaves Q:K =
#             224²:14² = 256:1 (vs 64:1 at 448²) -- the attention
#             collapse fixed point becomes 4× harder to escape.
#   Trap 3 -- ``min_lr >= scaled_lr`` flips CosineAnnealingLR upside
#             down (already fixed with a guard in lightning.py
#             configure_optimizers).
#   Trap 4 -- val transform used bilinear on the GT mask (false alarm:
#             albumentations defaults mask_interpolation=NEAREST; the
#             GT pipeline was already correct). Now made explicit.
#
# This launcher runs four sequential MLflow phases that exercise the
# Trap-1 + Trap-2 fixes in isolation, then layers aux losses on top:
#
#   P1  CLS_ONLY    (~16 h, 60 ep)  -- pure classifier at 896 with
#                                     fixed LR scaling. Test of Trap 1
#                                     in isolation. Expected
#                                     ``val/mAP >= 0.85`` (closing the
#                                     5-pp gap to the 448 baseline) and
#                                     ``val/cam_iou_best ~ 0.24-0.27``
#                                     (classifier-only baseline).
#                                     Also adopts the auto ref_pool_size
#                                     (rps=20 at 896 vs legacy 14) so
#                                     the classifier already sees 4×
#                                     more attention bandwidth even
#                                     though no aux loss exploits it.
#
#   P2  AUX_MASK    (~24 h, 80 ep, from scratch)
#                   D2-style pseudo-mask supervision only -- no
#                   attention regulariser, hence no risk of D1-style
#                   attention collapse. ``L_mask`` warmup [15, 20] gives
#                   the classifier 15 epochs to converge on top of the
#                   chvar∪cam_top-α teacher from RQ5. ``lambda_mask=0.05``
#                   is HALF the D4 RQ1-balanced value (0.10) -- explicit
#                   "regulariser, not competitor" calibration.
#
#   P3  AUX_MARGH   (~24 h, 80 ep, from scratch)
#                   D4-style L_marg_H + L_mask, both at half the
#                   RQ1-balanced values (lambda_marg_H=0.075,
#                   lambda_mask=0.05) and with longer warmups
#                   (start=18, ramp=8). Tests the RQ2 collapse-resistant
#                   attention loss at "regulariser, not competitor"
#                   strength on a from-scratch 896 run.
#
#   P4  AUX_AC_TINY (~24 h, 80 ep, from scratch)
#                   D1-style L_ac at tiny lambda (0.05), AS A CONTROL
#                   for whether L_ac with magnitude alone (no L_marg_H)
#                   can avoid collapse. Pairs with L_mask(union) at
#                   lambda=0.05. Same warmups as P3. RQ1's all-trainable
#                   λ* for L_ac was 0.15; we use 1/3 of that here, so
#                   gradient is ~33% of L_cls's RMS, which the §5.13
#                   D1 analysis predicts is below the collapse threshold.
#
# All four phases:
#   - run on 896² from a freshly-initialised SPDNet (no warmstart),
#   - use the new effective-batch LR rule (Trap 1 fix),
#   - use the auto ref_pool_size = max(14, image_size//44) = 20 at 896
#     (Trap 2 fix; can be overridden via REF_POOL_SIZE env var),
#   - log to MLflow experiment ``phase5_lr_fix``.
#
# Sequencing rationale:
#   * P1 first because it tests the LR fix in isolation. If it fails
#     (val/mAP at ep 30 < 0.45 -- below the H4 trajectory at the same
#     epoch) it makes no sense to launch P2/P3/P4 before fixing the
#     base optimisation; all aux runs would inherit the same gradient
#     starvation.
#   * P2 (mask-only, no attention regulariser) is the safest aux run.
#     If it ALSO fails, the issue is Trap 2 (SCA bandwidth) or a deeper
#     problem we haven't identified.
#   * P3 (L_marg_H + L_mask) tests the RQ2 attention regulariser which
#     was designed to NOT have a trivial minimum at uniform attention.
#   * P4 (L_ac at tiny λ) is the diagnostic baseline to attribute any
#     gain in P3 to L_marg_H specifically rather than to the
#     warmup-aware schedule alone.
#
# Idempotency: per-phase ``.DONE`` markers under ``outputs/_phase5_lr_fix/``.
# Remove a marker to force re-run.
#
# Memory envelope (verified by smoke_test_spdnet_highres.py):
#   * 896² + batch=6 + rps=14 + cls_only -> 23 GiB peak
#   * 896² + batch=6 + rps=20 + cls_only -> 26 GiB peak (P1)
#   * 896² + batch=4 + rps=20 + aux ON   -> 21 GiB peak (P2/P3/P4)
#   * 896² + batch=2 + rps=40 + aux ON   -> 22 GiB peak (alt)
# All numbers are from the smoke script on a 32 GiB RTX 5090 (bf16-mixed).
# At batch=4 + accum=8 we maintain effective batch 32, matching the 448
# baseline that produced val/mAP=0.888.
#
# Usage:
#   bash scripts/run_phase5_lr_fix.sh                       # all 4 phases
#   bash scripts/run_phase5_lr_fix.sh --preflight-only      # validate, don't train
#   PHASES="P1"             bash scripts/run_phase5_lr_fix.sh
#   PHASES="P2 P3 P4"       bash scripts/run_phase5_lr_fix.sh
#   MAX_EPOCHS_P1=80        bash scripts/run_phase5_lr_fix.sh
#   REF_POOL_SIZE=28        bash scripts/run_phase5_lr_fix.sh   # override auto
#   IMAGE_SIZE=448          bash scripts/run_phase5_lr_fix.sh   # smoke at 448
#
# Logs:    logs/phase5_lr_fix_<phase>_<timestamp>.log
# Outputs: outputs/phase5_lr_fix/<phase>_<DATE>/
# MLflow:  experiment "phase5_lr_fix", run names "phase5_lr_fix_<phase>_<DATE>"
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
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) sed -n '2,90p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. Try --help." >&2; exit 1 ;;
    esac
done

# ----------------------------------------------------------------------------
# Configurable knobs (override via env vars at top of call)
# ----------------------------------------------------------------------------

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
IMAGE_SIZE="${IMAGE_SIZE:-896}"
INCLUDE_PV="${INCLUDE_PV:-true}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-100}"
AUGMENTATION="${AUGMENTATION:-heavy}"
NUM_REFS="${NUM_REFS:-1}"

# Per-phase batch / accum / epoch counts.
#
# P1 (cls_only):     batch=6, accum=5  -> eff_batch=30 (matches H4 baseline)
# P2/P3/P4 (aux):    batch=4, accum=8  -> eff_batch=32 (matches 448 baseline)
# Both use the new LR rule which scales by eff_batch / 256.
P1_BATCH="${P1_BATCH:-6}"
P1_ACCUM="${P1_ACCUM:-5}"
AUX_BATCH="${AUX_BATCH:-4}"
AUX_ACCUM="${AUX_ACCUM:-8}"

MAX_EPOCHS_P1="${MAX_EPOCHS_P1:-60}"
MAX_EPOCHS_P2="${MAX_EPOCHS_P2:-80}"
MAX_EPOCHS_P3="${MAX_EPOCHS_P3:-80}"
MAX_EPOCHS_P4="${MAX_EPOCHS_P4:-80}"

# LR base. The new scaling rule yields scaled_lr = LR_BASE * eff_batch/256.
# At eff_batch=30: scaled_lr = LR_BASE * 0.117. With LR_BASE=5e-4 (the 448
# spec) this gives ~5.86e-5, slightly above the 448 reference 6.25e-5
# (eff_batch=32) -- well within the regime that produced val/mAP=0.888.
LR_BASE="${LR_BASE:-0.0005}"

# SCA reference pool grid side length. 0 means "auto" (=20 at 896 via the
# Trap-2 fix). Override to e.g. 14 to reproduce legacy behaviour.
REF_POOL_SIZE="${REF_POOL_SIZE:-0}"

# Warmup schedules for P2/P3/P4. The aux-loss is held at 0 for the first
# WARMUP_START epochs while the classifier converges, then linearly
# ramps to its full lambda over WARMUP_RAMP epochs. With start=15,
# ramp=5, full strength is reached at epoch 20 -- the classifier should
# already be at val/mAP ~0.5-0.6 by then on the 896 trajectory, giving
# the aux loss a meaningful target.
WARMUP_START="${WARMUP_START:-15}"
WARMUP_RAMP="${WARMUP_RAMP:-5}"

# Which phases to run. Default "all".
PHASES="${PHASES-P1 P2 P3 P4}"

MARKER_DIR="outputs/_phase5_lr_fix"
mkdir -p "$MARKER_DIR" logs

# ----------------------------------------------------------------------------
# Pre-flight banner
# ----------------------------------------------------------------------------

echo "================================================================"
echo "  Phase 5 LR-fix verification chain"
echo "  Started:                $(date)"
echo "  DATE_TAG:               $DATE_TAG"
echo "  IMAGE_SIZE:             $IMAGE_SIZE"
echo "  REF_POOL_SIZE:          $REF_POOL_SIZE  (0 = auto -> max(14, $IMAGE_SIZE/44))"
echo "  LR_BASE:                $LR_BASE  (per-config; eff_batch//256-scaled)"
echo "  P1 (cls_only):          batch=$P1_BATCH accum=$P1_ACCUM eff_batch=$((P1_BATCH * P1_ACCUM)) ep=$MAX_EPOCHS_P1"
echo "  P2/3/4 (aux):           batch=$AUX_BATCH accum=$AUX_ACCUM eff_batch=$((AUX_BATCH * AUX_ACCUM))"
echo "  Aux warmup:             start=$WARMUP_START ramp=$WARMUP_RAMP"
echo "  Include PlantVillage:   $INCLUDE_PV"
echo "  Phases:                 ${PHASES:-<none>}"
echo "  Preflight only:         $PREFLIGHT_ONLY"
echo "  GPU:                    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
echo "  Free VRAM:              $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
echo "================================================================"

# Validate requested phase tokens early.
for phase in ${PHASES:-}; do
    case "$phase" in
        P1|P2|P3|P4) : ;;
        *) echo "ERROR: PHASES contains unknown phase '$phase'. Valid: P1 P2 P3 P4." >&2; exit 4 ;;
    esac
done

# ----------------------------------------------------------------------------
# Pre-flight code sanity. Verifies the four trap fixes are present.
# ----------------------------------------------------------------------------

echo ""
echo "Pre-flight: verifying trap-fix code is present..."
python - <<'PY'
import sys, inspect
from src.conf.spdnet import SPDNetModelConfig, SPDNetSpatialLossesConfig
from src.wsss.spdnet.model import SPDNet, SpatialCrossAttention
from src.wsss.spdnet.lightning import SPDNetModule
from src import train_spdnet  # ensures the module imports cleanly

# Trap 1: SPDNetModelConfig must expose learning_rate_override.
assert hasattr(SPDNetModelConfig, "learning_rate_override"), \
    "Trap 1 fix missing: SPDNetModelConfig.learning_rate_override"

# Trap 2: SPDNetModelConfig must expose ref_pool_size and SPDNet must
# accept it (it must reach SpatialCrossAttention).
assert hasattr(SPDNetModelConfig, "ref_pool_size"), \
    "Trap 2 fix missing: SPDNetModelConfig.ref_pool_size"
sig = inspect.signature(SPDNet.__init__)
assert "ref_pool_size" in sig.parameters, \
    "Trap 2 fix missing: SPDNet.__init__(ref_pool_size=...)"
sig_l = inspect.signature(SPDNetModule.__init__)
assert "ref_pool_size" in sig_l.parameters, \
    "Trap 2 fix missing: SPDNetModule.__init__(ref_pool_size=...)"

# Trap 3: lightning.py configure_optimizers must raise ValueError on
# min_lr >= scaled_lr (string match, not exec).
import src.wsss.spdnet.lightning as L
src = inspect.getsource(L.SPDNetModule.configure_optimizers)
assert "min_lr" in src and "ValueError" in src, \
    "Trap 3 guard missing in lightning.py configure_optimizers"

# Trap 4: online_loc_metric.py mask resize must explicitly request
# mask_interpolation=0 (NEAREST).
import src.wsss.spdnet.online_loc_metric as M
src_m = inspect.getsource(M.OnlineCAMIoU.__init__)
assert "mask_interpolation" in src_m, \
    "Trap 4 verification missing in online_loc_metric.py"

# Quick functional smoke: a tiny SPDNet at the requested ref_pool_size.
import torch
m = SPDNet(num_classes=4, fpn_channels=16, fusion_mode="spatial",
           pretrained=False, ref_pool_size=20).eval()
with torch.no_grad():
    feats = m.extract_merged_features(
        torch.randn(1, 3, 64, 64), [torch.randn(1, 3, 64, 64)],
        return_attn=True,
    )
assert "attn_w" in feats and feats["attn_w"].shape[-1] == 400, \
    f"ref_pool_size=20 must give 400 keys; got {feats['attn_w'].shape}"

print("pre-flight OK")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: pre-flight code check failed. Fix src/ before launching." >&2
    exit 2
fi

if (( PREFLIGHT_ONLY )); then
    echo ""
    echo "--preflight-only supplied -> exiting before dispatch."
    exit 0
fi

# ----------------------------------------------------------------------------
# Run-phase helper
# ----------------------------------------------------------------------------

run_phase() {
    local name="$1"; shift
    local marker="$MARKER_DIR/${name}.DONE"
    local log_path="logs/phase5_lr_fix_${name}_$(date +%Y%m%d_%H%M%S).log"

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
# Common Hydra arguments shared by every phase.
# ----------------------------------------------------------------------------

# Note: ``model.learning_rate=$LR_BASE`` is the BASE (pre-scaling) LR;
# train_spdnet.py multiplies by eff_batch/256. We pass it via Hydra so
# the run config records the base, while the actual optimizer LR is
# what shows up in MLflow under ``learning_rate`` (post-scaling).
common_hydra_args() {
    local run_name="$1"
    local batch="$2"
    local accum="$3"
    local max_epochs="$4"

    cat <<EOF
run_name=${run_name}
experiment_name=phase5_lr_fix
model.fusion_mode=spatial
model.input_size=${IMAGE_SIZE}
model.learning_rate=${LR_BASE}
model.ref_pool_size=${REF_POOL_SIZE}
trainer.max_epochs=${max_epochs}
trainer.log_every_n_steps=${LOG_EVERY}
trainer.accumulate_grad_batches=${accum}
trainer.precision=bf16-mixed
trainer.warmup_epochs=5
trainer.min_lr=1e-6
data.image_size=${IMAGE_SIZE}
data.batch_size=${batch}
data.num_references=${NUM_REFS}
data.augmentation=${AUGMENTATION}
data.include_plantvillage=${INCLUDE_PV}
data.num_workers=${NUM_WORKERS}
EOF
}

# ----------------------------------------------------------------------------
# Phase definitions
# ----------------------------------------------------------------------------

phase_P1_cls_only() {
    local run_name="phase5_lr_fix_P1_cls_only_${DATE_TAG}"
    local args
    args="$(common_hydra_args "$run_name" "$P1_BATCH" "$P1_ACCUM" "$MAX_EPOCHS_P1")"

    # Pure classifier: every aux lambda zero, online metric ON.
    args+="
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0
losses.lambda_marg_H=0
losses.lambda_mask=0
losses.online_loc_eval_enabled=true
"
    # Disable accum>1 for cls-only if user explicitly requested batch >= 16
    # (matches existing 448 behaviour at large per-step batch).
    python -m src.train_spdnet $args
}

phase_P2_aux_mask_only() {
    local run_name="phase5_lr_fix_P2_aux_mask_only_${DATE_TAG}"
    local args
    args="$(common_hydra_args "$run_name" "$AUX_BATCH" "$AUX_ACCUM" "$MAX_EPOCHS_P2")"

    # D2-style: pseudo-mask supervision only. No attention regulariser
    # so no risk of D1-style collapse. lambda_mask=0.05 is HALF the D4
    # RQ1-balanced value (0.10) -- more conservative because we want a
    # gentle "regulariser not competitor" signal at high resolution.
    # mask_combiner=union (RQ5 winner: chvar∪cam_top-α teacher IoU 0.29
    # vs intersection 0.26).
    args+="
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0
losses.lambda_marg_H=0
losses.lambda_mask=0.05
losses.mask_alpha_pos=0.25
losses.mask_beta_neg=0.50
losses.mask_combiner=union
losses.mask_use_intersection=null
losses.mask_warmup_start_epoch=${WARMUP_START}
losses.mask_warmup_epochs=${WARMUP_RAMP}
losses.online_loc_eval_enabled=true
"
    python -m src.train_spdnet $args
}

phase_P3_aux_marg_H() {
    local run_name="phase5_lr_fix_P3_aux_marg_H_${DATE_TAG}"
    local args
    args="$(common_hydra_args "$run_name" "$AUX_BATCH" "$AUX_ACCUM" "$MAX_EPOCHS_P3")"

    # D4-style: L_marg_H (collapse-resistant attention regulariser)
    # + L_mask. Both at HALF the RQ1-balanced values (D4 used
    # lambda_marg_H=0.15, lambda_mask=0.10). Longer warmup
    # (start=18, ramp=8 -> full strength at ep 26) than P2 because
    # both losses ramp simultaneously.
    args+="
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0
losses.lambda_marg_H=0.075
losses.marg_H_beta=0.25
losses.lambda_mask=0.05
losses.mask_alpha_pos=0.25
losses.mask_beta_neg=0.50
losses.mask_combiner=union
losses.mask_use_intersection=null
losses.mask_warmup_start_epoch=$((WARMUP_START + 3))
losses.mask_warmup_epochs=$((WARMUP_RAMP + 3))
losses.ac_warmup_start_epoch=$((WARMUP_START + 3))
losses.ac_warmup_epochs=$((WARMUP_RAMP + 3))
losses.online_loc_eval_enabled=true
"
    python -m src.train_spdnet $args
}

phase_P4_aux_ac_tiny() {
    local run_name="phase5_lr_fix_P4_aux_ac_tiny_${DATE_TAG}"
    local args
    args="$(common_hydra_args "$run_name" "$AUX_BATCH" "$AUX_ACCUM" "$MAX_EPOCHS_P4")"

    # D1-style L_ac at TINY lambda (1/3 of the RQ1 all-trainable λ*).
    # Pairs with L_mask(union) at the same gentle 0.05 lambda. Tests
    # whether reduced magnitude alone (without L_marg_H's mathematical
    # collapse-resistance) is enough to avoid the §5.13.6 trivial
    # minimum at uniform attention. Same warmups as P2 (mask) and
    # P3-shifted (ac).
    args+="
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0.05
losses.ac_warmup_start_epoch=${WARMUP_START}
losses.ac_warmup_epochs=${WARMUP_RAMP}
losses.lambda_marg_H=0
losses.lambda_mask=0.05
losses.mask_alpha_pos=0.25
losses.mask_beta_neg=0.50
losses.mask_combiner=union
losses.mask_use_intersection=null
losses.mask_warmup_start_epoch=${WARMUP_START}
losses.mask_warmup_epochs=${WARMUP_RAMP}
losses.online_loc_eval_enabled=true
"
    python -m src.train_spdnet $args
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
declare -A RC=( [P1]=-1 [P2]=-1 [P3]=-1 [P4]=-1 )

if [[ -z "${PHASES// }" ]]; then
    echo "PHASES is empty -> nothing to dispatch. Exiting."
    exit 0
fi

for phase in ${PHASES}; do
    case "$phase" in
        P1) run_phase P1 phase_P1_cls_only;     RC[P1]=$? ;;
        P2) run_phase P2 phase_P2_aux_mask_only; RC[P2]=$? ;;
        P3) run_phase P3 phase_P3_aux_marg_H;    RC[P3]=$? ;;
        P4) run_phase P4 phase_P4_aux_ac_tiny;   RC[P4]=$? ;;
    esac
done

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Phase 5 LR-fix chain: wall clock $((t1 - t0))s ($(((t1 - t0) / 3600))h)"
echo "================================================================"
for p in P1 P2 P3 P4; do
    if [[ "${RC[$p]}" == "-1" ]]; then
        printf "  %-3s  SKIPPED (not in PHASES)\n" "$p"
    else
        printf "  %-3s  rc=%d\n" "$p" "${RC[$p]}"
    fi
done

echo ""
echo "MLflow runs to inspect (experiment 'phase5_lr_fix'):"
echo "  P1 cls_only:        phase5_lr_fix_P1_cls_only_${DATE_TAG}"
echo "  P2 mask_only:       phase5_lr_fix_P2_aux_mask_only_${DATE_TAG}"
echo "  P3 marg_H + mask:   phase5_lr_fix_P3_aux_marg_H_${DATE_TAG}"
echo "  P4 ac_tiny + mask:  phase5_lr_fix_P4_aux_ac_tiny_${DATE_TAG}"
echo ""
echo "Acceptance criteria (vs baselines):"
echo "  P1 success:  val/mAP @ ep 60 >= 0.85   (target: closes 5pp gap to 448's 0.888)"
echo "               val/cam_iou_best     >= 0.24   (matches H4 baseline -- proves no regression)"
echo "  P2 success:  val/mAP              >= 0.83  AND val/cam_iou_best >= 0.27"
echo "  P3 success:  val/mAP              >= 0.83  AND val/cam_iou_best >= 0.28"
echo "               attn_mean trajectory < 0.95 throughout training (no collapse)"
echo "  P4 success:  val/mAP              >= 0.83"
echo "               attn_mean @ ep 80    < 0.90 (vs ~1.0 in §5.14.3 collapse runs)"
echo ""
echo "Post-run analysis: scripts/measure_rq4_attn_dynamics.py on each new"
echo "checkpoint; compare attention-collapse signatures against §5.13/§5.14 runs."
echo ""

# Exit non-zero if ANY phase we tried to run failed.
any_fail=0
for p in P1 P2 P3 P4; do
    if [[ "${RC[$p]}" != "-1" && "${RC[$p]}" != "0" ]]; then
        any_fail=1
    fi
done
if (( any_fail )); then
    echo "At least one phase failed -- inspect logs/phase5_lr_fix_<phase>_*.log."
    exit 1
fi
echo "All requested phases succeeded."
exit 0
