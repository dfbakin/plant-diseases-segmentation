#!/usr/bin/env bash
###############################################################################
# Phase 5 5090 chain: post-P2' follow-up runner.
#
# Two phases that run after the in-flight P2' (cls + mask) finishes:
#
#   A) CRF baseline eval. Generates val CAMs from the P1' best-mAP and P2'
#      best-cam_iou checkpoints, then sweeps DenseCRF parameters
#      (srgb x bg_threshold x scale_factor) on a 200-image val subset.
#      The disease-IoU delta between P1' and P2' is the empirical answer
#      to "does AUC up actually translate to better CRF refinement?"
#      (the user's intuition we're checking).
#
#   B) P3' (warm_mask_eq). Warm-starts from P2' best_cam_iou.ckpt and
#      adds equivariance loss on top of the existing cls + mask combo.
#      Same DDP / rps=56 / eff_batch=32 setup as P2'. Targets:
#         val/cam_iou_best >= 0.30  (vs P2' ~0.27 ceiling)
#         val/cam_iou_auc  >= 0.24  (held or improved)
#         val/mAP          >= 0.83  (within 2 pp of P2' end)
#         attn_std         >=  attn_std @ P2' end (no equivariance-collapse)
#         train/L_eq       monotonic decrease (loss is actually optimised)
#
# Manual scheduling -- four ways to launch this AFTER the in-flight P2'
# finishes without sitting at the keyboard:
#
#   1) Polling background process (recommended; survives terminal close):
#        nohup bash scripts/run_phase5_5090_followup.sh --wait-for-p2 \
#          > logs/phase5_5090_chain/followup_$(date +%Y%m%d_%H%M).log 2>&1 &
#        disown
#
#   2) Foreground polling (one terminal, you watch it):
#        bash scripts/run_phase5_5090_followup.sh --wait-for-p2
#
#   3) `at` command (schedule for a specific time, e.g. ~7h from now):
#        echo "bash $PWD/scripts/run_phase5_5090_followup.sh" \
#          | at now + 7 hours
#
#   4) Marker-only chain (useful if you want to run in the same shell as
#      the parent chain script without modifying it):
#        bash scripts/run_phase5_5090_chain.sh \
#          && bash scripts/run_phase5_5090_followup.sh
#
# Idempotency: per-phase markers under outputs/_phase5_5090_chain/
# (CRF.DONE, P3.DONE). Remove a marker to force re-run.
#
# LR sanity (the user asked: is 1.2e-5 too low for warm-starts?):
#   * P2' empirical evidence: at lr_override=1.2e-5 + warmup=2 + 4-GPU
#     DDP eff_batch=32, P2' drove L_mask from 0.221 -> 0.025 (-89%) in
#     12 epochs while preserving the warm-started classifier (val/mAP
#     dipped 0.851 -> 0.807 -> 0.847; cls re-converged inside the same
#     run). cam_iou_best +15%, cam_iou_auc +36%. So 1.2e-5 was clearly
#     SUFFICIENT for joint cls + mask convergence on warm-start at this
#     hardware -- not too low.
#   * For P3' we keep the SAME lr_override=1.2e-5 because the model is
#     even more converged than at P2' start (cls AND mask are both
#     tuned), and AdamW per-parameter normalisation lets the fresh L_eq
#     gradient (concentrated on the SCA in_proj_weight) get its fair
#     share of step size at this LR. Going higher (e.g. 2e-5, 3e-5)
#     gains marginal headroom for L_eq at a real risk of perturbing the
#     converged classifier (mAP regression). The auto-scaled "fresh
#     classifier" LR at eff_batch=32 would be 5e-4 * 32/256 = 6.25e-5
#     (5x higher); that is the right number for FRESH training, the
#     wrong number for warm-start.
#   * Diagnostic: monitor train/L_eq for the first 5 epochs of P3'. If
#     it is not strictly decreasing by epoch 5, set ``LR_OVERRIDE_P3=
#     1.5e-5`` (a 25% bump) and re-launch. Don't increase lambda_eq
#     blindly: a stuck L_eq with a stable lambda_eq is an under-LR
#     signature; a destabilised mAP at a stable L_eq is an over-LR or
#     over-lambda_eq signature.
#
# Usage:
#   bash scripts/run_phase5_5090_followup.sh                    # CRF + P3'
#   bash scripts/run_phase5_5090_followup.sh --wait-for-p2      # poll then run
#   bash scripts/run_phase5_5090_followup.sh --skip-crf         # P3' only
#   bash scripts/run_phase5_5090_followup.sh --skip-p3          # CRF only
#   bash scripts/run_phase5_5090_followup.sh --preflight-only   # just verify
###############################################################################

set -uo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ----------------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------------

WAIT_FOR_P2=0
SKIP_CRF=0
SKIP_P3=0
PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-for-p2)    WAIT_FOR_P2=1; shift ;;
        --skip-crf)       SKIP_CRF=1; shift ;;
        --skip-p3)        SKIP_P3=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) sed -n '2,90p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. Try --help." >&2; exit 1 ;;
    esac
done

# ----------------------------------------------------------------------------
# Configurable knobs
# ----------------------------------------------------------------------------

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M)}"
EXPERIMENT="${EXPERIMENT:-phase5_5090_chain}"

# CRF eval: subsample the val set for speed. 500 images is enough to
# rank P1' vs P2' reliably (the std on disease_iou over a 200-image
# subset of val was <0.6 pp in prior eval campaigns). Set to 0 to use
# the full 4485-image val.
CRF_GEN_MAX_IMAGES="${CRF_GEN_MAX_IMAGES:-500}"
CRF_SWEEP_MAX_IMAGES="${CRF_SWEEP_MAX_IMAGES:-200}"
CRF_SWEEP_WORKERS="${CRF_SWEEP_WORKERS:-8}"

# P3' training: same shape as P2' (4-GPU DDP, eff_batch=32, rps=56).
IMAGE_SIZE="${IMAGE_SIZE:-896}"
INCLUDE_PV="${INCLUDE_PV:-true}"
LOG_EVERY="${LOG_EVERY:-50}"
AUGMENTATION="${AUGMENTATION:-heavy}"
NUM_REFS="${NUM_REFS:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
# P3' batch / accum. CRITICAL: kept at batch=1 + accum=8 (NOT
# batch=2 + accum=4 like P2') because L_eq's second forward through
# attention_map(q_aug, ref_merged_cached=...) adds ~4-5 GiB peak VRAM
# on top of the cls + mask graph. The 2026-05-09 smoke at batch=2 +
# rps=56 + image_size=896 + DDP=4 OOM'd on every rank (peak 28.7 GiB
# allocated, tried for another 4.69 GiB on the eq backward graph,
# total ~33 GiB / 32 GiB cap). The same setup at batch=1 peaked at
# 21.49 GiB on rank 0 with 9.5 GiB headroom -- safe. The eff_batch
# stays at 32 (1 * 8 * 4 = 32) so the experimental contract is
# preserved bit-for-bit; the only cost is slightly more per-step
# overhead (more Python+CUDA dispatch per optimizer step), at most a
# few percent on the per-epoch wall-clock since the same total number
# of training samples is consumed.
P3_BATCH="${P3_BATCH:-1}"
P3_ACCUM="${P3_ACCUM:-8}"
MAX_EPOCHS_P3="${MAX_EPOCHS_P3:-25}"
WARMUP_P3="${WARMUP_P3:-2}"
MIN_LR_P3="${MIN_LR_P3:-1e-7}"
LR_BASE="${LR_BASE:-0.0005}"
LR_OVERRIDE_P3="${LR_OVERRIDE_P3:-1.2e-5}"
REF_POOL_SIZE="${REF_POOL_SIZE:-56}"
ONLINE_LOC_EVAL_BS="${ONLINE_LOC_EVAL_BS:-2}"
LAMBDA_MASK_P3="${LAMBDA_MASK_P3:-0.05}"
LAMBDA_EQ_P3="${LAMBDA_EQ_P3:-0.1}"
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-ddp}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

OUT_BASE="outputs/${EXPERIMENT}"
MARKER_DIR="outputs/_${EXPERIMENT}"
LOG_DIR="logs/${EXPERIMENT}"
CRF_OUT_DIR="outputs/_phase5_followup/crf_eval_${DATE_TAG}"
mkdir -p "$OUT_BASE" "$MARKER_DIR" "$LOG_DIR" "$CRF_OUT_DIR"

# ----------------------------------------------------------------------------
# Wait for P2'.DONE marker if requested
# ----------------------------------------------------------------------------

P2_MARKER="${MARKER_DIR}/P2.DONE"
if (( WAIT_FOR_P2 )); then
    echo "================================================================"
    echo "  Waiting for P2' to finish: $P2_MARKER"
    echo "  (polling every 60s; safe to ^C and re-launch when ready)"
    echo "================================================================"
    sleep_step=60
    while [[ ! -f "$P2_MARKER" ]]; do
        if (( $(date +%s) % 600 < sleep_step )); then
            # Periodic heartbeat every ~10 min so the log isn't silent.
            echo "  [$(date +%H:%M:%S)] still waiting for $P2_MARKER..."
        fi
        sleep "$sleep_step"
    done
    echo "  P2 marker detected at $(date). Cooling down 120s for GPU memory to release..."
    sleep 120
fi

# ----------------------------------------------------------------------------
# Locate run directories and checkpoints
# ----------------------------------------------------------------------------

P1_RUN_DIR="${P1_RUN_DIR_OVERRIDE:-}"
P2_RUN_DIR="${P2_RUN_DIR_OVERRIDE:-}"
if [[ -z "$P1_RUN_DIR" ]]; then
    P1_RUN_DIR="$(ls -td outputs/${EXPERIMENT}/phase5_5090_P1_cls_only_rps${REF_POOL_SIZE}_* 2>/dev/null | head -n 1)"
fi
if [[ -z "$P2_RUN_DIR" ]]; then
    P2_RUN_DIR="$(ls -td outputs/${EXPERIMENT}/phase5_5090_P2_warm_mask_rps${REF_POOL_SIZE}_* 2>/dev/null | head -n 1)"
fi

if [[ -z "$P1_RUN_DIR" || ! -d "$P1_RUN_DIR" ]]; then
    echo "ERROR: could not auto-locate P1' run dir under outputs/${EXPERIMENT}/." >&2
    echo "       Set P1_RUN_DIR_OVERRIDE=/path/to/P1_run." >&2
    exit 5
fi
if [[ -z "$P2_RUN_DIR" || ! -d "$P2_RUN_DIR" ]]; then
    echo "ERROR: could not auto-locate P2' run dir under outputs/${EXPERIMENT}/." >&2
    echo "       Set P2_RUN_DIR_OVERRIDE=/path/to/P2_run." >&2
    exit 5
fi

# Pick the highest-mAP epoch ckpt from P1', else the deterministic
# best_cam_iou file. P1' creates "best_mAP_epoch{NN}.ckpt" with the
# raw epoch in the filename when the user committed the trajectory
# (the 2026-05-08 0711 run is the canonical P1').
P1_BEST_MAP_CKPT="${P1_BEST_MAP_OVERRIDE:-}"
if [[ -z "$P1_BEST_MAP_CKPT" ]]; then
    P1_BEST_MAP_CKPT="$(ls -1 "$P1_RUN_DIR"/checkpoints/best_mAP_epoch*.ckpt 2>/dev/null | sort -V | tail -n 1)"
fi
P2_BEST_CAM_CKPT="${P2_BEST_CAM_OVERRIDE:-${P2_RUN_DIR}/checkpoints/best_cam_iou.ckpt}"

if [[ ! -f "$P1_BEST_MAP_CKPT" ]]; then
    echo "ERROR: P1' best-mAP checkpoint not found at $P1_BEST_MAP_CKPT" >&2
    exit 6
fi
if [[ ! -f "$P2_BEST_CAM_CKPT" ]]; then
    echo "ERROR: P2' best-cam_iou checkpoint not found at $P2_BEST_CAM_CKPT" >&2
    echo "       (Did P2' actually finish? Check $P2_MARKER and the run log.)" >&2
    exit 6
fi

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Phase 5 5090 follow-up: CRF baseline eval + P3' (cls+mask+eq)"
echo "  Started:           $(date)"
echo "  DATE_TAG:          $DATE_TAG"
echo "  EXPERIMENT:        $EXPERIMENT"
echo "  P1' run:           $P1_RUN_DIR"
echo "  P1' best-mAP ckpt: $P1_BEST_MAP_CKPT"
echo "  P2' run:           $P2_RUN_DIR"
echo "  P2' best-cam ckpt: $P2_BEST_CAM_CKPT"
echo "  CRF subset:        gen=$CRF_GEN_MAX_IMAGES sweep=$CRF_SWEEP_MAX_IMAGES (workers=$CRF_SWEEP_WORKERS)"
echo "  P3' eff_batch:     $((DEVICES * P3_BATCH * P3_ACCUM)) (devices=$DEVICES, batch=$P3_BATCH, accum=$P3_ACCUM)"
echo "  P3' lr_override:   $LR_OVERRIDE_P3   (auto would be $(awk "BEGIN { printf \"%.6g\", $LR_BASE * $DEVICES * $P3_BATCH * $P3_ACCUM / 256.0 }"))"
echo "  P3' lambdas:       cls=1.0 mask=$LAMBDA_MASK_P3 eq=$LAMBDA_EQ_P3"
echo "  Skip CRF:          $SKIP_CRF"
echo "  Skip P3:           $SKIP_P3"
echo "  Preflight only:    $PREFLIGHT_ONLY"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | sed 's/^/                  /'
echo "================================================================"

# ----------------------------------------------------------------------------
# Pre-flight: same DDP-safety code asserts as the parent chain script,
# plus the auto-rps-detection patch + L_eq wiring guards new to P3'.
# ----------------------------------------------------------------------------

echo ""
echo "Pre-flight: verifying DDP-safety + L_eq wiring + cam_generator.rps auto-detect..."
python - <<'PY'
import sys, inspect

from src.conf.spdnet import SPDNetSpatialLossesConfig
from src.wsss.spdnet.lightning import SPDNetModule
from src.wsss.spdnet.cam_generator import load_spdnet_from_checkpoint

# L_eq config knobs are present.
assert hasattr(SPDNetSpatialLossesConfig, "lambda_eq"), \
    "Missing SPDNetSpatialLossesConfig.lambda_eq"
assert hasattr(SPDNetSpatialLossesConfig, "equivariance_transforms"), \
    "Missing SPDNetSpatialLossesConfig.equivariance_transforms"

# training_step branches on lambda_eq > 0 and computes equivariance_loss.
src = inspect.getsource(SPDNetModule.training_step)
assert "lambda_eq > 0" in src or "lambda_eq>0" in src, \
    "training_step does not branch on lambda_eq -- L_eq path may be dead"
assert "equivariance_loss(" in src, \
    "training_step does not call equivariance_loss(...)"
assert "self.model.attention_map(" in src, \
    "training_step does not run the second-pass attention_map for L_eq"

# cam_generator.load_spdnet_from_checkpoint propagates ref_pool_size.
# Without this, the rps=56 P1'/P2' ckpts would be loaded into a default
# rps=14 SPDNet and silently produce wrong CAMs in the CRF eval.
src_load = inspect.getsource(load_spdnet_from_checkpoint)
assert "ref_pool_size" in src_load, \
    "load_spdnet_from_checkpoint does not handle ref_pool_size at all"
assert "hp.get(\"ref_pool_size\"" in src_load \
    or "hp.get('ref_pool_size'" in src_load, \
    "load_spdnet_from_checkpoint does not auto-detect ref_pool_size " \
    "from checkpoint hyper_parameters; CRF eval on rps=56 ckpts would " \
    "silently use rps=14 and produce wrong CAMs."
assert "ref_pool_size=ref_pool_size" in src_load, \
    "load_spdnet_from_checkpoint does not pass ref_pool_size to SPDNet(...)"

print("pre-flight OK")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: pre-flight failed -- fix src/ before launching." >&2
    exit 2
fi

if (( PREFLIGHT_ONLY )); then
    echo ""
    echo "--preflight-only supplied -> exiting before dispatch."
    exit 0
fi

# ----------------------------------------------------------------------------
# Phase A: CRF baseline eval (P1' best-mAP vs P2' best-cam)
# ----------------------------------------------------------------------------

phase_crf_eval() {
    local marker="${MARKER_DIR}/CRF.DONE"
    if [[ -f "$marker" ]]; then
        echo "[CRF] marker exists -> skipping. (rm $marker to force re-run.)"
        return 0
    fi

    # The val label file is NOT in DVC -- only train labels are tracked
    # (plantseg_wsss_train.npy + plantseg_wsss_pv_all_train.npy). The val
    # split's image-level labels are derived from GT masks via
    # ``src/export_labels.py mode=plantseg_wsss pv_split=val`` which
    # takes ~15s on 1247 val images. We auto-generate it here if absent
    # so the user doesn't need to run it manually first.
    local label_file="outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy"
    if [[ ! -f "$label_file" ]]; then
        echo "[CRF] $label_file missing -> generating from GT masks via export_labels..."
        python -m src.export_labels mode=plantseg_wsss \
            root=data/plantsegv3 pv_split=val \
            "output=${label_file}" 2>&1 | tail -3
        if [[ ! -f "$label_file" ]]; then
            echo "[CRF] ERROR: failed to generate $label_file." >&2
            return 7
        fi
    fi

    # Build a subsampled label file for fast CAM generation. Stable
    # across runs at fixed seed so P1' and P2' see the same images.
    local subset_file="${CRF_OUT_DIR}/plantseg_wsss_val_subset.npy"
    if (( CRF_GEN_MAX_IMAGES > 0 )); then
        echo "[CRF] subsetting val labels to $CRF_GEN_MAX_IMAGES images at $subset_file"
        python - <<PY
import numpy as np, os
labels = np.load("${label_file}", allow_pickle=True).item()
gt_dir = "outputs/plantseg_binary_mc115/gt_binary_val"
have_gt = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(".png")}
keys = sorted(k for k in labels.keys() if k in have_gt)
rng = np.random.default_rng(1234)
n = min(${CRF_GEN_MAX_IMAGES}, len(keys))
chosen = rng.choice(keys, size=n, replace=False)
subset = {k: labels[k] for k in chosen}
np.save("${subset_file}", subset, allow_pickle=True)
print(f"  subset written: {len(subset)} images")
PY
        if [[ ! -f "$subset_file" ]]; then
            echo "[CRF] ERROR: failed to write subset label file." >&2
            return 7
        fi
    else
        subset_file="$label_file"
    fi

    # Generate single-scale CAMs at training resolution (896, rps=56,
    # auto-detected via load_spdnet_from_checkpoint patch). binary_
    # aggregate=max collapses per-class CAMs to {0: cam_HxW} which is
    # exactly the format sweep_crf_params expects for num_cls=2.
    local gpu_for_gen="${CRF_GEN_GPU:-0}"
    for tag_ckpt in "p1_best_map:${P1_BEST_MAP_CKPT}" "p2_best_cam:${P2_BEST_CAM_CKPT}"; do
        local tag="${tag_ckpt%%:*}"
        local ckpt="${tag_ckpt#*:}"
        local cam_dir="${CRF_OUT_DIR}/cams_${tag}"
        local sweep_json="${CRF_OUT_DIR}/sweep_${tag}.json"

        if [[ -d "$cam_dir" && -n "$(ls -A "$cam_dir" 2>/dev/null)" ]]; then
            echo "[CRF] $cam_dir already populated -- reusing."
        else
            mkdir -p "$cam_dir"
            echo ""
            echo "[CRF] generating CAMs for $tag (ckpt=$ckpt) on cuda:$gpu_for_gen"
            CUDA_VISIBLE_DEVICES="$gpu_for_gen" python -m src.generate_spdnet_cams \
                "checkpoint='${ckpt}'" \
                image_dir=data/plantsegv3/images/val \
                image_ext=.jpg \
                "label_file='${subset_file}'" \
                "output_dir='${cam_dir}'" \
                num_classes=115 \
                fpn_channels=256 \
                input_size="${IMAGE_SIZE}" \
                "scales=[1.0]" \
                num_ref_images="${NUM_REFS}" \
                binary_aggregate=max \
                seed_mode= \
                eval_threshold_sweep=false \
                || { echo "[CRF] gen failed for $tag" >&2; return 7; }
        fi

        echo ""
        echo "[CRF] sweeping CRF params for $tag -> $sweep_json"
        python scripts/sweep_crf_params.py \
            --seed_dir "$cam_dir" \
            --image_dir data/plantsegv3/images/val \
            --gt_dir outputs/plantseg_binary_mc115/gt_binary_val \
            --num_cls 2 \
            --max_images "$CRF_SWEEP_MAX_IMAGES" \
            --num_workers "$CRF_SWEEP_WORKERS" \
            --output_json "$sweep_json" \
            || { echo "[CRF] sweep failed for $tag" >&2; return 7; }
    done

    # Delta summary -- the actual answer to the user's intuition check.
    echo ""
    echo "================================================================"
    echo "  CRF baseline comparison (best config per ckpt):"
    echo "================================================================"
    python - <<PY
import json
p1 = json.load(open("${CRF_OUT_DIR}/sweep_p1_best_map.json"))["best"]
p2 = json.load(open("${CRF_OUT_DIR}/sweep_p2_best_cam.json"))["best"]
def fmt(label, r):
    return (f"  {label:<20} disease_iou={r['disease_iou']:6.2f}%  "
            f"bg_iou={r['bg_iou']:6.2f}%  mIoU={r['mIoU']:6.2f}%  "
            f"(srgb={r['srgb']:.0f} bg_thr={r['bg_threshold']:.2f} "
            f"sf={r['scale_factor']:.1f})")
print(fmt("P1' best-mAP:", p1))
print(fmt("P2' best-cam_iou:", p2))
delta = p2["disease_iou"] - p1["disease_iou"]
print(f"\n  Delta (P2 - P1): disease_iou={delta:+.2f}pp")
verdict = (
    "AUC-up DOES translate to better CRF refinement (intuition confirmed)."
    if delta >= 1.0 else
    "AUC-up does NOT translate to a meaningful CRF lift (intuition NOT "
    "confirmed; consider L_marg_H instead of L_eq for P3')."
    if delta <= 0.0 else
    "Marginal CRF lift (delta in noise band; check sweep std on full val)."
)
print(f"  Verdict: {verdict}\n")
PY
    touch "$marker"
    return 0
}

# ----------------------------------------------------------------------------
# Phase B: P3' = warm-start from P2' best_cam, cls + mask + eq
# ----------------------------------------------------------------------------

P3_NAME="phase5_5090_P3_warm_mask_eq_rps${REF_POOL_SIZE}_${DATE_TAG}"

phase_p3_warm_mask_eq() {
    local marker="${MARKER_DIR}/P3.DONE"
    if [[ -f "$marker" ]]; then
        echo "[P3] marker exists -> skipping. (rm $marker to force re-run.)"
        return 0
    fi

    local log_path="${LOG_DIR}/P3_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "================================================================"
    echo "  [P3]  start"
    echo "  log:        $log_path"
    echo "  marker:     $marker"
    echo "  warm-start: $P2_BEST_CAM_CKPT"
    echo "  started:    $(date)"
    echo "================================================================"

    python -m src.train_spdnet \
        run_name="${P3_NAME}" \
        experiment_name="${EXPERIMENT}" \
        +checkpoint="${P2_BEST_CAM_CKPT}" \
        model.fusion_mode=spatial \
        model.input_size="${IMAGE_SIZE}" \
        model.learning_rate="${LR_BASE}" \
        model.learning_rate_override="${LR_OVERRIDE_P3}" \
        model.ref_pool_size="${REF_POOL_SIZE}" \
        trainer.max_epochs="${MAX_EPOCHS_P3}" \
        trainer.log_every_n_steps="${LOG_EVERY}" \
        trainer.accumulate_grad_batches="${P3_ACCUM}" \
        trainer.precision=bf16-mixed \
        trainer.warmup_epochs="${WARMUP_P3}" \
        trainer.min_lr="${MIN_LR_P3}" \
        trainer.devices="${DEVICES}" \
        trainer.strategy="${STRATEGY}" \
        trainer.find_unused_parameters=true \
        trainer.sync_batchnorm=true \
        data.image_size="${IMAGE_SIZE}" \
        data.batch_size="${P3_BATCH}" \
        data.num_references="${NUM_REFS}" \
        data.augmentation="${AUGMENTATION}" \
        data.include_plantvillage="${INCLUDE_PV}" \
        data.num_workers="${NUM_WORKERS}" \
        losses.log_attn_stats=true \
        losses.online_loc_eval_enabled=true \
        losses.online_loc_eval_batch_size="${ONLINE_LOC_EVAL_BS}" \
        losses.lambda_mask="${LAMBDA_MASK_P3}" \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.50 \
        losses.mask_combiner=union \
        losses.mask_use_intersection=null \
        losses.mask_warmup_start_epoch=0 \
        losses.mask_warmup_epochs=0 \
        losses.lambda_eq="${LAMBDA_EQ_P3}" \
        "losses.equivariance_transforms=[1,2,3,4]" \
        losses.lambda_con=0 \
        losses.lambda_distill=0 \
        losses.lambda_ac=0 \
        losses.lambda_marg_H=0 \
        2>&1 | tee "$log_path"
    local rc=${PIPESTATUS[0]}
    echo ""
    echo "[P3] done (rc=$rc) at $(date)"
    if (( rc == 0 )); then
        touch "$marker"
    else
        echo "[P3] FAILED -- see $log_path; not writing $marker."
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
RC_CRF=-1
RC_P3=-1

if (( ! SKIP_CRF )); then
    phase_crf_eval
    RC_CRF=$?
fi

if (( ! SKIP_P3 )); then
    if (( RC_CRF != -1 && RC_CRF != 0 )); then
        echo "[P3] WARNING: CRF eval failed (rc=$RC_CRF); proceeding with P3' anyway."
        echo "       (CRF eval is diagnostic only -- not a P3' prerequisite.)"
    fi
    phase_p3_warm_mask_eq
    RC_P3=$?
fi

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Phase 5 5090 follow-up: wall clock $((t1 - t0))s ($(((t1 - t0) / 3600))h)"
echo "================================================================"
case "$RC_CRF" in
    -1) printf "  %-3s  SKIPPED\n" "CRF" ;;
    0)  printf "  %-3s  rc=0  (results: ${CRF_OUT_DIR}/sweep_*.json)\n" "CRF" ;;
    *)  printf "  %-3s  rc=%d (FAILED)\n" "CRF" "$RC_CRF" ;;
esac
case "$RC_P3" in
    -1) printf "  %-3s  SKIPPED\n" "P3" ;;
    0)  printf "  %-3s  rc=0  (run: ${P3_NAME})\n" "P3" ;;
    *)  printf "  %-3s  rc=%d (FAILED)\n" "P3" "$RC_P3" ;;
esac

echo ""
echo "Acceptance criteria (P3'):"
echo "  val/cam_iou_best  >= 0.30   (vs P2' ~0.27 ceiling)"
echo "  val/cam_iou_auc   >= 0.24   (held or improved vs P2' ~0.22)"
echo "  val/mAP           >= 0.83   (within 2 pp of P2' end)"
echo "  attn_std          >= attn_std @ P2' end  (no eq-collapse)"
echo "  train/L_eq        monotonic decrease for first 5 epochs"
echo ""

any_fail=0
[[ "$RC_P3" != "-1" && "$RC_P3" != "0" ]] && any_fail=1
[[ "$RC_CRF" != "-1" && "$RC_CRF" != "0" ]] && any_fail=1
if (( any_fail )); then
    echo "At least one phase failed -- inspect logs."
    exit 1
fi
echo "All requested phases succeeded."
exit 0
