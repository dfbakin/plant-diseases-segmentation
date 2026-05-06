#!/usr/bin/env bash
###############################################################################
# Phase 1 — Frozen seg-probes.
#
# Trains one ProbeHead per (ckpt, position) over PlantSeg with the host
# SPDNet completely frozen, then evaluates each probe (and three
# non-trainable baselines) with threshold sweep + per-distribution CRF
# tuning + visualizations.
#
# 11 runs total (5 token + 6 spatial; P6_attn_map only on spatial).
# Each run is skip-if-exists guarded -- safe to re-launch.
#
# Modes:
#   bash scripts/run_seg_probes_phase1.sh
#       Full Phase 1 -- ~14-16 h on a single 5090.
#       (11 probes x 20 epochs at ~94 s/epoch + ~10 min eval each on a
#        300-image val subset; see EVAL_FLAGS below.)
#       The 300-image subset (deterministic, seed=1234) is used for ranking
#       only -- the Phase 4 ceiling re-evaluates winners on full val.
#
#   SMOKE=1 bash scripts/run_seg_probes_phase1.sh
#       Tiny dataset, 1 epoch, 30-image CRF subset, 5-image viz.
#       Writes to outputs/spdnet_plantseg/_smoke/seg_probe_phase1/.
#       Should finish in ~5 min.
#
#   AUX_ONLY=1 CKPT_AUX=... AUX_TAG=... bash scripts/run_seg_probes_phase1.sh
#       Only runs the 6 aux-spatial probes on the AUX checkpoint. Skips
#       the token (5) and spatial-baseline (6) loops, and skips their
#       preflight file checks. Useful when the original baseline ckpts
#       have been pruned / aren't available locally.
#
#   AUX_ONLY=1 AUX_POSITIONS="P1_layer4 P3_query_merged P4_fused" \
#       CKPT_AUX=... AUX_TAG=... bash scripts/run_seg_probes_phase1.sh
#       Further narrows the aux leg to the positions named in
#       AUX_POSITIONS (space-separated). Used by the Phase 5 launch
#       guide to probe only the 3 most informative positions on
#       d4_ac_safe / highres-896 checkpoints, cutting the aux leg
#       from 6 probes to 3 (~3 h vs ~6 h on a single 5090).
###############################################################################

set -euo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

CKPT_TOKEN="${CKPT_TOKEN:-outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt}"
CKPT_SPATIAL="${CKPT_SPATIAL:-outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt}"

TOKEN_TAG="${TOKEN_TAG:-token_n1_heavy}"
SPATIAL_TAG="${SPATIAL_TAG:-spatial_n1_ps_pv}"

# OPTIONAL: 3rd checkpoint for the aux-loss-trained spatial model. When set,
# the script runs the 6 spatial positions on this checkpoint AFTER the
# baseline 11 probes -- the user uses this to rerun Phase 1 against the new
# `spdnet_spatial_eq_con` (or sibling) checkpoint without code edits.
#
#   CKPT_AUX=outputs/spdnet_plantseg/spdnet_aux_losses/.../<best>.ckpt \
#   AUX_TAG=spatial_eq_con \
#       bash scripts/run_seg_probes_phase1.sh
CKPT_AUX="${CKPT_AUX:-}"
AUX_TAG="${AUX_TAG:-spatial_aux_losses}"
AUX_ONLY="${AUX_ONLY:-0}"

ALL_POSITIONS=(P1_layer4 P2_fpn_p2 P3_query_merged P4_fused P5_cam_classifier P6_attn_map)
TOKEN_POSITIONS=(P1_layer4 P2_fpn_p2 P3_query_merged P4_fused P5_cam_classifier)
SPATIAL_POSITIONS=(P1_layer4 P2_fpn_p2 P3_query_merged P4_fused P5_cam_classifier P6_attn_map)

# AUX_POSITIONS override: when set, replaces the position list used for
# the aux-only leg. Accepts a space-separated list inside one string so
# the env var is easy to pass from the launch guide. Validated against
# ALL_POSITIONS below to catch typos before any probe training starts.
AUX_POSITIONS_RAW="${AUX_POSITIONS:-}"
if [[ -n "$AUX_POSITIONS_RAW" ]]; then
    read -r -a AUX_POSITIONS_ARR <<< "$AUX_POSITIONS_RAW"
    for pos in "${AUX_POSITIONS_ARR[@]}"; do
        found=0
        for valid in "${ALL_POSITIONS[@]}"; do
            if [[ "$pos" == "$valid" ]]; then found=1; break; fi
        done
        if [[ "$found" -eq 0 ]]; then
            echo "ERROR: AUX_POSITIONS contains unknown position '$pos'." >&2
            echo "       Valid: ${ALL_POSITIONS[*]}" >&2
            exit 1
        fi
    done
else
    AUX_POSITIONS_ARR=("${SPATIAL_POSITIONS[@]}")
fi

# ----------------------------------------------------------------------------
# Smoke vs full
# ----------------------------------------------------------------------------

SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
    OUT_ROOT="outputs/spdnet_plantseg/_smoke/seg_probe_phase1"
    EXTRA_TRAIN_OVERRIDES=(
        "data.limit_train=50"
        "data.limit_val=25"
        "trainer.max_epochs=1"
        "data.num_workers=0"
        "data.batch_size=4"
    )
    EVAL_FLAGS=(--smoke --crf-sweep-images 30 --viz-count 5 --crf-workers 4)
    echo "[phase1] SMOKE mode -- writing under $OUT_ROOT"
else
    OUT_ROOT="outputs/spdnet_plantseg/seg_probe_phase1"
    EXTRA_TRAIN_OVERRIDES=()
    # --cleanup-seeds drops the ~4.5 GB / probe of *_seeds/ npy files after
    # eval.json + viz/ are written. Seeds are fully reproducible from
    # head.pt + the source SPDNet ckpt; keeping them would push Phase 1
    # alone past 50 GB, well over the 41 GB free-disk budget.
    #
    # --limit-val 300 + --crf-sweep-images 50 are the *screen-mode* settings:
    # eval runs in ~10 min/baseline instead of ~65 min, ~5x faster, with
    # rankings stable on a 300-image subset (random sample, seed=1234,
    # same subset across all probes). Two probes from the previous launch
    # (P1_layer4 and P2_fpn_p2 under token_n1_heavy) were eval'd on full
    # val before this change and act as full-val reference points; they
    # are skipped via eval.json. The Phase 4 winner re-eval re-runs
    # without --limit-val for the final headline number.
    EVAL_FLAGS=(--crf-sweep-images 50 --viz-count 25 --crf-workers 8 --cleanup-seeds --limit-val 300)
fi

mkdir -p "$OUT_ROOT" logs

LOG_FILE="logs/seg_probe_phase1_$(date +%Y%m%d_%H%M%S).log"
DONE_MARKER="$OUT_ROOT/.DONE"

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------

echo "============================================================"
echo "  SPDNet Phase 1 — Frozen Probes"
echo "  Started:  $(date)"
echo "  Out root: $OUT_ROOT"
echo "  Logfile:  $LOG_FILE"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
free_gb=$(df -BG --output=avail "$OUT_ROOT" 2>/dev/null | tail -n1 | tr -dc '0-9' || echo 0)
echo "  Disk free: ${free_gb}G"
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
echo "  GPU free:  ${free_mib} MiB"
echo "============================================================"

if [[ -f "$DONE_MARKER" ]]; then
    echo "[phase1] $DONE_MARKER already exists -- nothing to do."
    exit 0
fi

if [[ "$AUX_ONLY" == "1" ]]; then
    if [[ -z "$CKPT_AUX" ]]; then
        echo "ERROR: AUX_ONLY=1 requires CKPT_AUX to be set." >&2
        exit 1
    fi
    echo "[phase1] AUX_ONLY=1 -- token/spatial baseline probes will be skipped."
else
    if [[ ! -f "$CKPT_TOKEN" ]]; then
        echo "ERROR: token checkpoint missing: $CKPT_TOKEN" >&2
        exit 1
    fi
    if [[ ! -f "$CKPT_SPATIAL" ]]; then
        echo "ERROR: spatial checkpoint missing: $CKPT_SPATIAL" >&2
        exit 1
    fi
fi
if [[ -n "$CKPT_AUX" && ! -f "$CKPT_AUX" ]]; then
    echo "ERROR: CKPT_AUX set but file missing: $CKPT_AUX" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Train + eval one (ckpt_tag, ckpt_path, position)
# ----------------------------------------------------------------------------

run_one_probe() {
    local tag="$1"; local ckpt="$2"; local pos="$3"

    local out_dir="$OUT_ROOT/$tag/$pos"
    local head_path="$out_dir/head.pt"
    local eval_path="$out_dir/eval.json"

    echo ""
    echo "=== [$tag/$pos] ============================================"
    echo "  out_dir: $out_dir"

    if [[ -f "$eval_path" ]]; then
        echo "  $eval_path exists -- skipping training and eval."
        return 0
    fi

    if [[ ! -f "$head_path" ]]; then
        echo "  Training probe head..."
        # Hydra needs the checkpoint value single-quoted because Lightning
        # ModelCheckpoint produced filenames with multiple "=" signs.
        python src/train_spdnet_probe.py \
            ckpt_tag="$tag" \
            "checkpoint='$ckpt'" \
            phase="phase1" \
            output_dir="$OUT_ROOT/\${ckpt_tag}/\${model.position}" \
            model.position="$pos" \
            model.freeze_backbone=true \
            model.seg_loss_weight=1.0 \
            model.cls_loss_weight=0.0 \
            "${EXTRA_TRAIN_OVERRIDES[@]}"
    else
        echo "  $head_path exists -- skipping training."
    fi

    echo "  Evaluating probe + baselines..."
    python scripts/eval_seg_probes.py \
        --probe-dir "$out_dir" \
        --checkpoint "$ckpt" \
        "${EVAL_FLAGS[@]}"
}

# ----------------------------------------------------------------------------
# Execute all 11 (or 5+6) probes sequentially
# ----------------------------------------------------------------------------

t0=$(date +%s)

if [[ "$AUX_ONLY" != "1" ]]; then
    echo ""
    echo "============================================================"
    echo "  Token checkpoint -- 5 positions"
    echo "============================================================"
    for pos in "${TOKEN_POSITIONS[@]}"; do
        run_one_probe "$TOKEN_TAG" "$CKPT_TOKEN" "$pos" 2>&1 | tee -a "$LOG_FILE"
    done

    echo ""
    echo "============================================================"
    echo "  Spatial checkpoint -- 6 positions"
    echo "============================================================"
    for pos in "${SPATIAL_POSITIONS[@]}"; do
        run_one_probe "$SPATIAL_TAG" "$CKPT_SPATIAL" "$pos" 2>&1 | tee -a "$LOG_FILE"
    done
fi

if [[ -n "$CKPT_AUX" ]]; then
    echo ""
    echo "============================================================"
    echo "  Aux-loss spatial checkpoint -- ${#AUX_POSITIONS_ARR[@]} positions  (tag=$AUX_TAG)"
    echo "  positions: ${AUX_POSITIONS_ARR[*]}"
    echo "  (set via CKPT_AUX=$CKPT_AUX)"
    echo "============================================================"
    for pos in "${AUX_POSITIONS_ARR[@]}"; do
        run_one_probe "$AUX_TAG" "$CKPT_AUX" "$pos" 2>&1 | tee -a "$LOG_FILE"
    done
fi

# ----------------------------------------------------------------------------
# Decision gate
# ----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Phase 1 decision gate"
echo "============================================================"
python scripts/seg_probe_decisions.py phase1 --root "$OUT_ROOT" 2>&1 | tee -a "$LOG_FILE"

touch "$DONE_MARKER"

t1=$(date +%s)
echo ""
echo "============================================================"
echo "  Phase 1 complete in $((t1 - t0))s -- $(date)"
echo "  Marker:  $DONE_MARKER"
echo "  Summary: $OUT_ROOT/SUMMARY.md"
echo "============================================================"
