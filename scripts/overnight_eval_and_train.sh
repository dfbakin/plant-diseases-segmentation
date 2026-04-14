#!/bin/bash
set -e

# ============================================================================
# Overnight SPDNet evaluation & training script
# Estimated total time: ~8-8.5 hours on RTX 5090
#
# Phase 1: CAM generation + threshold sweep  (~2.8h)
#   - 3 checkpoints × 2 modes = 6 CAM generation runs
#   - N=1 runs: ~15 min each, N=3 runs: ~35 min each
#   - Each run includes threshold sweep with IoU evaluation
#
# Phase 2: Visualization grids               (~0.2h)
#   - 25-image 8-panel grids for 3 checkpoints
#
# Phase 3: Training experiments              (~5.4h)
#   - Run 5: spdnet_fix_n3_light    (N=3, light aug, 80ep)  ~2.7h
#   - Run 6: spdnet_fix_n3_minimal  (N=3, minimal aug, 80ep) ~2.7h
#
# ============================================================================

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

LOG_FILE="outputs/overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  Overnight SPDNet pipeline started at $(date)"
echo "  Log: $LOG_FILE"
echo "================================================================"

# --- Paths ---
LABEL_FILE="outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy"
IMAGE_DIR="data/plantsegv3/images/val"
GT_DIR="outputs/plantseg_binary_mc115/gt_binary_val"

N1_BEST="outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"
N3_BEST="outputs/spdnet_plantseg/spdnet_fix_n3_heavy/checkpoints/best.ckpt"
N3_LAST="outputs/spdnet_plantseg/spdnet_fix_n3_heavy/checkpoints/last.ckpt"

# ============================================================================
# PHASE 1: CAM generation + threshold sweep
# ============================================================================

generate_cams() {
    local tag="$1" ckpt="$2" nrefs="$3" mode="$4"
    local out_dir="outputs/spdnet_plantseg/cams/${tag}_${mode}"

    echo ""
    echo "--- CAM generation: ${tag} / ${mode} (N=${nrefs}) ---"
    echo "  checkpoint: ${ckpt}"
    echo "  output: ${out_dir}"
    echo "  started: $(date)"

    python src/generate_spdnet_cams.py \
        "checkpoint=${ckpt}" \
        "output_dir=${out_dir}" \
        "binary_aggregate=${mode}" \
        "num_ref_images=${nrefs}" \
        "label_file=${LABEL_FILE}" \
        "image_dir=${IMAGE_DIR}" \
        "gt_dir=${GT_DIR}" \
        eval_threshold_sweep=true \
        eval_optimize_metric=disease_iou

    echo "  finished: $(date)"
}

echo ""
echo "================================================================"
echo "  PHASE 1: CAM generation + threshold sweep"
echo "  6 runs: 3 checkpoints × 2 modes (max, top_energy)"
echo "  Estimated: ~2.8 hours"
echo "================================================================"

# n1_best: N=1 (fast, ~15 min each)
generate_cams "n1_best" "$N1_BEST" 1 max
generate_cams "n1_best" "$N1_BEST" 1 top_energy

# n3_best: N=3 (~35 min each)
generate_cams "n3_best" "$N3_BEST" 3 max
generate_cams "n3_best" "$N3_BEST" 3 top_energy

# n3_last: N=3 (~35 min each)
generate_cams "n3_last" "$N3_LAST" 3 max
generate_cams "n3_last" "$N3_LAST" 3 top_energy

echo ""
echo "  Phase 1 complete at $(date)"

# ============================================================================
# PHASE 2: Visualization grids
# ============================================================================

generate_viz() {
    local tag="$1" ckpt="$2" cam_dir="$3"
    local out_dir="outputs/visualizations/spdnet_${tag}"

    echo ""
    echo "--- Visualization: ${tag} ---"
    echo "  started: $(date)"

    python scripts/visualize_spdnet_activations.py \
        --checkpoint "$ckpt" \
        --image_dir "$IMAGE_DIR" \
        --gt_dir "$GT_DIR" \
        --cam_dir "$cam_dir" \
        --label_file "$LABEL_FILE" \
        --output_dir "$out_dir" \
        --num_images 25 \
        --seed 42

    echo "  finished: $(date)"
}

echo ""
echo "================================================================"
echo "  PHASE 2: Visualization (25 images × 8 panels × 3 checkpoints)"
echo "  Estimated: ~15 minutes"
echo "================================================================"

generate_viz "n1_best" "$N1_BEST" "outputs/spdnet_plantseg/cams/n1_best_max"
generate_viz "n3_best" "$N3_BEST" "outputs/spdnet_plantseg/cams/n3_best_max"
generate_viz "n3_last" "$N3_LAST" "outputs/spdnet_plantseg/cams/n3_last_max"

echo ""
echo "  Phase 2 complete at $(date)"

# ============================================================================
# PHASE 3: Training experiments (augmentation ablation)
# ============================================================================

echo ""
echo "================================================================"
echo "  PHASE 3: Training experiments"
echo "  Run 5: spdnet_fix_n3_light   (N=3, light aug, 80 epochs)"
echo "  Run 6: spdnet_fix_n3_minimal (N=3, minimal aug, 80 epochs)"
echo "  Estimated: ~5.4 hours"
echo "================================================================"

echo ""
echo "--- Training: spdnet_fix_n3_light ---"
echo "  started: $(date)"

python src/train_spdnet.py \
    run_name=spdnet_fix_n3_light \
    trainer.max_epochs=80 \
    trainer.log_every_n_steps=200 \
    trainer.accumulate_grad_batches=2 \
    data.batch_size=16 \
    data.num_references=3 \
    data.augmentation=light \
    data.num_workers=8

echo "  spdnet_fix_n3_light finished: $(date)"

echo ""
echo "--- Training: spdnet_fix_n3_minimal ---"
echo "  started: $(date)"

python src/train_spdnet.py \
    run_name=spdnet_fix_n3_minimal \
    trainer.max_epochs=80 \
    trainer.log_every_n_steps=200 \
    trainer.accumulate_grad_batches=2 \
    data.batch_size=16 \
    data.num_references=3 \
    data.augmentation=minimal \
    data.num_workers=8

echo "  spdnet_fix_n3_minimal finished: $(date)"

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "================================================================"
echo "  ALL PHASES COMPLETE at $(date)"
echo "  Log: $LOG_FILE"
echo ""
echo "  Outputs:"
echo "    CAMs:   outputs/spdnet_plantseg/cams/{n1_best,n3_best,n3_last}_{max,top_energy}/"
echo "    Viz:    outputs/visualizations/spdnet_{n1_best,n3_best,n3_last}/"
echo "    Models: outputs/spdnet_plantseg/spdnet_fix_n3_{light,minimal}/"
echo "================================================================"
