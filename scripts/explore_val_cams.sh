#!/bin/bash
# Exploratory script: generate val CAMs from MC115 MCTformer, evaluate, visualize.
# Temporary / exploration only — does not modify the main pipeline.
#
# Usage:
#   ./scripts/explore_val_cams.sh           # full pipeline
#   SKIP_GEN=1 ./scripts/explore_val_cams.sh  # skip CAM generation, run eval + vis only
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

CKPT="outputs/mctformer_plantseg_multiclass/2026-03-08_11-32-35/checkpoints/last.ckpt"
OUT_BASE="outputs/plantseg_binary_mc115"
LABEL_FILE="${OUT_BASE}/labels/plantseg_wsss_val.npy"
CAM_DIR="${OUT_BASE}/cams/cam_npy_val"
GT_DIR="${OUT_BASE}/gt_binary_val"
VIS_DIR="outputs/visualizations/val_cam_exploration"

NUM_CLASSES=115
INPUT_SIZE=512
NUM_VIS_IMAGES=25

echo "=== Step 1: Export val labels ==="
if [ ! -f "$LABEL_FILE" ]; then
    python src/export_labels.py \
        mode=plantseg_wsss \
        root=data/plantsegv3 \
        pv_split=val \
        "output=${LABEL_FILE}"
else
    echo "Val labels already exist: ${LABEL_FILE}"
fi

echo ""
echo "=== Step 2: Generate val CAMs (binary-aggregated from ${NUM_CLASSES}-class MCTformer) ==="
if [ -z "${SKIP_GEN:-}" ]; then
    python src/generate_cams.py \
        "checkpoint='${CKPT}'" \
        image_dir=data/plantsegv3/images/val \
        image_ext=.jpg \
        "label_file=${LABEL_FILE}" \
        "output_dir=${CAM_DIR}" \
        "num_classes=${NUM_CLASSES}" \
        "input_size=${INPUT_SIZE}" \
        "scales=[1.0,0.75,1.25]" \
        n_layers=3 \
        attention_type=fused \
        patch_attn_refine=true \
        binary_aggregate=max \
        "gt_dir=${GT_DIR}" \
        eval_threshold_sweep=true \
        eval_sweep_samples=0 \
        eval_optimize_metric=disease_iou
else
    echo "SKIP_GEN set, skipping CAM generation"
fi

echo ""
echo "=== Step 3: Comprehensive activation visualization ==="
python scripts/visualize_cam_activations.py \
    --checkpoint "${CKPT}" \
    --image_dir data/plantsegv3/images/val \
    --gt_dir "${GT_DIR}" \
    --cam_dir "${CAM_DIR}" \
    --output_dir "${VIS_DIR}" \
    --num_images "${NUM_VIS_IMAGES}" \
    --num_classes "${NUM_CLASSES}" \
    --input_size "${INPUT_SIZE}" \
    --seed 42

echo ""
echo "=== Done ==="
echo "CAMs:           ${CAM_DIR}/"
echo "Visualizations: ${VIS_DIR}/"
