#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

PRED_DIR="${1:-outputs/pseudo_masks}"
GT_DIR="${2:-data/VOC2012/SegmentationClassAug}"

echo "=== Evaluate Pseudo Masks ==="
echo "Predictions: ${PRED_DIR}"
echo "Ground truth: ${GT_DIR}"

python src/evaluate_masks.py \
    pred_dir="${PRED_DIR}" \
    gt_dir="${GT_DIR}" \
    num_cls=21

echo "=== Evaluation complete ==="
