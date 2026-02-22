#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

PRED_DIR="${1:-outputs/pseudo_masks}"
SPLIT="${2:-train_aug_id}"

echo "=== Evaluate Pseudo Masks ==="
echo "Predictions: ${PRED_DIR}"
echo "Split: ${SPLIT}"

python src/evaluate_masks.py \
    pred_dir="${PRED_DIR}" \
    gt_dir=data/VOC2012/SegmentationClassAug \
    split_file="data/VOC2012/ImageSets/Segmentation/${SPLIT}.txt" \
    num_cls=21

echo "=== Evaluation complete ==="
