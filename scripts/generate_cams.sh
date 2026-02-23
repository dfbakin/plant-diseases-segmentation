#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [label_file]}"
LABEL_FILE="${2:-outputs/labels/voc_train_aug.npy}"

echo "=== CAM Generation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Labels: ${LABEL_FILE}"

python src/generate_cams.py \
    "checkpoint='${CHECKPOINT}'" \
    image_dir=data/VOC2012/JPEGImages \
    image_ext=.jpg \
    "label_file=${LABEL_FILE}" \
    output_dir=outputs/cams/cam_npy \
    num_classes=20 \
    input_size=448 \
    "scales=[1.0,0.75,1.25]" \
    n_layers=3 \
    attention_type=fused \
    patch_attn_refine=true \
    gt_dir=data/VOC2012/SegmentationClassAug \
    eval_threshold_sweep=false

echo "=== CAM generation complete ==="
echo "Output: outputs/cams/cam_npy/"
