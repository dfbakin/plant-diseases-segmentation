#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

CHECKPOINT="${1:-outputs/weakclip/weakclip-voc/checkpoints/weakclip-epoch=10-val/mIoU=0.6361.ckpt}"

echo "=== Generate WeakCLIP Pseudo Masks (multi-scale + flip + slide) ==="
echo "Checkpoint: ${CHECKPOINT}"

python src/generate_weakclip_masks.py \
    "checkpoint='${CHECKPOINT}'" \
    class_names_file=/workspace/plant-diseases-segmentation/outputs/labels/class_names.txt \
    image_dir=/workspace/plant-diseases-segmentation/data/VOC2012/JPEGImages \
    image_ext=.jpg \
    names_file=/workspace/plant-diseases-segmentation/stripped_list_voc_ids.txt \
    output_dir=/workspace/plant-diseases-segmentation/outputs/weakclip_probs \
    scales="[0.5,0.75,1.0,1.25,1.5,1.75]" \
    flip=true \
    crop_size=512 \
    stride=341 \
    clip_pretrained=/workspace/plant-diseases-segmentation/pretrained/ViT-B-16.pt \
    image_size=512 \
    num_classes=21

echo "=== Probability maps saved to outputs/weakclip_probs ==="
