#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Train WeakCLIP on Pseudo Masks ==="

python src/train_weakclip.py \
    voc_root=data/VOC2012 \
    pseudo_mask_dir=outputs/pseudo_masks \
    clip_pretrained=pretrained/ViT-B-16.pt \
    num_classes=21 \
    image_size=512 \
    batch_size=8 \
    max_epochs=20 \
    learning_rate=1e-4 \
    num_workers=8 \
    precision="16-mixed" \
    experiment_name=weakclip_voc

echo "=== WeakCLIP training complete ==="
