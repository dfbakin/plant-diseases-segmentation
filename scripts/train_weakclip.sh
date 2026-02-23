#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Train WeakCLIP on Pseudo Masks ==="

python src/train_weakclip.py \
    class_names_file=outputs/labels/class_names.txt \
    train_image_dir=data/VOC2012/JPEGImages \
    train_mask_dir=outputs/pseudo_masks \
    val_image_dir=data/VOC2012/JPEGImages \
    val_mask_dir=data/VOC2012/SegmentationClassAug \
    val_names_file=data/VOC2012/ImageSets/Segmentation/val.txt \
    image_ext=.jpg \
    clip_pretrained=pretrained/ViT-B-16.pt \
    image_size=512 \
    batch_size=32 \
    max_epochs=80 \
    learning_rate=4e-4 \
    poly_power=0.9 \
    num_workers=8 \
    precision="32" \
    experiment_name=weakclip_voc_scheduler

echo "=== WeakCLIP training complete ==="
