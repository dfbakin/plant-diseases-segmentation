#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Train DeepLab-v3+ on Pseudo Masks ==="

python src/train_deeplab_wsss.py \
    train_image_dir=data/VOC2012/JPEGImages \
    train_mask_dir=outputs/weakclip_masks \
    val_image_dir=data/VOC2012/JPEGImages \
    val_mask_dir=data/VOC2012/SegmentationClassAug \
    val_names_file=data/VOC2012/ImageSets/Segmentation/val.txt \
    image_ext=.jpg \
    encoder_name=resnet101 \
    num_classes=21 \
    ignore_index=255 \
    image_size=512 \
    batch_size=32 \
    max_epochs=40 \
    learning_rate=5e-4 \
    weight_decay=5e-4 \
    num_workers=8 \
    precision="16-mixed" \
    experiment_name=deeplab_wsss_voc

echo "=== DeepLab WSSS training complete ==="
