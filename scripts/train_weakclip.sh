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
    batch_size=16 \
    max_epochs=31 \
    learning_rate=1e-4 \
    min_lr=1e-6 \
    warmup_iters=1500 \
    weight_decay=3e-5 \
    identity_loss_weight=0.4 \
    use_crf_loss=true \
    crf_iters=10 \
    norm_eval=true \
    num_workers=16 \
    precision="32" \
    experiment_name=weakclip-voc

echo "=== WeakCLIP training complete ==="
