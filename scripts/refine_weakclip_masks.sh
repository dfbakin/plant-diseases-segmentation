#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Refine WeakCLIP Masks (CRF + Label Filtering) ==="

python src/refine_weakclip_masks.py \
    prob_dir=outputs/weakclip_probs \
    image_dir=data/VOC2012/JPEGImages \
    image_ext=.jpg \
    labels_file=outputs/labels/voc_train_aug.npy \
    output_dir=outputs/weakclip_masks \
    num_classes=21 \
    crf_t=10 \
    crf_sxy_gauss=3.0 \
    crf_compat_gauss=3.0 \
    crf_sxy_bilat=83.0 \
    crf_srgb_bilat=5.0 \
    crf_compat_bilat=3.0 \
    n_jobs=32

echo "=== Refined masks saved to outputs/weakclip_masks ==="
