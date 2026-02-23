#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Apply CRF to CAMs (la + ha) ==="

python src/apply_crf.py \
    cam_dir=outputs/cams/cam_npy \
    image_dir=data/VOC2012/JPEGImages \
    image_ext=.jpg \
    la_crf_dir=outputs/cams/la_crf \
    ha_crf_dir=outputs/cams/ha_crf \
    bg_threshold=0.3 \
    la_alpha=4.0 \
    ha_alpha=32.0 \
    crf_iters=5 \
    num_workers=64

echo "=== CRF complete ==="
echo "Output: outputs/cams/la_crf/ and outputs/cams/ha_crf/"
