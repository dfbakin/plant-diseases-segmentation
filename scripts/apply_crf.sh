#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Apply CRF to CAMs (la + ha) ==="
echo "Using PSA scale_factor parameterization: la_sf=1 (sxy=80), ha_sf=12 (sxy≈6.67), t=10"

python src/apply_crf.py \
    cam_dir=outputs/cams/cam_npy \
    image_dir=data/VOC2012/JPEGImages \
    image_ext=.jpg \
    la_crf_dir=outputs/cams/la_crf \
    ha_crf_dir=outputs/cams/ha_crf \
    bg_threshold=0.3 \
    la_scale_factor=1.0 \
    ha_scale_factor=12.0 \
    crf_iters=10 \
    num_workers=64

echo "=== CRF complete ==="
echo "Output: outputs/cams/la_crf/ and outputs/cams/ha_crf/"
