#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Random Walk Refinement ==="

python src/run_random_walk.py \
    cam_dir=outputs/cams/cam_npy \
    aff_checkpoint=outputs/psa/psa_aff.pth \
    output_dir=outputs/pseudo_masks \
    voc_root=data/VOC2012 \
    split_file=data/VOC2012/ImageSets/Segmentation/train_aug_id.txt \
    bg_threshold=0.3 \
    beta=8 \
    logt=6

echo "=== Random walk complete ==="
echo "Output: outputs/pseudo_masks/"
