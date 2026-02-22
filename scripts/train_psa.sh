#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

echo "=== Train PSA Affinity Network ==="

python src/train_psa.py \
    voc_root=data/VOC2012 \
    la_crf_dir=outputs/cams/la_crf \
    ha_crf_dir=outputs/cams/ha_crf \
    backbone_weights=pretrained/res38_cls.pth \
    output_path=outputs/psa/psa_aff.pth \
    batch_size=8 \
    max_epochs=5 \
    lr=0.01 \
    num_workers=8 \
    cropsize=448

echo "=== PSA training complete ==="
echo "Output: outputs/psa/psa_aff.pth"
