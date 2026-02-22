#!/bin/bash
set -e

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

MODE="${1:?Usage: $0 <mode> [extra args...]}"
shift

echo "=== Export Labels (mode=${MODE}) ==="

case "${MODE}" in
    voc)
        python src/export_labels.py \
            mode=voc_masks \
            voc_root=data/VOC2012 \
            split=train_aug_id \
            num_classes=20 \
            output=outputs/labels/voc_train_aug.npy \
            "$@"
        ;;
    plantseg)
        python src/export_labels.py \
            mode=plantseg \
            root=data/plantsegv3 \
            pv_split=train \
            output=outputs/labels/plantseg_train.npy \
            "$@"
        ;;
    plantvillage)
        python src/export_labels.py \
            mode=plantvillage \
            root=data/plant-village \
            pv_split=train \
            output=outputs/labels/plantvillage_train.npy \
            "$@"
        ;;
    *)
        echo "Unknown mode: ${MODE}. Choose from: voc, plantseg, plantvillage"
        exit 1
        ;;
esac

echo "=== Export complete ==="
