#!/bin/bash
set -e

# MCTformer-V2 multi-label classification on PASCAL VOC 2012
#
# Matches original MCTformer hyperparameters:
#   - input_size=448, batch_size=32, weight_decay=0.05
#   - LR=5e-4 (scaled by batch/512 in code), 5 warmup epochs
#   - drop_path=0.1, RandAugment+RandomErasing via timm
#
# After training, run CAM generation with src/generate_cams.py (TODO)

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

EXPERIMENT_NAME="mctformer_voc_v2"
SEED=0
IMAGE_SIZE=448
BATCH_SIZE=32
MAX_EPOCHS=60
NUM_WORKERS=8

export MLFLOW_TRACKING_URI=null

echo "=== MCTformer-V2 Training on VOC 2012 ==="
echo "Train split: train_aug_id (10582 images)"
echo "Val split: val (1449 images)"
echo "Image size: ${IMAGE_SIZE}, Batch size: ${BATCH_SIZE}, Epochs: ${MAX_EPOCHS}"
echo "Weight decay: 0.05, Drop path: 0.1, Warmup: 5 epochs"
echo ""

python src/train_mctformer.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=mctformer_v2 \
    model.pretrained=true \
    model.learning_rate=5e-4 \
    model.weight_decay=0.05 \
    model.drop_path_rate=0.1 \
    model.label_smoothing=0.1 \
    model.input_size=${IMAGE_SIZE} \
    data.root=data/VOC2012 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=${NUM_WORKERS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="16-mixed"

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to outputs/${EXPERIMENT_NAME}/"
echo "Next step: generate CAMs with src/generate_cams.py"
