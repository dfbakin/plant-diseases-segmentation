#!/bin/bash
set -e

# MCTformer-V2 multi-label classification on PlantSeg
#
# Uses DISEASE_CLASSES (115 foreground diseases) so that CAM indices
# match the PlantSeg GT annotation class system directly.
#
# After training, run the full pipeline with scripts/run_plantseg_pipeline.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

EXPERIMENT_NAME="mctformer_plantseg"
SEED=0
IMAGE_SIZE=448
BATCH_SIZE=32
MAX_EPOCHS=45
NUM_WORKERS=8

export MLFLOW_TRACKING_URI=null

echo "=== MCTformer-V2 Training on PlantSeg ==="
echo "Classes: 115 foreground diseases (DISEASE_CLASSES)"
echo "Image size: ${IMAGE_SIZE}, Batch size: ${BATCH_SIZE}, Epochs: ${MAX_EPOCHS}"
echo ""

python src/train_mctformer.py \
    dataset=plantseg \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=mctformer_v2 \
    model.pretrained=true \
    model.learning_rate=5e-4 \
    model.weight_decay=0.05 \
    model.drop_path_rate=0.1 \
    model.label_smoothing=0.1 \
    model.input_size=${IMAGE_SIZE} \
    plantseg_data.root=data/plantsegv3 \
    plantseg_data.train_split=train \
    plantseg_data.val_split=val \
    plantseg_data.image_size=${IMAGE_SIZE} \
    plantseg_data.batch_size=${BATCH_SIZE} \
    plantseg_data.num_workers=${NUM_WORKERS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="16-mixed"

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to outputs/${EXPERIMENT_NAME}/"
echo "Next step: run full pipeline with scripts/run_plantseg_pipeline.sh"
