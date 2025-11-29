#!/bin/bash
# Benchmark all segmentation models on PlantSeg dataset
#
# Usage:
#   ./scripts/benchmark_models.sh
#
# This runs training for DeepLabv3+, SegFormer, and SegNeXt sequentially
# with consistent settings for fair comparison.

set -e

# Experiment settings
EXPERIMENT_NAME="plantseg_benchmark"
SEED=42
MAX_EPOCHS=1

echo "=== PlantSeg Model Benchmark ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo ""

# DeepLabv3+ with ResNet50
echo "[1/4] Training DeepLabv3+ (ResNet50)..."
python src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=deeplabv3plus \
    model.encoder_name=resnet50 \
    trainer.max_epochs=${MAX_EPOCHS}

# U-Net with ResNet50
echo "[2/4] Training U-Net (ResNet50)..."
python src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=unet \
    model.encoder_name=resnet50 \
    trainer.max_epochs=${MAX_EPOCHS}

# SegFormer B3
echo "[3/4] Training SegFormer (B3)..."
python src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segformer \
    model.variant=b3 \
    trainer.max_epochs=${MAX_EPOCHS}

# SegNeXt Base
echo "[4/4] Training SegNeXt (Base)..."
python src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    trainer.max_epochs=${MAX_EPOCHS}

echo ""
echo "=== Benchmark Complete ==="
echo "Results logged to MLflow. View with: mlflow ui"




