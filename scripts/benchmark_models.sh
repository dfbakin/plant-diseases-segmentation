set -e

# Experiment settings
EXPERIMENT_NAME="plantseg_architecture_benchmark"
SEED=42
MAX_EPOCHS=20

echo "=== PlantSeg Model Benchmark ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo ""

# DeepLabv3+ with ResNet50
echo "[1/5] Training DeepLabv3+ (ResNet50)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=deeplabv3plus \
    model.encoder_name=resnet50 \
    data.image_size=384 \
    trainer.max_epochs=${MAX_EPOCHS}

# U-Net with ResNet50
echo "[2/5] Training U-Net (ResNet50)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=unet \
    model.encoder_name=resnet50 \
    data.image_size=384 \
    trainer.max_epochs=${MAX_EPOCHS}

# SegFormer B3
echo "[3/5] Training SegFormer (B3)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segformer \
    model.variant=b3 \
    data.image_size=384 \
    trainer.max_epochs=${MAX_EPOCHS}

# SegNeXt Large
echo "[4/5] Training SegNeXt (Large)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=large \
    model.learning_rate=3e-4 \
    data.image_size=384 \
    trainer.precision="32" \
    trainer.max_epochs=30

# SegNeXt Base
echo "[5/5] Training SegNeXt (Base)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=3e-4 \
    data.image_size=384 \
    trainer.precision="32" \
    trainer.max_epochs=30

echo ""
echo "=== Benchmark Complete ==="
