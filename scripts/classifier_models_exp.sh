set -e

EXPERIMENT_NAME="dfbakin_classifier_cam_benchmark"
SEED=42
MAX_EPOCHS=30
IMAGE_SIZE=384
BATCH_SIZE=16
GRAD_ACCUMULATION_STEPS=2

export MLFLOW_TRACKING_URI=null
# export MLFLOW_TRACKING_USERNAME="your_username"
# export MLFLOW_TRACKING_PASSWORD="your_password"

echo "[1/6] Training EfficientNet-B4..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=efficientnet_b4 \
    model.pretrained=true \
    model.learning_rate=1e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.precision="32" \
    trainer.max_epochs=${MAX_EPOCHS} \
    cam.enabled=true

echo "[2/6] Training ResNet18..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=resnet18 \
    model.pretrained=true \
    model.learning_rate=4e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    cam.enabled=true

echo "[3/6] Training ResNet50..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=resnet50 \
    model.pretrained=true \
    model.learning_rate=4e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    cam.enabled=true


echo "[4/6] Training EfficientNet-B0..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=efficientnet_b0 \
    model.pretrained=true \
    model.learning_rate=4e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    cam.enabled=true

echo "[5/6] Training EfficientNet-B2..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=efficientnet_b2 \
    model.pretrained=true \
    model.learning_rate=4e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    cam.enabled=true

echo "[6/6] Testing LayerCAM with EfficientNet-B4..."
python3 src/train_classifier.py \
    experiment_name="${EXPERIMENT_NAME}" \
    seed=${SEED} \
    model.name=efficientnet_b4 \
    model.pretrained=true \
    model.learning_rate=1e-4 \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=16 \
    data.num_workers=2 \
    trainer.accumulate_grad_batches=${GRAD_ACCUMULATION_STEPS} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    cam.enabled=true \
    cam.method=layercam
