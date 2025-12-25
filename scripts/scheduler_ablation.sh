set -e

export MLFLOW_TRACKING_URI=null
export MLFLOW_TRACKING_USERNAME="username"
export MLFLOW_TRACKING_PASSWORD="password"

EXPERIMENT_NAME="dfbakin-plantseg-scheduler-ablation"
SEED=42
MAX_EPOCHS=40
IMAGE_SIZE=384
BATCH_SIZE=16
ACCUM_GRAD=2
LR=5e-4

echo "[1/6] Training with constant LR..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=constant \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[2/6] Training with cosine annealing..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=cosine \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[3/6] Training with step LR..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=step \
    scheduler.step_size=10 \
    scheduler.gamma=0.1 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[4/6] Training with cyclic LR (triangular2)..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=cyclic \
    scheduler.step_size_up=10 \
    scheduler.mode=triangular2 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[5/6] Training with one-cycle LR..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=one_cycle \
    scheduler.max_lr_factor=10.0 \
    scheduler.pct_start=0.3 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[6/6] Training with reduce-on-plateau..."
python3 src/train.py \
    experiment.name="${EXPERIMENT_NAME}" \
    experiment.seed=${SEED} \
    model=segnext \
    model.variant=base \
    model.learning_rate=${LR} \
    augmentation=spatial_color_light \
    data.image_size=${IMAGE_SIZE} \
    data.batch_size=${BATCH_SIZE} \
    data.multiclass=true \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.precision="32" \
    trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    scheduler=plateau \
    scheduler.patience=5 \
    scheduler.factor=0.5 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}
