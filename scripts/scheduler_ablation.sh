set -e

export MLFLOW_TRACKING_URI="mlruns"
# export MLFLOW_TRACKING_USERNAME="username"
# export MLFLOW_TRACKING_PASSWORD="password"

EXPERIMENT_NAME="dfbakin-plantseg-scheduler-ablation"
SEED=42
MAX_EPOCHS=40
IMAGE_SIZE=384
BATCH_SIZE=16
ACCUM_GRAD=2
LR=2e-4

# [1/11] DONE - constant LR: val/miou=0.314, test/miou=0.377
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

# [2/11] DONE - cosine annealing: val/miou=0.374, test/miou=0.419
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

# [3/11] DONE - step LR: val/miou=0.363, test/miou=0.392
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

# [4/11] DONE - cyclic (triangular2, default): val/miou=0.406, test/miou=0.439
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
    scheduler.mode=triangular2 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

# [5/11] DONE - one-cycle LR: val/miou=0.250, test/miou=0.264
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

#  [6/11] DONE - reduce-on-plateau: val/miou=0.350, test/miou=0.395
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

# [7/11] DONE - Training with polynomial LR (power=0.9)
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
    scheduler=polynomial \
    scheduler.power=0.9 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[9/11] DONE Training with cyclic LR (triangular2, max_lr=3e-3, 4 cycles)..."
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
    scheduler.mode=triangular2 \
    scheduler.max_lr=3e-3 \
    scheduler.num_cycles=4 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[10/11] DONE Training with cyclic LR (exp_range, max_lr=2e-3, 3 cycles)..."
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
    scheduler.mode=exp_range \
    scheduler.max_lr=2e-3 \
    scheduler.gamma=0.9999 \
    scheduler.num_cycles=3 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}

echo "[11/11] DONE Training with cyclic LR (triangular2, max_lr=1.5e-3, 5 cycles)..."
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
    scheduler.mode=triangular2 \
    scheduler.max_lr=1.5e-3 \
    scheduler.num_cycles=5 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}


echo "[11/11] DONE Training with cyclic LR (triangular2, max_lr=3e-4, 5 cycles)..."
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
    scheduler.mode=triangular2 \
    scheduler.max_lr=3e-4 \
    scheduler.num_cycles=5 \
    mlflow.tracking_uri=${MLFLOW_TRACKING_URI}
