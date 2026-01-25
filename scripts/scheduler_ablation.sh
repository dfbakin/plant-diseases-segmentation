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

# [1/11] DONE - constant LR: val/miou=0.3659, test/miou=0.4085, lr_end=2.0e-4
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

# [2/11] DONE - cosine annealing: val/miou=0.4158, test/miou=0.4483, lr_end=1.0e-6
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

# [3/11] DONE - step LR (step_size=10, gamma=0.1): val/miou=0.3907, test/miou=0.4218, lr_end=0.0
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

# [4/11] DONE - cyclic (triangular2, max_lr=1e-3, num_cycles=3): val/miou=0.3397, test/miou=0.4058, lr_end=4.63e-4
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

# [5/11] DONE - one-cycle LR (max_lr_factor=10, pct_start=0.3): val/miou=0.2839, test/miou=0.3197, lr_end=1.666e-3
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

# [6/11] DONE - reduce-on-plateau (patience=5, factor=0.5): val/miou=0.3774, test/miou=0.4093, lr_end=2.0e-4
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

# [7/11] DONE - polynomial LR (power=0.9): val/miou=0.4195, test/miou=0.4411, lr_end=7.0e-6
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

# [8/11] DONE - cyclic (triangular2, max_lr=3e-3, num_cycles=4): val/miou=0.3312, test/miou=0.3519, lr_end=1.51e-4
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

# [9/11] DONE - cyclic (exp_range, max_lr=2e-3, gamma=0.9999, num_cycles=3): val/miou=0.2891, test/miou=0.3876, lr_end=7.07e-4
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

# [10/11] DONE - cyclic (triangular2, max_lr=1.5e-3, num_cycles=5): val/miou=0.3324, test/miou=0.3936, lr_end=3.29e-4
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


# [11/11] DONE - cyclic (triangular2, max_lr=3e-4, num_cycles=5): val/miou=0.3822, test/miou=0.4472, lr_end=6.6e-5
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
