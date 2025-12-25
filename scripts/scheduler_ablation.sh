set -e

EXPERIMENT_NAME="plantseg_scheduler_ablation"
SEED=42
MAX_EPOCHS=40
IMAGE_SIZE=384
BATCH_SIZE=16
LR=3e-4

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
    scheduler=constant

echo "[2/6] Training with cosine annealing..."
poetry run python src/train.py \
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
    scheduler=cosine

echo "[3/6] Training with step LR..."
poetry run python src/train.py \
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
    scheduler=step \
    scheduler.step_size=10 \
    scheduler.gamma=0.1

echo "[4/6] Training with cyclic LR (triangular2)..."
poetry run python src/train.py \
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
    scheduler=cyclic \
    scheduler.step_size_up=10 \
    scheduler.mode=triangular2

echo "[5/6] Training with one-cycle LR..."
poetry run python src/train.py \
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
    scheduler=one_cycle \
    scheduler.max_lr_factor=10.0 \
    scheduler.pct_start=0.3

echo "[6/6] Training with reduce-on-plateau..."
poetry run python src/train.py \
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
    scheduler=plateau \
    scheduler.patience=5 \
    scheduler.factor=0.5
