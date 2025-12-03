set -e

# Experiment settings
EXPERIMENT_NAME="plantseg_augmentation_ablation"
SEED=42
MAX_EPOCHS=30

# Phase 1: Full ablation (9 presets × 1 model)
python3 src/train.py --multirun \
  augmentation=baseline,spatial_light,spatial_heavy,color_natural,artificial_color,noise_blur,natural_color,spatial_color_light,full \
  model=segformer \
  model.variant=b3 \
  data.image_size=384 \
  experiment.name="${EXPERIMENT_NAME}" \
  experiment.seed=${SEED} \
  trainer.max_epochs=${MAX_EPOCHS} \
  data.batch_size=32 \
  model.learning_rate=3e-4
  
# Phase 2: Validate on second architecture
python3 src/train.py --multirun \
  augmentation=baseline,spatial_light,color_natural,spatial_color_light,full \
  model=deeplabv3plus \
  model.encoder_name=resnet50 \
  data.image_size=384 \
  experiment.name="${EXPERIMENT_NAME}" \
  experiment.seed=${SEED} \
  trainer.max_epochs=${MAX_EPOCHS} \
  data.batch_size=32 \
  model.learning_rate=3e-4

python3 src/train.py --multirun \
  augmentation=spatial_color_light \
  augmentation.transforms.3.p=0.3,0.5,0.7 \
  model=segformer \
  model.variant=b3 \
  data.image_size=384 \
  experiment.name="${EXPERIMENT_NAME}" \
  experiment.seed=${SEED} \
  trainer.max_epochs=${MAX_EPOCHS} \
  data.batch_size=32
  model.learning_rate=3e-4
