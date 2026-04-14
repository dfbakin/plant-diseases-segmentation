#!/usr/bin/env bash
# SPDNet: full pipeline — train, generate CAMs, evaluate, compare with MCTformer.
#
# Usage:
#   bash scripts/compare_cam_methods.sh          # full training (45 epochs)
#   EPOCHS=5 bash scripts/compare_cam_methods.sh  # smoke test
set -euo pipefail

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

EPOCHS="${EPOCHS:-45}"
BATCH="${BATCH:-16}"
SEED="${SEED:-42}"

echo "=== Phase 1: SPDNet Training (${EPOCHS} epochs) ==="
python src/train_spdnet.py \
    trainer.max_epochs="${EPOCHS}" \
    data.batch_size="${BATCH}" \
    seed="${SEED}"

# Find latest training run
LATEST_RUN=$(ls -td outputs/spdnet_plantseg/2026-* | head -1)
CKPT="${LATEST_RUN}/checkpoints/last.ckpt"
echo "Using checkpoint: ${CKPT}"

echo "=== Phase 2: Generate Validation CAMs ==="
python src/generate_spdnet_cams.py \
    checkpoint="${CKPT}" \
    image_dir=data/plantsegv3/images/val \
    image_ext=.jpg \
    label_file=outputs/plantseg_binary_mc115/labels/plantseg_wsss_val.npy \
    output_dir=outputs/spdnet_plantseg/cams/cam_npy_val \
    num_classes=115 \
    input_size=448 \
    "scales=[1.0,0.75,1.25]" \
    num_ref_images=1 \
    binary_aggregate=max \
    gt_dir=outputs/plantseg_binary_mc115/gt_binary_val \
    eval_threshold_sweep=true \
    eval_optimize_metric=disease_iou

echo "=== Done ==="
echo "MCTformer baseline: disease_iou=30.81% @ threshold=0.58"
echo "SPDNet results: see output above"
