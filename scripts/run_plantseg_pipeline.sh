#!/bin/bash
set -e

# Full WSSS pipeline on PlantSeg dataset
#
# Steps:
#   0. Export image-level labels (plantseg_wsss mode, 115 disease classes)
#   1. Train MCTformer classifier (optional, skip with MCTFORMER_CKPT)
#   2. Generate CAMs
#   3. Apply CRF (la + ha)
#   4. Evaluate CRF masks vs GT
#   5. Train PSA affinity network
#   6. Random Walk refinement -> pseudo masks
#   7. Evaluate pseudo masks vs GT
#   8. Train WeakCLIP on pseudo masks
#   9. Generate + refine WeakCLIP masks (streaming, no intermediate probs)
#  10. Evaluate WeakCLIP masks vs GT
#
# Usage:
#   ./scripts/run_plantseg_pipeline.sh                          # full pipeline
#   MCTFORMER_CKPT=outputs/.../last.ckpt ./scripts/run_plantseg_pipeline.sh  # skip training
#   SKIP_STEPS="0,1" ./scripts/run_plantseg_pipeline.sh         # skip specific steps

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# ─── Configuration ────────────────────────────────────────────
DATA_ROOT="data/plantsegv3"
IMAGE_DIR="${DATA_ROOT}/images/train"
GT_DIR="${DATA_ROOT}/annotations/train"
VAL_IMAGE_DIR="${DATA_ROOT}/images/val"
VAL_GT_DIR="${DATA_ROOT}/annotations/val"

OUT_BASE="outputs/plantseg_wsss"
LABEL_FILE="${OUT_BASE}/labels/plantseg_wsss_train.npy"
CLASS_NAMES="${OUT_BASE}/labels/class_names.txt"

NUM_FG=115          # foreground disease classes
NUM_CLS=116         # total including background
IMAGE_EXT=".jpg"

# MCTformer
MCTFORMER_EXPERIMENT="mctformer_plantseg"
MCTFORMER_CKPT="${MCTFORMER_CKPT:-}"   # set externally to skip training

# CAMs
CAM_DIR="${OUT_BASE}/cams/cam_npy"
LA_CRF_DIR="${OUT_BASE}/cams/la_crf"
HA_CRF_DIR="${OUT_BASE}/cams/ha_crf"

# PSA
PSA_BACKBONE="pretrained/res38_cls.pth"
PSA_CKPT="${OUT_BASE}/psa/psa_aff.pth"

# Pseudo masks
PSEUDO_MASK_DIR="${OUT_BASE}/pseudo_masks"

# WeakCLIP
WEAKCLIP_EXPERIMENT="weakclip-plantseg"
CLIP_PRETRAINED="pretrained/ViT-B-16.pt"
WEAKCLIP_MASK_DIR="${OUT_BASE}/weakclip_masks"

NUM_WORKERS=16

# Parse SKIP_STEPS as comma-separated list
SKIP_STEPS="${SKIP_STEPS:-}"

should_skip() {
    echo "${SKIP_STEPS}" | grep -qw "$1"
}

echo "============================================"
echo "  PlantSeg WSSS Pipeline"
echo "  Classes: ${NUM_FG} fg + 1 bg = ${NUM_CLS}"
echo "  Data: ${DATA_ROOT}"т
echo "  Output: ${OUT_BASE}"
echo "============================================"
echo ""

# ─── Step 0: Export Labels ────────────────────────────────────
if ! should_skip 0; then
    echo "=== Step 0: Export image-level labels (plantseg_wsss) ==="
    python src/export_labels.py \
        mode=plantseg_wsss \
        root="${DATA_ROOT}" \
        pv_split=train \
        output="${LABEL_FILE}"
    echo "Labels: ${LABEL_FILE}"
    echo "Class names: ${CLASS_NAMES}"
    echo ""
fi

# ─── Step 1: Train MCTformer ─────────────────────────────────
if ! should_skip 1; then
    if [ -n "${MCTFORMER_CKPT}" ] && [ -f "${MCTFORMER_CKPT}" ]; then
        echo "=== Step 1: Skipping MCTformer training (using ${MCTFORMER_CKPT}) ==="
    else
        echo "=== Step 1: Train MCTformer-V2 on PlantSeg ==="
        python src/train_mctformer.py \
            dataset=plantseg \
            experiment_name="${MCTFORMER_EXPERIMENT}" \
            seed=0 \
            model.name=mctformer_v2 \
            model.pretrained=true \
            model.learning_rate=5e-4 \
            model.weight_decay=0.05 \
            model.drop_path_rate=0.1 \
            model.label_smoothing=0.1 \
            model.input_size=512 \
            plantseg_data.root="${DATA_ROOT}" \
            plantseg_data.image_size=512 \
            plantseg_data.batch_size=32 \
            plantseg_data.num_workers=${NUM_WORKERS} \
            trainer.max_epochs=45 \
            trainer.precision="32"

        MCTFORMER_CKPT=$(ls -t outputs/${MCTFORMER_EXPERIMENT}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
        if [ -z "${MCTFORMER_CKPT}" ]; then
            echo "ERROR: No MCTformer checkpoint found after training"
            exit 1
        fi
        echo "MCTformer checkpoint: ${MCTFORMER_CKPT}"
    fi
    echo ""
fi

# Ensure we have a checkpoint for subsequent steps
if [ -z "${MCTFORMER_CKPT}" ]; then
    MCTFORMER_CKPT=$(ls -t outputs/${MCTFORMER_EXPERIMENT}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    if [ -z "${MCTFORMER_CKPT}" ]; then
        echo "ERROR: MCTFORMER_CKPT not set and no checkpoint found in outputs/${MCTFORMER_EXPERIMENT}/"
        exit 1
    fi
fi
echo "Using MCTformer checkpoint: ${MCTFORMER_CKPT}"

# ─── Step 2: Generate CAMs ───────────────────────────────────
if ! should_skip 2; then
    echo "=== Step 2: Generate CAMs ==="
    python src/generate_cams.py \
        "checkpoint='${MCTFORMER_CKPT}'" \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        "label_file=${LABEL_FILE}" \
        output_dir="${CAM_DIR}" \
        num_classes=${NUM_FG} \
        input_size=512 \
        max_size=896 \
        "scales=[1.0,0.75,1.25,1.5,1.75]" \
        n_layers=3 \
        attention_type=fused \
        patch_attn_refine=true \
        gt_dir="${GT_DIR}" \
        eval_threshold_sweep=false
    echo "CAMs: ${CAM_DIR}"
    echo ""
fi

# ─── Step 3: Apply CRF (la + ha) ─────────────────────────────
if ! should_skip 3; then
    echo "=== Step 3: Apply CRF ==="
    python src/apply_crf.py \
        cam_dir="${CAM_DIR}" \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        la_crf_dir="${LA_CRF_DIR}" \
        ha_crf_dir="${HA_CRF_DIR}" \
        bg_threshold=0.3 \
        la_scale_factor=1.0 \
        ha_scale_factor=12.0 \
        crf_iters=10 \
        num_cls=${NUM_CLS} \
        num_workers=${NUM_WORKERS}
    echo "CRF masks: ${LA_CRF_DIR}, ${HA_CRF_DIR}"
    echo ""
fi

# ─── Step 4: Evaluate CRF masks vs GT ────────────────────────
if ! should_skip 4; then
    echo "=== Step 4: Evaluate la_crf masks ==="
    python src/evaluate_masks.py \
        pred_dir="${LA_CRF_DIR}" \
        gt_dir="${GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"

    echo ""
    echo "=== Step 4: Evaluate ha_crf masks ==="
    python src/evaluate_masks.py \
        pred_dir="${HA_CRF_DIR}" \
        gt_dir="${GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"
    echo ""
fi

# ─── Step 5: Train PSA ───────────────────────────────────────
if ! should_skip 5; then
    echo "=== Step 5: Train PSA Affinity Network ==="
    python src/train_psa.py \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        la_crf_dir="${LA_CRF_DIR}" \
        ha_crf_dir="${HA_CRF_DIR}" \
        backbone_weights="${PSA_BACKBONE}" \
        output_path="${PSA_CKPT}" \
        batch_size=8 \
        max_epochs=10 \
        lr=0.01 \
        num_workers=${NUM_WORKERS} \
        cropsize=512
    echo "PSA: ${PSA_CKPT}"
    echo ""
fi

# ─── Step 6: Random Walk ─────────────────────────────────────
if ! should_skip 6; then
    echo "=== Step 6: Random Walk Refinement ==="
    python src/run_random_walk.py \
        cam_dir="${CAM_DIR}" \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        aff_checkpoint="${PSA_CKPT}" \
        output_dir="${PSEUDO_MASK_DIR}" \
        bg_threshold=0.39 \
        beta=8 \
        logt=6 \
        num_cls=${NUM_CLS} \
        cropsize=512 \
        max_size=640
    echo "Pseudo masks: ${PSEUDO_MASK_DIR}"
    echo ""
fi

# ─── Step 7: Evaluate pseudo masks vs GT ─────────────────────
if ! should_skip 7; then
    echo "=== Step 7: Evaluate pseudo masks (MCTformer + PSA + RW) ==="
    python src/evaluate_masks.py \
        pred_dir="${PSEUDO_MASK_DIR}" \
        gt_dir="${GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"
    echo ""
fi

# ─── Step 8: Train WeakCLIP ──────────────────────────────────
if ! should_skip 8; then
    echo "=== Step 8: Train WeakCLIP on pseudo masks ==="
    python src/train_weakclip.py \
        class_names_file="${CLASS_NAMES}" \
        num_classes=${NUM_CLS} \
        train_image_dir="${IMAGE_DIR}" \
        train_mask_dir="${PSEUDO_MASK_DIR}" \
        val_image_dir="${VAL_IMAGE_DIR}" \
        val_mask_dir="${VAL_GT_DIR}" \
        val_names_file="" \
        image_ext="${IMAGE_EXT}" \
        clip_pretrained="${CLIP_PRETRAINED}" \
        image_size=512 \
        batch_size=16 \
        max_epochs=31 \
        learning_rate=1e-4 \
        min_lr=1e-6 \
        warmup_iters=1500 \
        weight_decay=3e-5 \
        identity_loss_weight=0.4 \
        use_crf_loss=true \
        crf_iters=10 \
        norm_eval=true \
        num_workers=${NUM_WORKERS} \
        precision="32" \
        experiment_name="${WEAKCLIP_EXPERIMENT}"
    echo ""
fi

# Find WeakCLIP checkpoint
WEAKCLIP_CKPT=$(ls -t outputs/weakclip/${WEAKCLIP_EXPERIMENT}/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -z "${WEAKCLIP_CKPT}" ]; then
    echo "WARNING: No WeakCLIP checkpoint found; skipping steps 9-10"
    exit 0
fi
echo "Using WeakCLIP checkpoint: ${WEAKCLIP_CKPT}"

# ─── Step 9: Generate + Refine WeakCLIP masks (streaming) ────
if ! should_skip 9; then
    echo "=== Step 9: Generate + Refine WeakCLIP masks (streaming) ==="
    python src/generate_refine_weakclip_masks.py \
        "checkpoint='${WEAKCLIP_CKPT}'" \
        class_names_file="${CLASS_NAMES}" \
        num_classes=${NUM_CLS} \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        labels_file="${LABEL_FILE}" \
        output_dir="${WEAKCLIP_MASK_DIR}" \
        "scales=[0.5,0.75,1.0,1.25,1.5,1.75]" \
        flip=true \
        crop_size=512 \
        stride=341 \
        clip_pretrained="${CLIP_PRETRAINED}" \
        image_size=512 \
        crf_t=10 \
        crf_sxy_gauss=3.0 \
        crf_compat_gauss=3.0 \
        crf_sxy_bilat=83.0 \
        crf_srgb_bilat=5.0 \
        crf_compat_bilat=3.0
    echo "WeakCLIP masks: ${WEAKCLIP_MASK_DIR}"
    echo ""
fi

# ─── Step 10: Evaluate WeakCLIP masks vs GT ──────────────────
if ! should_skip 10; then
    echo "=== Step 10: Evaluate WeakCLIP refined masks ==="
    python src/evaluate_masks.py \
        pred_dir="${WEAKCLIP_MASK_DIR}" \
        gt_dir="${GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"
    echo ""
fi

echo "============================================"
echo "  Pipeline complete!"
echo "  Pseudo masks (MCTformer): ${PSEUDO_MASK_DIR}"
echo "  WeakCLIP masks:          ${WEAKCLIP_MASK_DIR}"
echo "============================================"
