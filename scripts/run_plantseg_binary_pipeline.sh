#!/bin/bash
set -e

# Binary (2-class) WSSS pipeline on PlantSeg + PlantVillage
#
# Downstream pipeline always operates in 2-class mode: background (0) + disease (1).
# MCTformer can be trained in binary (1 fg) or multiclass (115 fg) mode.
# When multiclass, CAMs are aggregated to binary via BINARY_AGGREGATE (max/mean).
#
# Steps:
#   0a. Export combined labels (MCTformer training: PlantSeg + PlantVillage)
#   0b. Export PlantSeg-only labels (CAM generation)
#   0c. Generate binary GT masks from multiclass PlantSeg annotations
#   1.  Train MCTformer classifier (binary or multiclass)
#   2.  Generate CAMs (PlantSeg only; aggregated to binary if multiclass)
#   3.  Apply CRF (la + ha)
#   4.  Evaluate CRF masks vs binary GT
#   5.  Train PSA affinity network
#   6.  Random Walk refinement -> pseudo masks
#   7.  Evaluate pseudo masks vs binary GT
#   8.  Train WeakCLIP on pseudo masks
#   9.  Generate + refine WeakCLIP masks (streaming)
#  10.  Evaluate WeakCLIP masks vs binary GT
#
# Usage:
#   ./scripts/run_plantseg_binary_pipeline.sh
#   MCTFORMER_CKPT=outputs/.../last.ckpt ./scripts/run_plantseg_binary_pipeline.sh
#   SKIP_STEPS="0,1" ./scripts/run_plantseg_binary_pipeline.sh
#   WEAKCLIP_QUALITY=fast ./scripts/run_plantseg_binary_pipeline.sh   # ~12x faster step 9
#
#   # Multiclass MCTformer (115 classes) -> binary aggregation:
#   MCTFORMER_DATASET=plantseg_with_pv \
#   MCTFORMER_EXPERIMENT=mctformer_plantseg_mc115_pv \
#   BINARY_AGGREGATE=max \
#   OUT_BASE=outputs/plantseg_binary_mc115 \
#   BINARY_BASE=outputs/plantseg_binary \
#   WEAKCLIP_EXPERIMENT=weakclip-plantseg-binary-mc115-t_0.73 \
#   scripts/run_plantseg_binary_pipeline.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# ─── Configuration ────────────────────────────────────────────
DATA_ROOT="data/plantsegv3"
PV_ROOT="data/plant-village"
IMAGE_DIR="${DATA_ROOT}/images/train"
GT_DIR="${DATA_ROOT}/annotations/train"
VAL_IMAGE_DIR="${DATA_ROOT}/images/val"
VAL_GT_DIR="${DATA_ROOT}/annotations/val"

OUT_BASE="${OUT_BASE:-outputs/plantseg_binary}"
BINARY_BASE="${BINARY_BASE:-${OUT_BASE}}"

# Binary pipeline always uses 2 classes for downstream (CRF, PSA, RW, WeakCLIP, eval)
NUM_CLS=2           # bg + disease
IMAGE_EXT=".jpg"

# MCTformer: binary (default) or multiclass
#   plantseg_binary   -> 1 fg class, binary labels
#   plantseg_with_pv  -> 115 fg classes, multiclass labels
MCTFORMER_DATASET="${MCTFORMER_DATASET:-plantseg_binary}"
MCTFORMER_EXPERIMENT="${MCTFORMER_EXPERIMENT:-mctformer_plantseg_binary}"
MCTFORMER_CKPT="${MCTFORMER_CKPT:-}"

# Binary aggregation: when MCTformer is multiclass, aggregate class CAMs to binary
#   ""     -> disabled (default, for binary MCTformer)
#   "max"  -> np.max over all class CAMs
#   "mean" -> np.mean over all class CAMs
BINARY_AGGREGATE="${BINARY_AGGREGATE:-}"

# Derive MCTformer model size + label export modes from dataset choice
if [ "${MCTFORMER_DATASET}" = "plantseg_with_pv" ]; then
    MCTFORMER_NUM_CLASSES=115
    TRAIN_LABEL_MODE="plantseg_wsss_with_pv"
    CAM_LABEL_MODE="plantseg_wsss"
    COMBINED_LABEL_FILE="${OUT_BASE}/labels/plantseg_wsss_pv_all_train.npy"
    CAM_LABEL_FILE="${OUT_BASE}/labels/plantseg_wsss_train.npy"
    if [ -z "${BINARY_AGGREGATE}" ]; then
        echo "WARNING: MCTFORMER_DATASET=plantseg_with_pv but BINARY_AGGREGATE is empty."
        echo "  Multiclass CAMs won't be aggregated to binary. Setting BINARY_AGGREGATE=max."
        BINARY_AGGREGATE="max"
    fi
else
    MCTFORMER_NUM_CLASSES=1
    TRAIN_LABEL_MODE="plantseg_binary"
    CAM_LABEL_MODE="plantseg_binary"
    COMBINED_LABEL_FILE="${OUT_BASE}/labels/plantseg_binary_all_train.npy"
    CAM_LABEL_FILE="${OUT_BASE}/labels/plantseg_binary_train.npy"
fi

# Binary shared resources (GT masks, binary labels, class_names)
LABEL_FILE="${BINARY_BASE}/labels/plantseg_binary_train.npy"
BINARY_GT_DIR="${BINARY_BASE}/gt_binary_train"
BINARY_GT_VAL_DIR="${BINARY_BASE}/gt_binary_val"

# Binary class_names file for downstream eval/WeakCLIP.
# When multiclass, label exports write a 115-class class_names.txt into
# ${OUT_BASE}/labels/. Use a distinct filename so it doesn't get overwritten.
if [ "${MCTFORMER_DATASET}" = "plantseg_binary" ]; then
    CLASS_NAMES="${BINARY_BASE}/labels/class_names.txt"
else
    CLASS_NAMES="${OUT_BASE}/labels/binary_class_names.txt"
fi

# CAMs
CAM_DIR="${OUT_BASE}/cams/cam_npy"
LA_CRF_DIR="${OUT_BASE}/cams/la_crf_t_0.73"
HA_CRF_DIR="${OUT_BASE}/cams/ha_crf_t_0.73"

# PSA
PSA_BACKBONE="pretrained/res38_cls.pth"
PSA_CKPT="${OUT_BASE}/psa/psa_aff_t_0.73.pth"

# Pseudo masks
PSEUDO_MASK_DIR="${OUT_BASE}/pseudo_masks_t_0.73"

# WeakCLIP
WEAKCLIP_EXPERIMENT="${WEAKCLIP_EXPERIMENT:-weakclip-plantseg-binary-t_0.73}"
CLIP_PRETRAINED="pretrained/ViT-B-16.pt"
WEAKCLIP_MASK_DIR="${OUT_BASE}/weakclip_masks_t_0.73"

NUM_WORKERS=16
SWEEP_SAMPLES=500

# WeakCLIP inference quality: "fast" for quick trial, "full" for best results
# fast  -> scales=[1.0], flip=false              (~12x faster)
# full  -> scales=[0.5,0.75,1.0,1.25,1.5,1.75], flip=true
WEAKCLIP_QUALITY="${WEAKCLIP_QUALITY:-full}"

SKIP_STEPS="${SKIP_STEPS:-}"

should_skip() {
    echo "${SKIP_STEPS}" | grep -qw "$1"
}

echo "============================================"
echo "  PlantSeg Binary WSSS Pipeline"
echo "  MCTformer: ${MCTFORMER_DATASET} (${MCTFORMER_NUM_CLASSES} fg classes)"
if [ -n "${BINARY_AGGREGATE}" ]; then
    echo "  CAM aggregation: ${BINARY_AGGREGATE} -> binary"
fi
echo "  Downstream: ${NUM_CLS} classes (bg + disease)"
echo "  Data: ${DATA_ROOT} + ${PV_ROOT}"
echo "  Output: ${OUT_BASE}"
if [ "${BINARY_BASE}" != "${OUT_BASE}" ]; then
    echo "  Shared resources: ${BINARY_BASE}"
fi
echo "============================================"
echo ""

# ─── Step 0a: Export combined labels (MCTformer training) ─────
if ! should_skip 0; then
    echo "=== Step 0a: Export combined labels (${TRAIN_LABEL_MODE}) ==="
    if [ "${TRAIN_LABEL_MODE}" = "plantseg_binary" ]; then
        python src/export_labels.py \
            mode=plantseg_binary \
            root="${DATA_ROOT}" \
            pv_root="${PV_ROOT}" \
            pv_split=train \
            include_plantvillage=true \
            output="${COMBINED_LABEL_FILE}"
    else
        python src/export_labels.py \
            mode="${TRAIN_LABEL_MODE}" \
            root="${DATA_ROOT}" \
            pv_root="${PV_ROOT}" \
            pv_split=train \
            output="${COMBINED_LABEL_FILE}"
    fi
    echo "Combined labels: ${COMBINED_LABEL_FILE}"
    echo ""

    # ─── Step 0b: Export CAM-generation labels (PlantSeg only) ─
    echo "=== Step 0b: Export CAM-generation labels (${CAM_LABEL_MODE}) ==="
    if [ "${CAM_LABEL_MODE}" = "plantseg_binary" ]; then
        python src/export_labels.py \
            mode=plantseg_binary \
            root="${DATA_ROOT}" \
            pv_split=train \
            include_plantvillage=false \
            output="${CAM_LABEL_FILE}"
    else
        python src/export_labels.py \
            mode="${CAM_LABEL_MODE}" \
            root="${DATA_ROOT}" \
            pv_split=train \
            output="${CAM_LABEL_FILE}"
    fi
    echo "CAM-gen labels: ${CAM_LABEL_FILE}"
    echo ""

    # Ensure binary class_names exists for downstream eval/WeakCLIP
    if [ "${MCTFORMER_DATASET}" != "plantseg_binary" ]; then
        mkdir -p "$(dirname "${CLASS_NAMES}")"
        echo "disease" > "${CLASS_NAMES}"
        echo "Binary class_names (downstream): ${CLASS_NAMES}"
        echo ""
    fi

    # ─── Step 0c: Generate binary GT masks ────────────────────
    echo "=== Step 0c: Generate binary GT masks ==="
    python -c "
from pathlib import Path
import numpy as np
from PIL import Image

for split, src_name, dst_name in [
    ('train', '${GT_DIR}', '${BINARY_GT_DIR}'),
    ('val', '${VAL_GT_DIR}', '${BINARY_GT_VAL_DIR}'),
]:
    src = Path(src_name)
    dst = Path(dst_name)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src.glob('*.png')):
        m = np.array(Image.open(f))
        m[(m > 0) & (m < 255)] = 1
        Image.fromarray(m.astype(np.uint8)).save(dst / f.name)
        count += 1
    print(f'{split}: converted {count} masks -> {dst}')
"
    echo "Binary GT: ${BINARY_GT_DIR}, ${BINARY_GT_VAL_DIR}"
    echo ""
fi

# ─── Step 1: Train MCTformer ─────────────────────────────────
if ! should_skip 1; then
    if [ -n "${MCTFORMER_CKPT}" ] && [ -f "${MCTFORMER_CKPT}" ]; then
        echo "=== Step 1: Skipping MCTformer training (using ${MCTFORMER_CKPT}) ==="
    else
        echo "=== Step 1: Train MCTformer-V2 (${MCTFORMER_DATASET}, ${MCTFORMER_NUM_CLASSES} fg classes) ==="
        python src/train_mctformer.py \
            dataset="${MCTFORMER_DATASET}" \
            experiment_name="${MCTFORMER_EXPERIMENT}" \
            seed=0 \
            model.name=mctformer_v2 \
            model.pretrained=true \
            model.num_classes=${MCTFORMER_NUM_CLASSES} \
            model.learning_rate=5e-4 \
            model.weight_decay=0.05 \
            model.drop_path_rate=0.1 \
            model.label_smoothing=0.1 \
            model.input_size=512 \
            plantseg_data.root="${DATA_ROOT}" \
            plantseg_data.pv_root="${PV_ROOT}" \
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

# Ensure we have a checkpoint (only needed for step 2)
if [ -z "${MCTFORMER_CKPT}" ] && ! should_skip 2; then
    MCTFORMER_CKPT=$(ls -t outputs/${MCTFORMER_EXPERIMENT}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    if [ -z "${MCTFORMER_CKPT}" ]; then
        echo "ERROR: MCTFORMER_CKPT not set and no checkpoint found in outputs/${MCTFORMER_EXPERIMENT}/"
        exit 1
    fi
fi
if [ -n "${MCTFORMER_CKPT}" ]; then
    echo "Using MCTformer checkpoint: ${MCTFORMER_CKPT}"
fi

# ─── Step 2: Generate CAMs (PlantSeg only) ───────────────────
if ! should_skip 2; then
    CAM_AGG_ARGS=""
    if [ -n "${BINARY_AGGREGATE}" ]; then
        CAM_AGG_ARGS="binary_aggregate=${BINARY_AGGREGATE}"
        echo "=== Step 2: Generate CAMs (${MCTFORMER_NUM_CLASSES} classes -> ${BINARY_AGGREGATE} -> binary) ==="
    else
        echo "=== Step 2: Generate CAMs (PlantSeg only, binary) ==="
    fi
    python src/generate_cams.py \
        "checkpoint='${MCTFORMER_CKPT}'" \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        "label_file=${CAM_LABEL_FILE}" \
        output_dir="${CAM_DIR}" \
        num_classes=${MCTFORMER_NUM_CLASSES} \
        input_size=512 \
        max_size=896 \
        "scales=[1.0,0.75,1.25,1.5,1.75]" \
        n_layers=3 \
        attention_type=fused \
        patch_attn_refine=true \
        gt_dir="${BINARY_GT_DIR}" \
        eval_threshold_sweep=true \
        eval_sweep_samples=${SWEEP_SAMPLES} \
        ${CAM_AGG_ARGS}
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
        bg_threshold=0.73 \
        la_scale_factor=1.0 \
        ha_scale_factor=12.0 \
        crf_iters=10 \
        num_cls=${NUM_CLS} \
        num_workers=${NUM_WORKERS}
    echo "CRF masks: ${LA_CRF_DIR}, ${HA_CRF_DIR}"
    echo ""
fi

# ─── Step 4: Evaluate CRF masks vs binary GT ─────────────────
if ! should_skip 4; then
    echo "=== Step 4: Evaluate la_crf masks ==="
    python src/evaluate_masks.py \
        pred_dir="${LA_CRF_DIR}" \
        gt_dir="${BINARY_GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"

    echo ""
    echo "=== Step 4: Evaluate ha_crf masks ==="
    python src/evaluate_masks.py \
        pred_dir="${HA_CRF_DIR}" \
        gt_dir="${BINARY_GT_DIR}" \
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
        max_epochs=20 \
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

# ─── Step 7: Evaluate pseudo masks vs binary GT ──────────────
if ! should_skip 7; then
    echo "=== Step 7: Evaluate pseudo masks (MCTformer + PSA + RW) ==="
    python src/evaluate_masks.py \
        pred_dir="${PSEUDO_MASK_DIR}" \
        gt_dir="${BINARY_GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"
    echo ""
fi

# ─── Step 8: Train WeakCLIP ──────────────────────────────────
if ! should_skip 8; then
    echo "=== Step 8: Train WeakCLIP on pseudo masks (binary) ==="
    python src/train_weakclip.py \
        class_names_file="${CLASS_NAMES}" \
        num_classes=${NUM_CLS} \
        train_image_dir="${IMAGE_DIR}" \
        train_mask_dir="${PSEUDO_MASK_DIR}" \
        val_image_dir="${VAL_IMAGE_DIR}" \
        val_mask_dir="${BINARY_GT_VAL_DIR}" \
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
    if [ "${WEAKCLIP_QUALITY}" = "fast" ]; then
        WC_SCALES="[1.0]"
        WC_FLIP="false"
        echo "=== Step 9: Generate + Refine WeakCLIP masks (FAST mode: single scale, no flip) ==="
    else
        WC_SCALES="[0.5,0.75,1.0,1.25,1.5]"
        WC_FLIP="true"
        echo "=== Step 9: Generate + Refine WeakCLIP masks (FULL mode: 6 scales + flip) ==="
    fi
    python src/generate_refine_weakclip_masks.py \
        "checkpoint='${WEAKCLIP_CKPT}'" \
        class_names_file="${CLASS_NAMES}" \
        num_classes=${NUM_CLS} \
        image_dir="${IMAGE_DIR}" \
        image_ext="${IMAGE_EXT}" \
        labels_file="${LABEL_FILE}" \
        output_dir="${WEAKCLIP_MASK_DIR}" \
        "scales=${WC_SCALES}" \
        flip=${WC_FLIP} \
        crop_size=512 \
        stride=341 \
        max_long_edge=2048 \
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

# ─── Step 10: Evaluate WeakCLIP masks vs binary GT ───────────
if ! should_skip 10; then
    echo "=== Step 10: Evaluate WeakCLIP refined masks ==="
    python src/evaluate_masks.py \
        pred_dir="${WEAKCLIP_MASK_DIR}" \
        gt_dir="${BINARY_GT_DIR}" \
        num_cls=${NUM_CLS} \
        class_names_file="${CLASS_NAMES}"
    echo ""
fi

echo "============================================"
echo "  Binary Pipeline complete!"
echo "  Pseudo masks (MCTformer): ${PSEUDO_MASK_DIR}"
echo "  WeakCLIP masks:          ${WEAKCLIP_MASK_DIR}"
echo "============================================"
