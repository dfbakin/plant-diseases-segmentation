#!/bin/bash
set -e

# SAM1 Pseudomask Refinement Experiments (v2)
#
# Six experiments exploring prompt modes and mask selection strategies:
#
#   A: PSA+RW,   mask_only,       best_iou       (baseline)
#   B: PSA+RW,   mask_only,       smallest_area   (anti-overseg)
#   C: PSA+RW,   box_only,        best_iou        (decoupled from mask shape)
#   D: WeakCLIP, mask_only,       smallest_area   (anti-overseg on high-recall input)
#   E: HA-CRF,   mask_and_points, best_iou        (high-precision input + CAM points)
#   F: HA-CRF,   box_and_points,  best_iou        (bbox + CAM points, free segmentation)
#
# Rationale:
#   - A is the original baseline (mask prompt, pick best IoU).
#   - B tests whether picking SAM's tightest mask reduces oversegmentation.
#   - C removes the mask prompt entirely; SAM segments freely inside the bbox.
#   - D applies the tightest-mask strategy to WeakCLIP (high recall, high overseg).
#   - E uses HA-CRF masks (high BG IoU / low overseg) + CAM points to expand coverage.
#   - F same as E but with bbox instead of mask, letting SAM decide boundaries.
#
# Usage:
#   ./scripts/run_sam_refinement_experiments.sh
#   SAM_MODEL=facebook/sam-vit-large ./scripts/run_sam_refinement_experiments.sh
#   BATCH_SIZE=4 ./scripts/run_sam_refinement_experiments.sh
#   EXPERIMENTS="B D" ./scripts/run_sam_refinement_experiments.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# ─── Configuration ────────────────────────────────────────────
IMAGE_DIR="data/plantsegv3/images/train"
GT_DIR="outputs/plantseg_binary/gt_binary_train"
CAM_DIR="outputs/plantseg_binary/cams/cam_npy"

PSA_MASK_DIR="outputs/plantseg_binary/pseudo_masks_t_0.64"
WEAKCLIP_MASK_DIR="outputs/plantseg_binary/weakclip_masks_t_0.64"
HACRF_MASK_DIR="outputs/plantseg_binary/cams/ha_crf_t_0.64"

OUT_BASE="outputs/plantseg_binary/sam_refined"
VIS_DIR="outputs/visualizations/sam_comparison"

SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-huge}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-50}"
NUM_CLASSES=2
CLASS_NAMES="outputs/plantseg_binary/labels/class_names.txt"

EXPERIMENTS="${EXPERIMENTS:-A B C D E F}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       SAM1 Refinement Experiments (v2)                    ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Model:       ${SAM_MODEL}"
echo "║  Batch size:  ${BATCH_SIZE}"
echo "║  Min comp:    ${MIN_COMPONENT_SIZE} px"
echo "║  Experiments: ${EXPERIMENTS}"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  A: PSA+RW   mask_only       best_iou      (baseline)    ║"
echo "║  B: PSA+RW   mask_only       smallest_area               ║"
echo "║  C: PSA+RW   box_only        best_iou                    ║"
echo "║  D: WeakCLIP mask_only       smallest_area               ║"
echo "║  E: HA-CRF   mask_and_points best_iou                    ║"
echo "║  F: HA-CRF   box_and_points  best_iou                    ║"
echo "╚════════════════════════════════════════════════════════════╝"

run_experiment() {
    local tag="$1" mask_dir="$2" mode="$3" selection="$4" mask_ext="$5"
    shift 5
    echo ""
    echo "=== Experiment ${tag} ==="
    python src/refine_masks_sam.py \
        image_dir="${IMAGE_DIR}" \
        mask_dir="${mask_dir}" \
        mask_ext="${mask_ext}" \
        output_dir="${OUT_BASE}/${tag}" \
        model_name="${SAM_MODEL}" \
        prompt_mode="${mode}" \
        mask_selection="${selection}" \
        num_classes=${NUM_CLASSES} \
        batch_size=${BATCH_SIZE} \
        min_component_size=${MIN_COMPONENT_SIZE} \
        "$@"

    echo "--- Evaluating ${tag} ---"
    python src/evaluate_masks.py \
        pred_dir="${OUT_BASE}/${tag}" \
        gt_dir="${GT_DIR}" \
        num_cls=${NUM_CLASSES} \
        class_names_file="${CLASS_NAMES}"
}

# ─── A: PSA+RW, mask_only, best_iou (baseline) ───────────────
if echo "${EXPERIMENTS}" | grep -qw "A"; then
    run_experiment A "${PSA_MASK_DIR}" mask_only best_iou .png
fi

# ─── B: PSA+RW, mask_only, smallest_area ─────────────────────
if echo "${EXPERIMENTS}" | grep -qw "B"; then
    run_experiment B "${PSA_MASK_DIR}" mask_only smallest_area .png
fi

# ─── C: PSA+RW, box_only, best_iou ───────────────────────────
if echo "${EXPERIMENTS}" | grep -qw "C"; then
    run_experiment C "${PSA_MASK_DIR}" box_only best_iou .png
fi

# ─── D: WeakCLIP, mask_only, smallest_area ────────────────────
if echo "${EXPERIMENTS}" | grep -qw "D"; then
    run_experiment D "${WEAKCLIP_MASK_DIR}" mask_only smallest_area .png
fi

# ─── E: HA-CRF, mask_and_points, best_iou ────────────────────
if echo "${EXPERIMENTS}" | grep -qw "E"; then
    run_experiment E "${HACRF_MASK_DIR}" mask_and_points best_iou .npy \
        cam_dir="${CAM_DIR}" \
        num_pos_points=3 \
        num_neg_points=3 \
        pos_quantile=0.95 \
        neg_quantile=0.05
fi

# ─── F: HA-CRF, box_and_points, best_iou ─────────────────────
if echo "${EXPERIMENTS}" | grep -qw "F"; then
    run_experiment F "${HACRF_MASK_DIR}" box_and_points best_iou .npy \
        cam_dir="${CAM_DIR}" \
        num_pos_points=3 \
        num_neg_points=3 \
        pos_quantile=0.95 \
        neg_quantile=0.05
fi

# ─── Visualization ────────────────────────────────────────────
echo ""
echo "=== Generating visual comparison (20 samples) ==="

VIS_MASK_DIRS="mask_dirs=[{path: ${PSA_MASK_DIR}, label: PSA+RW},{path: ${WEAKCLIP_MASK_DIR}, label: WeakCLIP}"
for tag in A B C D E F; do
    if echo "${EXPERIMENTS}" | grep -qw "${tag}" && [ -d "${OUT_BASE}/${tag}" ]; then
        VIS_MASK_DIRS="${VIS_MASK_DIRS},{path: ${OUT_BASE}/${tag}, label: SAM-${tag}}"
    fi
done
VIS_MASK_DIRS="${VIS_MASK_DIRS}]"

python src/visualize_mask_comparison.py \
    image_dir="${IMAGE_DIR}" \
    gt_dir="${GT_DIR}" \
    "${VIS_MASK_DIRS}" \
    output_dir="${VIS_DIR}" \
    num_samples=20

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete.                                ║"
echo "║  Masks:   ${OUT_BASE}/{A,B,C,D,E,F}/                     ║"
echo "║  Visuals: ${VIS_DIR}/                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
