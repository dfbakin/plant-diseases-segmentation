#!/bin/bash
set -e

# SAM1 Pseudomask Refinement Experiments (v3 — MC115 pipeline)
#
# All experiments use MC115 pipeline outputs (multiclass MCTformer -> binary
# aggregation). The MC115 CAMs are continuous [0,1] float arrays that serve
# as both soft mask prompts and point-sampling sources.
#
# Experiment matrix:
#
#   G: MC115 WeakCLIP (binary),  mask_only,       smallest_area
#        Baseline: better masks from MC115 pipeline + anti-overseg selection
#
#   H: MC115 CAMs (continuous),  soft_mask,        smallest_area
#        Graded logits (scale=3.0) — SAM decides uncertain pixels itself
#
#   I: MC115 CAMs (gated p>0.3), soft_mask,        smallest_area
#        Same as H but with noise floor removal via confidence_threshold
#
#   J: MC115 WeakCLIP bbox from CAM>0.5 + CAM points, box_and_points, smallest_area
#        Conservative spatial extent from high-confidence CAM region + anchor points
#
#   K: MC115 CAMs (gated p>0.3) + CAM points,  soft_mask_and_points, smallest_area
#        Full signal: graded confidence + spatially diverse anchor points
#
# Usage:
#   ./scripts/run_sam_refinement_experiments.sh
#   SAM_MODEL=facebook/sam-vit-large ./scripts/run_sam_refinement_experiments.sh
#   BATCH_SIZE=4 ./scripts/run_sam_refinement_experiments.sh
#   EXPERIMENTS="G H I" ./scripts/run_sam_refinement_experiments.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# ─── Configuration ────────────────────────────────────────────
IMAGE_DIR="data/plantsegv3/images/train"
GT_DIR="outputs/plantseg_binary/gt_binary_train"

# MC115 pipeline artefacts
CAM_DIR="outputs/plantseg_binary_mc115/cams/cam_npy"
WEAKCLIP_MASK_DIR="outputs/plantseg_binary_mc115/weakclip_masks_t_0.73"

OUT_BASE="outputs/plantseg_binary_mc115/sam_refined"
VIS_DIR="outputs/visualizations/sam_mc115_comparison"

SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-huge}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-50}"
NUM_CLASSES=2
CLASS_NAMES="outputs/plantseg_binary_mc115/labels/binary_class_names.txt"

# Fall back to the binary pipeline class_names if mc115 one doesn't exist
if [ ! -f "${CLASS_NAMES}" ]; then
    CLASS_NAMES="outputs/plantseg_binary/labels/class_names.txt"
fi
# Generate class_names on the fly if neither exists
if [ ! -f "${CLASS_NAMES}" ]; then
    mkdir -p "$(dirname "${CLASS_NAMES}")"
    echo "disease" > "${CLASS_NAMES}"
fi

EXPERIMENTS="${EXPERIMENTS:-G H I J K}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       SAM1 Refinement Experiments v3 (MC115)                   ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Model:       ${SAM_MODEL}"
echo "║  Batch size:  ${BATCH_SIZE}"
echo "║  Min comp:    ${MIN_COMPONENT_SIZE} px"
echo "║  CAM dir:     ${CAM_DIR}"
echo "║  WeakCLIP:    ${WEAKCLIP_MASK_DIR}"
echo "║  Experiments: ${EXPERIMENTS}"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  G: MC115 WeakCLIP  mask_only          smallest_area           ║"
echo "║  H: MC115 CAM       soft_mask(s=3.0)   smallest_area           ║"
echo "║  I: MC115 CAM       soft_mask(s=3,t=.3) smallest_area          ║"
echo "║  J: MC115 bbox+pts  box_and_points     smallest_area           ║"
echo "║  K: MC115 CAM+pts   soft_mask_and_pts  smallest_area           ║"
echo "╚════════════════════════════════════════════════════════════════╝"

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

# ─── G: MC115 WeakCLIP binary masks, mask_only, smallest_area ─
if echo "${EXPERIMENTS}" | grep -qw "G"; then
    run_experiment G "${WEAKCLIP_MASK_DIR}" mask_only smallest_area .png
fi

# ─── H: MC115 CAMs as soft mask, logit_scale=3.0, smallest_area
if echo "${EXPERIMENTS}" | grep -qw "H"; then
    run_experiment H "${CAM_DIR}" soft_mask smallest_area .npy \
        prob_dir="${CAM_DIR}" \
        logit_scale=3.0 \
        confidence_threshold=0.0
fi

# ─── I: MC115 CAMs gated at 0.3, logit_scale=3.0, smallest_area
if echo "${EXPERIMENTS}" | grep -qw "I"; then
    run_experiment I "${CAM_DIR}" soft_mask smallest_area .npy \
        prob_dir="${CAM_DIR}" \
        logit_scale=3.0 \
        confidence_threshold=0.3
fi

# ─── J: tight bbox from CAM>0.5 + CAM points, smallest_area ──
if echo "${EXPERIMENTS}" | grep -qw "J"; then
    run_experiment J "${WEAKCLIP_MASK_DIR}" box_and_points smallest_area .png \
        cam_dir="${CAM_DIR}" \
        num_pos_points=5 \
        num_neg_points=5 \
        pos_quantile=0.90 \
        neg_quantile=0.10
fi

# ─── K: MC115 CAMs (gated) + CAM points, smallest_area ────────
if echo "${EXPERIMENTS}" | grep -qw "K"; then
    run_experiment K "${CAM_DIR}" soft_mask_and_points smallest_area .npy \
        prob_dir="${CAM_DIR}" \
        logit_scale=3.0 \
        confidence_threshold=0.3 \
        cam_dir="${CAM_DIR}" \
        num_pos_points=5 \
        num_neg_points=5 \
        pos_quantile=0.90 \
        neg_quantile=0.10
fi

# ─── Visualization ────────────────────────────────────────────
echo ""
echo "=== Generating visual comparison (20 samples) ==="

VIS_MASK_DIRS="mask_dirs=[{path: ${WEAKCLIP_MASK_DIR}, label: MC115-WeakCLIP}"
for tag in G H I J K; do
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
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete.                                     ║"
echo "║  Masks:   ${OUT_BASE}/{G,H,I,J,K}/                             ║"
echo "║  Visuals: ${VIS_DIR}/                                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
