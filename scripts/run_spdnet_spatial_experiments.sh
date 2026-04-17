#!/bin/bash
set -e

# SPDNet Spatial Cross-Attention Fusion Experiments
#
# Sequential runs on RTX 5090 (32GB):
#
#   Run 0: spdnet_spatial_smoke     — 1-epoch sanity check on PS+PV    (~10 min)
#   Run 1: spdnet_spatial_n1_ps     — PlantSeg only, 80 epochs         (~2.4h)
#   Run 2: spdnet_spatial_n1_ps_pv  — PlantSeg + PlantVillage, 80 ep   (~12.5h)
#
#   Total estimated wall time: ~15h
#
# Run 0 validates the pipeline before committing to overnight training.
# If it fails, the script exits immediately (set -e).
#
# Usage:
#   ./scripts/run_spdnet_spatial_experiments.sh
#   RUNS="1 2" ./scripts/run_spdnet_spatial_experiments.sh   # skip smoke
#   MAX_EPOCHS=40 ./scripts/run_spdnet_spatial_experiments.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

MAX_EPOCHS="${MAX_EPOCHS:-80}"
LOG_EVERY="${LOG_EVERY:-200}"
RUNS="${RUNS:-0 1 2}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     SPDNet Spatial Cross-Attention Experiments                 ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Max epochs:    ${MAX_EPOCHS}"
echo "║  Log every:     ${LOG_EVERY} steps"
echo "║  Runs:          ${RUNS}"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  0: smoke  N=1  B=16  PS+PV  1 epoch       ~10 min           ║"
echo "║  1: spatial N=1 B=16  PS     80 epochs      ~2.4h             ║"
echo "║  2: spatial N=1 B=16  PS+PV  80 epochs      ~12.5h            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

run_train() {
    local run_name="$1"
    local epochs="$2"
    local include_pv="$3"
    local log_steps="$4"

    echo ""
    echo "================================================================"
    echo "  Starting: ${run_name}"
    echo "    fusion=spatial  epochs=${epochs}  include_pv=${include_pv}"
    echo "    batch=16  accum=2  refs=1  aug=heavy  log_every=${log_steps}"
    echo "================================================================"

    python src/train_spdnet.py \
        run_name="${run_name}" \
        model.fusion_mode=spatial \
        trainer.max_epochs="${epochs}" \
        trainer.log_every_n_steps="${log_steps}" \
        trainer.accumulate_grad_batches=2 \
        data.batch_size=16 \
        data.num_references=1 \
        data.augmentation=heavy \
        data.include_plantvillage="${include_pv}" \
        data.num_workers=8

    echo "  Finished: ${run_name}"
}

# --- Run 0: Smoke test (1 epoch, PS+PV) ---
if echo "${RUNS}" | grep -qw "0"; then
    run_train "spdnet_spatial_smoke" 1 true 10
    echo "  Smoke test passed — proceeding with full runs"
    rm -rf outputs/spdnet_plantseg/spdnet_spatial_smoke
fi

# --- Run 1: Spatial fusion on PlantSeg only ---
if echo "${RUNS}" | grep -qw "1"; then
    run_train "spdnet_spatial_n1_ps" "${MAX_EPOCHS}" false "${LOG_EVERY}"
fi

# --- Run 2: Spatial fusion on PlantSeg + PlantVillage ---
if echo "${RUNS}" | grep -qw "2"; then
    run_train "spdnet_spatial_n1_ps_pv" "${MAX_EPOCHS}" true "${LOG_EVERY}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  All requested runs complete.                                 ║"
echo "║  Outputs: outputs/spdnet_plantseg/<run_name>/                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
