#!/bin/bash
set -e

# SPDNet Architecture Fix + Multi-Reference Experiment Sweep
#
# 6 sequential training runs on RTX 5090 (32GB):
#
#   Run 1: spdnet_fix_n1_heavy    — N=1, B=16, accum=2, heavy aug      (~2.1h)
#   Run 2: spdnet_fix_n3_heavy    — N=3, B=16, accum=2, heavy aug      (~3.7h)
#   Run 3: spdnet_fix_n5_heavy    — N=5, B=10, accum=4, heavy aug      (~5.2h)
#   Run 4: spdnet_fix_n8_heavy    — N=8, B=8,  accum=4, heavy aug      (~7.8h)
#   Run 5: spdnet_fix_n3_light    — N=3, B=16, accum=2, light aug      (~3.7h)
#   Run 6: spdnet_fix_n3_minimal  — N=3, B=16, accum=2, minimal aug    (~3.7h)
#
#   Total estimated wall time: ~26.2 hours (sequential)
#
# Usage:
#   ./scripts/run_spdnet_experiments.sh
#   RUNS="1 2 3" ./scripts/run_spdnet_experiments.sh    # subset
#   MAX_EPOCHS=40 ./scripts/run_spdnet_experiments.sh   # override epochs

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

MAX_EPOCHS="${MAX_EPOCHS:-80}"
LOG_EVERY="${LOG_EVERY:-200}"
RUNS="${RUNS:-1 2 3 4 5 6}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SPDNet Fix + Multi-Reference Experiments              ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Max epochs:    ${MAX_EPOCHS}"
echo "║  Log every:     ${LOG_EVERY} steps"
echo "║  Runs:          ${RUNS}"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  1: N=1  B=16 accum=2  heavy    ~2.1h                        ║"
echo "║  2: N=3  B=16 accum=2  heavy    ~3.7h                        ║"
echo "║  3: N=5  B=10 accum=4  heavy    ~5.2h                        ║"
echo "║  4: N=8  B=8  accum=4  heavy    ~7.8h                        ║"
echo "║  5: N=3  B=16 accum=2  light    ~3.7h                        ║"
echo "║  6: N=3  B=16 accum=2  minimal  ~3.7h                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"

run_train() {
    local run_name="$1"
    local num_refs="$2"
    local batch_size="$3"
    local accum="$4"
    local augmentation="$5"

    echo ""
    echo "================================================================"
    echo "  Starting: ${run_name}"
    echo "    refs=${num_refs}  batch=${batch_size}  accum=${accum}  aug=${augmentation}"
    echo "    epochs=${MAX_EPOCHS}  log_every=${LOG_EVERY}"
    echo "================================================================"

    python src/train_spdnet.py \
        run_name="${run_name}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        trainer.log_every_n_steps="${LOG_EVERY}" \
        trainer.accumulate_grad_batches="${accum}" \
        data.batch_size="${batch_size}" \
        data.num_references="${num_refs}" \
        data.augmentation="${augmentation}" \
        data.num_workers=8

    echo "  Finished: ${run_name}"
}

# --- Run 1: Validate architecture fix alone (single ref, heavy aug) ---
if echo "${RUNS}" | grep -qw "1"; then
    run_train "spdnet_fix_n1_heavy" 1 16 2 heavy
fi

# --- Run 2: Multi-ref benefit with N=3 (heavy aug) ---
if echo "${RUNS}" | grep -qw "2"; then
    run_train "spdnet_fix_n3_heavy" 3 16 2 heavy
fi

# --- Run 3: Multi-ref benefit with N=5 (heavy aug, reduced batch) ---
if echo "${RUNS}" | grep -qw "3"; then
    run_train "spdnet_fix_n5_heavy" 5 10 4 heavy
fi

# --- Run 4: Multi-ref benefit with N=8 (heavy aug, minimal batch) ---
if echo "${RUNS}" | grep -qw "4"; then
    run_train "spdnet_fix_n8_heavy" 8 8 4 heavy
fi

# --- Run 5: N=3 with light augmentation ---
if echo "${RUNS}" | grep -qw "5"; then
    run_train "spdnet_fix_n3_light" 3 16 2 light
fi

# --- Run 6: N=3 with minimal augmentation ---
if echo "${RUNS}" | grep -qw "6"; then
    run_train "spdnet_fix_n3_minimal" 3 16 2 minimal
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  All requested runs complete.                                 ║"
echo "║  Outputs: outputs/spdnet_plantseg/<run_name>/                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
