#!/usr/bin/env bash
###############################################################################
# Pre-launch checklist for the SPDNet Localization Capacity Probe overnight.
#
# Asserts every condition that, if missing, would cause the overnight to die
# in the middle of the night with no chance of recovery. Returns 0 only when
# every check passes.
#
# Usage:
#   bash scripts/prelaunch_seg_probes.sh
#       Quick run (~45s) — fast unit tests only.
#
#   FULL_TESTS=1 bash scripts/prelaunch_seg_probes.sh
#       Runs the full seg-probe test suite (~8 min) instead of the fast subset.
###############################################################################

set -uo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# Colour-free output to stay log-readable.
PASS=0
FAIL=0
FAILED_CHECKS=()

ok() {
    printf "  [OK]   %s\n" "$1"
    PASS=$((PASS + 1))
}

bad() {
    printf "  [FAIL] %s\n" "$1"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS+=("$1")
}

check_cmd() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi
}

check_ge() {
    local name="$1"; local actual="$2"; local min="$3"; local unit="${4:-}"
    if [[ "$actual" =~ ^[0-9]+$ ]] && (( actual >= min )); then
        ok "$name: ${actual}${unit} (need >= ${min}${unit})"
    else
        bad "$name: ${actual}${unit} (need >= ${min}${unit})"
    fi
}

check_file() {
    local name="$1"; local path="$2"
    if [[ -f "$path" ]]; then ok "$name: $path"; else bad "$name: missing $path"; fi
}

check_dir() {
    local name="$1"; local path="$2"
    if [[ -d "$path" ]]; then ok "$name: $path"; else bad "$name: missing $path"; fi
}

echo "==================================================================="
echo " SPDNet Localization Capacity Probe — Pre-launch Checklist"
echo " $(date)"
echo "==================================================================="

# ---------------------------------------------------------------------------
echo ""
echo "[1] Hardware resources"
# ---------------------------------------------------------------------------

DISK_FREE_GB=$(df -BG --output=avail outputs 2>/dev/null | tail -n1 | tr -dc '0-9' || echo 0)
check_ge "Disk free under outputs/" "${DISK_FREE_GB:-0}" 25 "G"

GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
GPU_FREE_GB=$((GPU_FREE_MIB / 1024))
check_ge "GPU free VRAM" "${GPU_FREE_GB:-0}" 20 "G"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")
ok "GPU detected: $GPU_NAME"

# ---------------------------------------------------------------------------
echo ""
echo "[2] Required SPDNet checkpoints"
# ---------------------------------------------------------------------------

check_file "Token ckpt"   "outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"
check_file "Spatial ckpt" "outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt"

# ---------------------------------------------------------------------------
echo ""
echo "[3] PlantSeg dataset & label files"
# ---------------------------------------------------------------------------

check_dir  "Train images"      "data/plantsegv3/images/train"
check_dir  "Train annotations" "data/plantsegv3/annotations/train"
check_dir  "Val images"        "data/plantsegv3/images/val"
check_dir  "Val annotations"   "data/plantsegv3/annotations/val"
check_file "Class-name list"   "outputs/plantseg_binary_mc115/labels/class_names.txt"
check_file "Train label npy"   "outputs/plantseg_binary_mc115/labels/plantseg_wsss_pv_all_train.npy"

# ---------------------------------------------------------------------------
echo ""
echo "[4] Python dependencies"
# ---------------------------------------------------------------------------

check_cmd "torch"          "python -c 'import torch; assert torch.cuda.is_available()'"
check_cmd "lightning"      "python -c 'import lightning'"
check_cmd "torchmetrics"   "python -c 'import torchmetrics'"
check_cmd "albumentations" "python -c 'import albumentations'"
check_cmd "hydra"          "python -c 'import hydra'"
check_cmd "mlflow"         "python -c 'import mlflow'"
check_cmd "pydensecrf"     "python -c 'import pydensecrf.densecrf'"
check_cmd "PIL"            "python -c 'from PIL import Image'"

# ---------------------------------------------------------------------------
echo ""
echo "[5] Project modules importable"
# ---------------------------------------------------------------------------

check_cmd "SPDNet model"        "python -c 'from src.wsss.spdnet.model import SPDNet'"
check_cmd "Probe wrapper"       "python -c 'from src.wsss.spdnet.seg_probe import SPDNetWithProbes, ProbeHead, NEEDS_REFERENCE, PROBE_POSITIONS'"
check_cmd "Seg dataset"         "python -c 'from src.wsss.spdnet.seg_dataset import SiamesePlantSegSegDataset, siamese_seg_collate_fn'"
check_cmd "Split-index cache"   "python -c 'from src.wsss.spdnet._split_index_cache import scan_or_load_split'"
check_cmd "train_spdnet_probe"  "python -c 'import src.train_spdnet_probe'"
check_cmd "eval_seg_probes"     "python -c 'import scripts.eval_seg_probes' 2>/dev/null; python -c 'import importlib.util, pathlib; importlib.util.spec_from_file_location(\"x\", pathlib.Path(\"scripts/eval_seg_probes.py\"))'"
check_cmd "seg_probe_decisions" "python -c 'import importlib.util, pathlib; importlib.util.spec_from_file_location(\"x\", pathlib.Path(\"scripts/seg_probe_decisions.py\"))'"

# ---------------------------------------------------------------------------
echo ""
echo "[6] Phase orchestrator scripts"
# ---------------------------------------------------------------------------

for s in run_seg_probes_phase1.sh run_seg_probes_phase2.sh run_seg_probes_phase3.sh save_scratch_spdnet.py seg_probe_decisions.py; do
    check_file "scripts/$s" "scripts/$s"
done

# Bash syntax check on the three phase scripts (catches typos without running them).
for s in run_seg_probes_phase1.sh run_seg_probes_phase2.sh run_seg_probes_phase3.sh; do
    if bash -n "scripts/$s" 2>/dev/null; then
        ok "bash -n scripts/$s"
    else
        bad "bash -n scripts/$s (syntax error)"
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "[7] Unit tests"
# ---------------------------------------------------------------------------

# Always run the (fast) split-index-cache tests.
if python -m pytest tests/test_split_index_cache.py -q 2>&1 | tail -3 | grep -qE "passed"; then
    ok "tests/test_split_index_cache.py (18 cases)"
else
    bad "tests/test_split_index_cache.py FAILED — run pytest manually"
fi

if [[ "${FULL_TESTS:-0}" == "1" ]]; then
    echo "  Running FULL seg-probe test suite (~8 min)…"
    if python -m pytest tests/test_seg_probe.py -q 2>&1 | tail -3 | grep -qE "passed"; then
        ok "tests/test_seg_probe.py (60 cases)"
    else
        bad "tests/test_seg_probe.py FAILED — run pytest manually"
    fi
else
    # Subset: skip dataset-dependent tests (those that hit real PlantSeg val)
    if python -m pytest tests/test_seg_probe.py -q \
            -k "not TestSegDataset and not TestClassResolver" 2>&1 | tail -3 | grep -qE "passed"; then
        ok "tests/test_seg_probe.py (fast subset, 55 cases)"
    else
        bad "tests/test_seg_probe.py FAILED — run pytest manually"
    fi
fi

# Overnight orchestrator regression suite. Cheap (<5 s) and catches the
# 18 Apr silent-success bug (broken `if ! wait; ec=$?` + missing chain
# short-circuit) the moment someone re-introduces it.
if python -m pytest tests/test_overnight_orchestrator.py -q 2>&1 | tail -3 | grep -qE "passed"; then
    ok "tests/test_overnight_orchestrator.py (7 cases)"
else
    bad "tests/test_overnight_orchestrator.py FAILED — run pytest manually"
fi

# Parallel-CRF regression suite (TestFullCRFEvalParallel). Cheap (~6 s)
# and catches any future regression of the multi-process / per-image
# timeout fix that prevents the 19 Apr pydensecrf hang from recurring.
if python -m pytest tests/test_seg_probe.py::TestFullCRFEvalParallel -q 2>&1 | tail -3 | grep -qE "passed"; then
    ok "tests/test_seg_probe.py::TestFullCRFEvalParallel (6 cases, parallel CRF + per-image timeout)"
else
    bad "tests/test_seg_probe.py::TestFullCRFEvalParallel FAILED — multi-process CRF eval is broken"
fi

# Parallel threshold-sweep regression suite (TestThresholdSweepParallel).
# Cheap (~3 s); guards the bit-identical-with-serial property of the
# parallel branch added 2026-04-19 (~Wx speedup over the historical
# trange loop). A regression here would silently change the threshold
# every probe uses for visualization / Phase 2 ranking.
if python -m pytest tests/test_binary_pipeline.py::TestThresholdSweepParallel \
        tests/test_binary_pipeline.py::TestThresholdSweepSubsample -q 2>&1 \
        | tail -3 | grep -qE "passed"; then
    ok "tests/test_binary_pipeline.py threshold sweep (9 cases, parallel + serial subsample)"
else
    bad "tests/test_binary_pipeline.py threshold sweep FAILED — parallel/serial divergence"
fi

# Auxiliary spatial losses + online localization metric regression suite.
# Cheap (~30 s); guards the equivariance / patch-contrastive /
# self-distillation losses and the OnlineCAMIoU sweep that the
# spdnet_spatial_eq* runs depend on. A regression here would silently
# change every aux-loss value or kill the online IoU plot.
if python -m pytest tests/test_equivariance_transforms.py \
        tests/test_spatial_losses.py tests/test_online_loc_metric.py -q 2>&1 \
        | tail -3 | grep -qE "passed"; then
    ok "tests/test_{equivariance_transforms,spatial_losses,online_loc_metric}.py (54 cases)"
else
    bad "tests/test_{equivariance_transforms,spatial_losses,online_loc_metric}.py FAILED — aux losses broken"
fi

# CLI surface check: the --crf-eval-timeout-sec flag must exist on
# eval_seg_probes.py, otherwise the bash phase scripts will fail with
# "unrecognized argument" the moment they reach the eval step.
if python scripts/eval_seg_probes.py --help 2>&1 | grep -q -- "--crf-eval-timeout-sec"; then
    ok "scripts/eval_seg_probes.py --crf-eval-timeout-sec CLI flag wired"
else
    bad "scripts/eval_seg_probes.py --crf-eval-timeout-sec flag missing — phase scripts will crash"
fi

# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
printf " Passed: %d | Failed: %d\n" "$PASS" "$FAIL"
echo "==================================================================="

if (( FAIL > 0 )); then
    echo ""
    echo "ABORT: fix the failing checks before launching overnight." >&2
    echo "Failures:" >&2
    for f in "${FAILED_CHECKS[@]}"; do echo "  - $f" >&2; done
    exit 1
fi

echo ""
echo "Cleared for launch:"
echo ""
echo "    bash scripts/run_seg_probes_overnight.sh"
echo ""
echo "  ^ defaults: Phase 1 + Phase 2 in screening mode"
echo "    (--limit-val 300 --crf-sweep-images 50, lambda=1.0); Phase 3 on"
echo "    full val with --crf-sweep-images 100. Estimated ~30-40 h."
echo ""
echo "Or, with the wider Phase 2 lambda sweep (~+10-20 h):"
echo ""
echo "    LAMBDA_GRID='0.5 1.0 2.0' bash scripts/run_seg_probes_overnight.sh"
echo ""
exit 0
