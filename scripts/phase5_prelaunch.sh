#!/usr/bin/env bash
###############################################################################
# Phase 5 pre-launch checks.
#
# Quick green/red sanity pass before you commit a multi-hour GPU run. Only
# runs fast things (<2 min total):
#
#   1. Unit tests for the Phase 5 code (gradient CAM, screening, smoke).
#   2. Checkpoint existence (d4_ac_safe; highres warnings when missing).
#   3. Disk headroom on the output filesystem.
#   4. NVIDIA + CUDA probe (or graceful "no GPU" warning).
#   5. Short eval-d4-localization config sanity (just imports).
#
# Exits non-zero if any must-pass check fails (tests, d4_ac_safe ckpt,
# <50 GB of free disk). Warnings (missing highres ckpt, no GPU) do not
# fail the script because those are expected at different points in
# the rollout.
#
# Usage:
#     bash scripts/phase5_prelaunch.sh
###############################################################################
set -uo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

readonly CKPT_D4AC="outputs/spdnet_aux_losses/spdnet_spatial_d4_ac_safe_warmstart_20260427/checkpoints/last.ckpt"
# Highres checkpoint is only created by Stage C (see launch guide). Missing
# on Day 0 is EXPECTED -- we just warn on absence.
readonly CKPT_HIGHRES_GLOB="outputs/spdnet_plantseg/spdnet_highres896_d4_ac_safe_*/checkpoints/best_cam_iou.ckpt"
readonly MIN_FREE_GIB=50.0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf "${GREEN}[ OK   ]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[ WARN ]${NC} %s\n" "$*"; }
fail() { printf "${RED}[ FAIL ]${NC} %s\n" "$*"; }

RESULTS_FAILED=0
RESULTS_WARNED=0

mark_fail() { fail "$@"; RESULTS_FAILED=$((RESULTS_FAILED + 1)); }
mark_warn() { warn "$@"; RESULTS_WARNED=$((RESULTS_WARNED + 1)); }

echo "=============================================================="
echo " Phase 5 pre-launch checks"
echo "=============================================================="
echo

# -----------------------------------------------------------------
# 1. Unit tests for Phase 5 code.
# -----------------------------------------------------------------
echo "--- 1. Phase 5 unit tests ---"
# The tests live in a single subprocess so a pass here means every new
# module is wired correctly; individual test files take <15 s total.
TEST_PATHS=(
    "tests/test_gradient_cam_methods.py"
    "tests/test_screen_cam_methods.py"
    "tests/test_smoke_spdnet_highres.py"
    "tests/test_spdnet.py::TestGradientCamDispatch"
    "tests/test_spdnet.py::TestSPDNetTrainerConfig"
)
missing_tests=()
for tp in "${TEST_PATHS[@]}"; do
    # Only path portion (before ::) is checked on disk.
    path="${tp%%::*}"
    if [[ ! -f "$path" ]]; then
        missing_tests+=("$tp")
    fi
done
if (( ${#missing_tests[@]} > 0 )); then
    mark_fail "Missing test files: ${missing_tests[*]}"
else
    if uv run python -m pytest "${TEST_PATHS[@]}" -q --tb=line 2>&1 | tail -8; then
        ok "All Phase 5 unit tests pass"
    else
        mark_fail "Phase 5 unit tests failed (see output above)"
    fi
fi
echo

# -----------------------------------------------------------------
# 2. Checkpoint existence.
# -----------------------------------------------------------------
echo "--- 2. Checkpoints ---"
if [[ -f "$CKPT_D4AC" ]]; then
    size=$(du -h "$CKPT_D4AC" | awk '{print $1}')
    ok "d4_ac_safe present: $CKPT_D4AC  (${size})"
else
    mark_fail "d4_ac_safe MISSING: $CKPT_D4AC"
fi

# shellcheck disable=SC2086
highres_matches=( $(ls $CKPT_HIGHRES_GLOB 2>/dev/null) )
if (( ${#highres_matches[@]} > 0 )); then
    ok "highres checkpoint(s) present: ${#highres_matches[@]}"
    for m in "${highres_matches[@]}"; do
        printf "         %s\n" "$m"
    done
else
    mark_warn "highres checkpoint not yet present (expected before Stage C finishes): $CKPT_HIGHRES_GLOB"
fi
echo

# -----------------------------------------------------------------
# 3. Disk headroom on the workspace partition.
# -----------------------------------------------------------------
echo "--- 3. Disk headroom ---"
avail_gib=$(df -BG --output=avail "$REPO_ROOT" | tail -n 1 | tr -dc '0-9.')
if [[ -n "$avail_gib" ]]; then
    if (( $(echo "$avail_gib < $MIN_FREE_GIB" | bc -l) )); then
        mark_fail "Free disk ${avail_gib}G < required ${MIN_FREE_GIB}G on $(df "$REPO_ROOT" | tail -n 1 | awk '{print $1}')"
    else
        ok "Free disk ${avail_gib}G >= required ${MIN_FREE_GIB}G"
    fi
else
    mark_warn "Could not parse df output; skipping disk check"
fi
echo

# -----------------------------------------------------------------
# 4. NVIDIA + CUDA visibility.
# -----------------------------------------------------------------
echo "--- 4. GPU visibility ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | head -4; then
        ok "nvidia-smi reports at least one GPU"
    else
        mark_warn "nvidia-smi returned an error (driver / device problem)"
    fi
else
    mark_warn "nvidia-smi not in PATH -- no GPU? Fine for launch-guide review, not for Stage C/D."
fi

uv run python -c "
import torch, sys
have = torch.cuda.is_available()
print(f'torch.cuda.is_available: {have}')
if have:
    print(f'  device_count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  [{i}] {name}: {total:.1f} GiB')
    sys.exit(0)
sys.exit(0 if have else 3)
" && ok "torch.cuda reports GPU(s)" || mark_warn "torch.cuda.is_available() is False"
echo

# -----------------------------------------------------------------
# 5. Eval-script + screening-script import smoke.
# -----------------------------------------------------------------
echo "--- 5. Phase 5 entry-point import smoke ---"
if uv run python -c "
from scripts.eval_d4_localization import _sweep_crf_fullres, FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT
from scripts.screen_cam_methods import ALLOWED_MODES, main as screen_main
from scripts.smoke_test_spdnet_highres import _run_one_step
from scripts.phase5_update_summary import build_summary
from scripts.phase5_generate_report import render
from src.wsss.spdnet.gradient_cam_methods import list_methods
print('eval_d4_localization.FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT =',
      FULLRES_SWEEP_CFG_TIMEOUT_SEC_DEFAULT)
print('gradient CAM methods:', list_methods())
print('screen allowed modes count:', len(ALLOWED_MODES))
" 2>&1; then
    ok "All Phase 5 entry points importable"
else
    mark_fail "Import smoke failed (see error above)"
fi
echo

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
echo "=============================================================="
if (( RESULTS_FAILED > 0 )); then
    fail "${RESULTS_FAILED} MUST-FIX item(s), ${RESULTS_WARNED} warning(s)."
    fail "Do NOT start long-running stages until failures are resolved."
    echo "=============================================================="
    exit 1
fi
if (( RESULTS_WARNED > 0 )); then
    warn "No blockers, but ${RESULTS_WARNED} warning(s) above -- review before launching expensive stages."
else
    ok "All checks passed. You can safely proceed with the launch guide."
fi
echo "=============================================================="
exit 0
