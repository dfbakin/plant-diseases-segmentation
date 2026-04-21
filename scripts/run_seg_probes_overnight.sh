#!/usr/bin/env bash
###############################################################################
# Master overnight orchestrator for the SPDNet Localization Capacity Probe.
#
# Chains Phase 1 -> Phase 2 -> Phase 3 sequentially (fail-fast). Each phase
# script is itself idempotent (top-level .DONE marker plus per-probe
# skip-if-exists), so re-running this orchestrator after an interrupt
# resumes from where it stopped.
#
# Usage:
#   bash scripts/run_seg_probes_overnight.sh
#       Full overnight (~10-12 h on a single 5090).
#
#   SMOKE=1 bash scripts/run_seg_probes_overnight.sh
#       Quick end-to-end smoke (~50 min). Final pre-flight before real launch.
#
#   LAMBDA_GRID="0.5 1.0 2.0" bash scripts/run_seg_probes_overnight.sh
#       Widens Phase 2 to a 3-point lambda grid (default is "1.0" only).
#       Adds ~2x to Phase 2 wall time.
#
#   HEARTBEAT_SECS=300 bash scripts/run_seg_probes_overnight.sh
#       Print "still running" every 5 minutes instead of the default 10.
###############################################################################

set -uo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

SMOKE="${SMOKE:-0}"
# Phase 2 lambda grid. Default narrowed from "0.5 1.0 2.0" to "1.0" so
# the overnight orchestrator fits under ~24 h on a single 5090. The wider
# grid is still available via override (e.g. LAMBDA_GRID="0.5 1.0 2.0").
LAMBDA_GRID="${LAMBDA_GRID:-1.0}"
HEARTBEAT_SECS="${HEARTBEAT_SECS:-600}"

if [[ "$SMOKE" == "1" ]]; then
    LOG_DIR="logs/seg_probe_overnight_smoke_$(date +%Y%m%d_%H%M%S)"
    OUT_BASE="outputs/spdnet_plantseg/_smoke"
else
    LOG_DIR="logs/seg_probe_overnight_$(date +%Y%m%d_%H%M%S)"
    OUT_BASE="outputs/spdnet_plantseg"
fi
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

# tee the master log — every line we emit goes to both stdout and disk.
log() {
    echo "[overnight $(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

# ---------------------------------------------------------------------------
# Signal handling.
#
# Each phase is launched in its own process group via setsid so that one
# kill -- -PID brings down the whole subtree (bash + python + child
# workers). Without this the orchestrator can exit while a python child
# keeps running and locks the GPU.
# ---------------------------------------------------------------------------

CHILD_PID=""
HEARTBEAT_PID=""

stop_heartbeat() {
    if [[ -n "$HEARTBEAT_PID" ]]; then
        # Kill the whole heartbeat process group so the orphan-prone `sleep`
        # child dies with its bash parent (otherwise it gets reparented to
        # init and keeps the orchestrator's stdout pipe open for HEARTBEAT_SECS).
        kill -TERM -- "-$HEARTBEAT_PID" 2>/dev/null || true
        sleep 0.2
        kill -KILL -- "-$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    fi
}

start_heartbeat() {
    local label="$1"
    # setsid -> own process group; >>MASTER_LOG instead of inheriting stdout
    # so a stuck heartbeat can never hold the orchestrator's parent pipe open.
    setsid bash -c '
        n=0
        while true; do
            sleep "'"$HEARTBEAT_SECS"'"
            n=$((n + 1))
            echo "[overnight $(date +%H:%M:%S)]   ('"$label"' heartbeat ~$((n * '"$HEARTBEAT_SECS"' / 60)) min in)" >> "'"$MASTER_LOG"'"
        done
    ' </dev/null >/dev/null 2>&1 &
    HEARTBEAT_PID=$!
}

on_signal() {
    local sig="$1"
    log "received SIG${sig} -- terminating active phase (pid $CHILD_PID)"
    stop_heartbeat
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
        sleep 3
        kill -KILL -- "-$CHILD_PID" 2>/dev/null || true
    fi
    log "Aborted at $(date)"
    exit 130
}
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM

# ---------------------------------------------------------------------------
# Single-phase runner.
# Returns the phase script's own exit code.
# ---------------------------------------------------------------------------

run_phase() {
    local label="$1"      # human-readable name (e.g. "phase1")
    local script="$2"     # path to scripts/run_seg_probes_phaseN.sh
    local env_prefix="$3" # extra env to pass to the phase

    local phase_log="$LOG_DIR/${label}.log"
    local t0
    t0=$(date +%s)

    log "=========================================================="
    log " ▶ $label start  ($(date))"
    log "    script: $script"
    log "    log:    $phase_log"
    log "=========================================================="

    # setsid puts the phase in its own process group, which `on_signal`
    # then sends a single -TERM to so every Python child dies with it.
    setsid bash -c "$env_prefix bash $script" > "$phase_log" 2>&1 &
    CHILD_PID=$!

    start_heartbeat "$label"

    # Capture the phase script's real exit code.
    #
    # NOTE: do NOT use `if ! wait $CHILD_PID; then ec=$?; fi` -- inside the
    # then-block `$?` is the negated pipeline result (always 0), which means
    # every failed phase used to be silently reported as ✓. Use the explicit
    # form below instead. Verified empirically:
    #     bash -c 'if ! (exit 7); then echo $?; fi'   # -> 0  (BUG)
    #     bash -c '(exit 7); echo $?'                 # -> 7  (correct)
    local ec=0
    wait "$CHILD_PID" || ec=$?

    stop_heartbeat
    CHILD_PID=""

    local t1
    t1=$(date +%s)
    local dur_min=$(( (t1 - t0) / 60 ))

    if [[ "$ec" -ne 0 ]]; then
        log " ✗ $label FAILED (exit=$ec) after ${dur_min} min"
        log "   Inspect: $phase_log"
        echo ""
        echo "Last 30 lines of $phase_log:"
        tail -30 "$phase_log"
        return "$ec"
    fi
    log " ✓ $label complete in ${dur_min} min"
    return 0
}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

T0=$(date +%s)
log "=========================================================="
log " SPDNet Localization Capacity Probe — Overnight"
log " Mode: SMOKE=$SMOKE  LAMBDA_GRID=\"$LAMBDA_GRID\"  HEARTBEAT_SECS=$HEARTBEAT_SECS"
log " Logs: $LOG_DIR"
log " Out:  $OUT_BASE"
log "=========================================================="

# Fail-fast chain. Each `|| exit` short-circuits the orchestrator the
# moment a phase script exits non-zero, so we never attempt to run a
# downstream phase whose inputs (selected.json / chosen.json) don't yet
# exist. Without this guard, a Phase 1 crash silently lets Phase 2 + 3
# start, immediately fail with "ERROR: ... missing", and produce an
# "ALL PHASES COMPLETE" master.log entry that hides the original failure.
run_phase "phase1" "scripts/run_seg_probes_phase1.sh" "SMOKE=$SMOKE" || {
    log "Aborting orchestrator chain at phase1 (exit=$?)"
    exit 1
}
run_phase "phase2" "scripts/run_seg_probes_phase2.sh" "SMOKE=$SMOKE LAMBDA_GRID='$LAMBDA_GRID'" || {
    log "Aborting orchestrator chain at phase2 (exit=$?)"
    exit 1
}
run_phase "phase3" "scripts/run_seg_probes_phase3.sh" "SMOKE=$SMOKE" || {
    log "Aborting orchestrator chain at phase3 (exit=$?)"
    exit 1
}

T1=$(date +%s)
TOTAL_MIN=$(( (T1 - T0) / 60 ))

# ---------------------------------------------------------------------------
# Final master summary table — picks the salient bits out of each phase's
# SUMMARY.md / selected.json / chosen.json.
# ---------------------------------------------------------------------------

log "=========================================================="
log " ▣ ALL PHASES COMPLETE in ${TOTAL_MIN} min"
log "=========================================================="

python - "$OUT_BASE" "$LOG_DIR/SUMMARY.md" <<'PY' 2>&1 | tee -a "$MASTER_LOG"
"""Aggregate per-phase artefacts into one master SUMMARY.md."""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
dst  = Path(sys.argv[2])

p1_root = base / "seg_probe_phase1"
p2_root = base / "seg_probe_phase2"
p3_root = base / "seg_probe_phase3"

md = ["# SPDNet Localization Capacity Probe — Master Summary", ""]

# ---- Phase 1 ----
sel_path = p1_root / "selected.json"
if sel_path.exists():
    sel = json.loads(sel_path.read_text())
    md += ["## Phase 1 — frozen probe positions selected for Phase 2", ""]
    for ckpt, positions in sel.get("selected_per_ckpt", {}).items():
        md.append(f"- **{ckpt}**: {', '.join(positions)}")
    md.append("")
    md.append(f"Full table: `{p1_root / 'SUMMARY.md'}`")
    md.append("")

# ---- Phase 2 ----
chs_path = p2_root / "chosen.json"
if chs_path.exists():
    c = json.loads(chs_path.read_text())
    def fmt(k):
        v = c.get(k)
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"
    md += [
        "## Phase 2 — chosen design for Phase 3",
        "",
        f"- ckpt: `{c.get('ckpt')}`",
        f"- position: `{c.get('position')}`",
        f"- λ (seg) = {c.get('seg_loss_weight')} | cls = {c.get('cls_loss_weight')}",
        f"- probe IoU (CRF) = {fmt('probe_iou')}",
        f"- chmean baseline  = {fmt('chmean_iou')}",
        f"- chvar baseline   = {fmt('chvar_iou')}",
        f"- cam_cls baseline = {fmt('cam_cls_iou')}",
        f"- composite score S = {fmt('score_S')}",
        "",
        f"Full table: `{p2_root / 'SUMMARY.md'}`",
        "",
    ]

# ---- Phase 3 ----
p3_sum = p3_root / "SUMMARY.md"
if p3_sum.exists():
    md += ["## Phase 3 — from-scratch SPDNet ceiling", ""]
    md += [p3_sum.read_text(), ""]

# ---- Reference benchmarks ----
md += [
    "## Reference benchmarks",
    "",
    "- Best WSSS SPDNet (cam_classifier + CRF, weak supervision):  **32.49%** disease IoU",
    "- Fully-supervised SegNeXt baseline (PlantSeg paper):          **70.10%** disease IoU",
    "",
    "## Recommended Phase 4 reading order",
    "",
    f"1. `{p3_sum}`  — does the from-scratch ceiling validate the architecture?",
    f"2. `{p2_root / 'SUMMARY.md'}`  — does fine-tuning lift the existing ckpt?",
    f"3. `{p1_root / 'SUMMARY.md'}`  — which intermediate features already carry signal?",
]

dst.write_text("\n".join(md) + "\n")
print(open(dst).read())
PY

log "Master summary: $LOG_DIR/SUMMARY.md"
log "Done at $(date)"
exit 0
