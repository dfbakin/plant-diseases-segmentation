#!/usr/bin/env bash
###############################################################################
# Phase 3 — From-scratch supervised SPDNet (the "ceiling" experiment).
#
# Reads outputs/spdnet_plantseg/seg_probe_phase2/chosen.json to know the
# winning (position, lambda). Materialises a fresh SPDNet (ImageNet
# ResNet50, rest random) and trains it with the multi-task loss for 80
# epochs, then evaluates the resulting probe + baselines.
#
# This bounds the IoU achievable by SPDNet's architecture under full GT
# supervision and is the input to the Phase 4 go/no-go decision.
#
# Modes:
#   bash scripts/run_seg_probes_phase3.sh
#       Full Phase 3 -- ~8-9 h on a single 5090. (80 epoch train +
#       full-val eval with --crf-sweep-images 100.)
#       Phase 3 *intentionally* runs on the full 1247-image val set --
#       this is the headline ceiling number, not a screening run, so
#       precision matters more than wall time.
#
#   SMOKE=1 bash scripts/run_seg_probes_phase3.sh
#       Tiny dataset, 1 epoch, fewer eval imgs. ~3 min.
###############################################################################

set -euo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

CLS_LOSS_WEIGHT=1.0
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
    P2_ROOT="outputs/spdnet_plantseg/_smoke/seg_probe_phase2"
    OUT_ROOT="outputs/spdnet_plantseg/_smoke/seg_probe_phase3"
    EXTRA_TRAIN_OVERRIDES=(
        "data.limit_train=50"
        "data.limit_val=25"
        "trainer.max_epochs=1"
        "data.num_workers=0"
        "data.batch_size=4"
    )
    EVAL_FLAGS=(--smoke --crf-sweep-images 30 --viz-count 5 --crf-workers 4 --crf-eval-timeout-sec 60)
    echo "[phase3] SMOKE mode -- writing under $OUT_ROOT"
else
    P2_ROOT="outputs/spdnet_plantseg/seg_probe_phase2"
    OUT_ROOT="outputs/spdnet_plantseg/seg_probe_phase3"
    EXTRA_TRAIN_OVERRIDES=(
        "trainer.max_epochs=80"
        "model.head_lr=1e-3"
        "model.backbone_lr=1e-4"
        "data.train_aug=true"
    )
    # --cleanup-seeds drops the ~4.5 GB of *_seeds/ npy files after
    # eval.json + viz/ are written. Phase 3 only runs once so the saving
    # is modest, but we keep the behaviour consistent across phases.
    #
    # Phase 3 deliberately does NOT use --limit-val: this is the ceiling
    # measurement, the headline number that's compared against the
    # current WSSS best (32.49%) and the SegNeXt upper bound (70.1%).
    # We trim --crf-sweep-images 200 -> 100 (still a robust sweep on
    # full val) to recover ~10 min/baseline; full CRF eval still uses
    # all 1247 images.
    # --crf-workers 8 + --crf-eval-timeout-sec 300: same multi-process +
    # per-image hard cap that Phase 2 uses (see scripts/eval_seg_probes.py
    # ::_full_crf_eval). Phase 3 runs on full val (1247 imgs) and is
    # especially exposed to pathological pydensecrf hangs given the bigger
    # batch size.
    EVAL_FLAGS=(--crf-sweep-images 100 --viz-count 25 --crf-workers 8 --crf-eval-timeout-sec 300 --cleanup-seeds)
fi

mkdir -p "$OUT_ROOT" logs
LOG_FILE="logs/seg_probe_phase3_$(date +%Y%m%d_%H%M%S).log"
DONE_MARKER="$OUT_ROOT/.DONE"

echo "============================================================"
echo "  SPDNet Phase 3 — From-scratch Supervised"
echo "  Started:    $(date)"
echo "  Phase2 src: $P2_ROOT"
echo "  Out root:   $OUT_ROOT"
echo "  Logfile:    $LOG_FILE"
echo "============================================================"

if [[ -f "$DONE_MARKER" ]]; then
    echo "[phase3] $DONE_MARKER already exists -- nothing to do."
    exit 0
fi

if [[ ! -f "$P2_ROOT/chosen.json" ]]; then
    echo "ERROR: Phase 2 chosen.json missing at $P2_ROOT/chosen.json" >&2
    echo "Run Phase 2 first (or its decision gate)." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Read winning (position, lambda) from Phase 2 chosen.json
# ----------------------------------------------------------------------------

read_chosen() {
    python -c "
import json
c = json.load(open('${P2_ROOT}/chosen.json'))
print(c['position'], c['seg_loss_weight'])
"
}

read POSITION LAMBDA <<<"$(read_chosen)"
echo "  Chosen design: position=$POSITION  lambda=$LAMBDA  cls_weight=$CLS_LOSS_WEIGHT"

# ----------------------------------------------------------------------------
# Materialise scratch SPDNet (ImageNet ResNet50, rest random)
# ----------------------------------------------------------------------------

SCRATCH_CKPT="$OUT_ROOT/scratch_init.pt"
TAG="from_scratch_spatial"

if [[ ! -f "$SCRATCH_CKPT" ]]; then
    echo "  Creating scratch SPDNet at $SCRATCH_CKPT..."
    python scripts/save_scratch_spdnet.py \
        --output "$SCRATCH_CKPT" \
        --fusion-mode spatial \
        --num-classes 115
else
    echo "  $SCRATCH_CKPT already exists -- reusing."
fi

# ----------------------------------------------------------------------------
# Train + evaluate
# ----------------------------------------------------------------------------

t0=$(date +%s)
PROBE_DIR="$OUT_ROOT/$TAG/$POSITION"
HEAD_PATH="$PROBE_DIR/head.pt"
EVAL_PATH="$PROBE_DIR/eval.json"

if [[ -f "$EVAL_PATH" ]]; then
    echo "  $EVAL_PATH exists -- skipping training and eval."
elif [[ -f "$HEAD_PATH" ]]; then
    echo "  $HEAD_PATH exists -- skipping training, evaluating only."
else
    echo "  Training from-scratch SPDNet + probe (joint multi-task)..."
    # Hydra needs the checkpoint value single-quoted because the scratch
    # checkpoint path may contain "=" signs (mirrors Lightning convention).
    python src/train_spdnet_probe.py \
        ckpt_tag="$TAG" \
        "checkpoint='$SCRATCH_CKPT'" \
        phase="phase3" \
        output_dir="$OUT_ROOT/\${ckpt_tag}/\${model.position}" \
        model.position="$POSITION" \
        model.freeze_backbone=false \
        model.seg_loss_weight="$LAMBDA" \
        model.cls_loss_weight="$CLS_LOSS_WEIGHT" \
        "${EXTRA_TRAIN_OVERRIDES[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

if [[ ! -f "$EVAL_PATH" ]]; then
    echo "  Evaluating from-scratch SPDNet + probe..."
    python scripts/eval_seg_probes.py \
        --probe-dir "$PROBE_DIR" \
        --checkpoint "$SCRATCH_CKPT" \
        "${EVAL_FLAGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

# ----------------------------------------------------------------------------
# Final summary -- one-shot per run
# ----------------------------------------------------------------------------

python - "$EVAL_PATH" "$OUT_ROOT/SUMMARY.md" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if not src.exists():
    print(f"WARNING: missing {src}, no SUMMARY.md written.")
    sys.exit(0)
e = json.load(open(src))
def _fmt(v):
    return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"
md = ["# Phase 3 — From-scratch SPDNet ceiling", "",
      f"- position: {e['position']}", f"- channels_in: {e['channels_in']}",
      f"- fusion_mode: {e['fusion_mode']}", "",
      "| Variant | Disease IoU (CRF) |",
      "|---|---:|",
      f"| Probe | {_fmt(e['probe_iou'])} |",
      f"| chmean | {_fmt(e.get('chmean_iou'))} |",
      f"| chvar | {_fmt(e.get('chvar_iou'))} |",
      f"| cam_cls | {_fmt(e.get('cam_cls_iou'))} |",
      f"| Score S | {_fmt(e['score_S'])} |", "",
      "Reference benchmarks:",
      "- Current best WSSS SPDNet: 32.49% disease IoU (cam_classifier + CRF)",
      "- Fully supervised SegNeXt: 70.1% disease IoU"]
dst.write_text("\n".join(md) + "\n")
print(open(dst).read())
PY

touch "$DONE_MARKER"

t1=$(date +%s)
echo ""
echo "============================================================"
echo "  Phase 3 complete in $((t1 - t0))s -- $(date)"
echo "  Marker:  $DONE_MARKER"
echo "  Summary: $OUT_ROOT/SUMMARY.md"
echo "============================================================"
