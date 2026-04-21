#!/usr/bin/env bash
###############################################################################
# Phase 2 — Targeted unfrozen fine-tunes on the positions selected by Phase 1.
#
# For each (ckpt, position) listed in
#   <PHASE1_ROOT>/selected.json:selected_per_ckpt[<ckpt>]
# trains the SPDNet (backbone unfrozen, joint multi-task loss) for each
# lambda in the grid. lambda multiplies the segmentation loss; the
# classification loss weight is fixed at 1.
#
# Output layout:
#   outputs/spdnet_plantseg/seg_probe_phase2/<ckpt>/<pos>/seg<L>_cls<C>/
#     ├── head.pt
#     ├── spdnet_finetuned.pt
#     ├── eval.json
#     └── viz/
#
# Skip-if-exists per (ckpt, position, lambda).
#
# Modes:
#   bash scripts/run_seg_probes_phase2.sh
#       Default: lambda=1.0 only, --limit-val 300, --crf-sweep-images 50.
#       Designed to fit within ~12-15 h on a single 5090.
#   SMOKE=1 bash scripts/run_seg_probes_phase2.sh
#   LAMBDA_GRID="0.5 1.0 2.0" bash scripts/run_seg_probes_phase2.sh
#       Override the lambda grid (e.g. for Phase 4 winner sweeps).
###############################################################################

set -euo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

CKPT_TOKEN="outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt"
CKPT_SPATIAL="outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt"

TOKEN_TAG="token_n1_heavy"
SPATIAL_TAG="spatial_n1_ps_pv"

CLS_LOSS_WEIGHT=1.0
# Compute-budget tradeoff: the original 3-point grid (0.5, 1.0, 2.0) gave
# ~3x the runs and proved hard to fit on a single 5090 in <24 h. The
# screening-mode grid is a single point at lambda=1.0 (equal weighting of
# seg + cls loss); winners get a per-position lambda sweep in Phase 4.
LAMBDA_GRID_DEFAULT="1.0"
LAMBDA_GRID="${LAMBDA_GRID:-$LAMBDA_GRID_DEFAULT}"

SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
    P1_ROOT="outputs/spdnet_plantseg/_smoke/seg_probe_phase1"
    OUT_ROOT="outputs/spdnet_plantseg/_smoke/seg_probe_phase2"
    EXTRA_TRAIN_OVERRIDES=(
        "data.limit_train=50"
        "data.limit_val=25"
        "trainer.max_epochs=1"
        "data.num_workers=0"
        "data.batch_size=4"
    )
    # --crf-eval-timeout-sec 60 in smoke -- short cap matches the tiny
    # synthetic dataset, surfaces any worker-init regression quickly.
    EVAL_FLAGS=(--smoke --crf-sweep-images 30 --viz-count 5 --crf-workers 4 --crf-eval-timeout-sec 60)
    if [[ -z "${LAMBDA_GRID_OVERRIDE:-}" ]]; then
        LAMBDA_GRID="1.0"
    fi
    echo "[phase2] SMOKE mode -- lambdas=$LAMBDA_GRID, root=$OUT_ROOT"
else
    P1_ROOT="outputs/spdnet_plantseg/seg_probe_phase1"
    OUT_ROOT="outputs/spdnet_plantseg/seg_probe_phase2"
    # max_epochs bumped 10 -> 15 in line with the Phase 1 budget bump
    # (5 -> 20). Phase 2 has unfrozen backbone + multi-task loss, so it
    # needs head + body to co-converge -- 10 epochs was tight for that.
    EXTRA_TRAIN_OVERRIDES=(
        "trainer.max_epochs=15"
        "model.head_lr=1e-4"
        "model.backbone_lr=1e-5"
    )
    # --cleanup-seeds drops the ~4.5 GB / run of *_seeds/ npy files after
    # eval.json + viz/ are written. Seeds are reproducible from head.pt +
    # spdnet_finetuned.pt, so reruns can regenerate them with --force.
    #
    # --limit-val 300 keeps Phase 2 in screening mode: the same 300-image
    # subset (deterministic, seed=1234) used in Phase 1, so Phase 2's gain
    # is directly comparable to Phase 1's frozen baseline. Phase 4
    # re-evaluates the winners on full val.
    # --crf-workers 8 parallelises the full-val CRF pass (was serial pre
    # 2026-04-19; serial path deadlocked on a single 1.8 MP zucchini image).
    # --crf-eval-timeout-sec 300 caps any individual pydensecrf inference at
    # 5 min -- ~5x our slowest-ever measurement (16 MP image @ ~55 s) so any
    # image that hits the cap is hung, not slow. Skipped images are reported
    # in master.log + counted in eval.json for transparency.
    EVAL_FLAGS=(--crf-sweep-images 50 --viz-count 25 --crf-workers 8 --crf-eval-timeout-sec 300 --cleanup-seeds --limit-val 300)
fi

mkdir -p "$OUT_ROOT" logs
LOG_FILE="logs/seg_probe_phase2_$(date +%Y%m%d_%H%M%S).log"
DONE_MARKER="$OUT_ROOT/.DONE"

echo "============================================================"
echo "  SPDNet Phase 2 — Targeted Unfrozen Fine-Tunes"
echo "  Started:    $(date)"
echo "  Phase1 src: $P1_ROOT"
echo "  Out root:   $OUT_ROOT"
echo "  Lambdas:    $LAMBDA_GRID  cls_weight=$CLS_LOSS_WEIGHT"
echo "  Logfile:    $LOG_FILE"
echo "============================================================"

if [[ -f "$DONE_MARKER" ]]; then
    echo "[phase2] $DONE_MARKER already exists -- nothing to do."
    exit 0
fi

if [[ ! -f "$P1_ROOT/selected.json" ]]; then
    echo "ERROR: Phase 1 selected.json missing at $P1_ROOT/selected.json" >&2
    echo "Run Phase 1 first (or its decision gate)." >&2
    exit 1
fi

resolve_ckpt() {
    case "$1" in
        "$TOKEN_TAG")    echo "$CKPT_TOKEN" ;;
        "$SPATIAL_TAG")  echo "$CKPT_SPATIAL" ;;
        *) echo ""; ;;
    esac
}

# ----------------------------------------------------------------------------
# One (ckpt_tag, position, lambda) run
# ----------------------------------------------------------------------------

run_one() {
    local tag="$1"; local ckpt="$2"; local pos="$3"; local lam="$4"

    local subdir="seg${lam}_cls${CLS_LOSS_WEIGHT}"
    local out_dir="$OUT_ROOT/$tag/$pos/$subdir"
    local head_path="$out_dir/head.pt"
    local eval_path="$out_dir/eval.json"

    echo ""
    echo "=== [$tag/$pos/seg=$lam cls=$CLS_LOSS_WEIGHT] ==========="
    echo "  out_dir: $out_dir"

    if [[ -f "$eval_path" ]]; then
        echo "  $eval_path exists -- skipping training and eval."
        return 0
    fi

    if [[ ! -f "$head_path" ]]; then
        echo "  Training (unfrozen) probe + backbone..."
        # Hydra needs the checkpoint value single-quoted because Lightning
        # ModelCheckpoint produced filenames with multiple "=" signs.
        python src/train_spdnet_probe.py \
            ckpt_tag="$tag" \
            "checkpoint='$ckpt'" \
            phase="phase2" \
            output_dir="$OUT_ROOT/\${ckpt_tag}/\${model.position}" \
            model.position="$pos" \
            model.freeze_backbone=false \
            model.seg_loss_weight="$lam" \
            model.cls_loss_weight="$CLS_LOSS_WEIGHT" \
            "${EXTRA_TRAIN_OVERRIDES[@]}"
    else
        echo "  $head_path exists -- skipping training."
    fi

    echo "  Evaluating probe + baselines (with fine-tuned SPDNet)..."
    python scripts/eval_seg_probes.py \
        --probe-dir "$out_dir" \
        --checkpoint "$ckpt" \
        "${EVAL_FLAGS[@]}"
}

# ----------------------------------------------------------------------------
# Parse Phase-1 selection and execute
# ----------------------------------------------------------------------------

t0=$(date +%s)

mapfile -t lines < <(
    python -c "
import json, sys
sel = json.load(open('${P1_ROOT}/selected.json'))['selected_per_ckpt']
for ckpt, positions in sel.items():
    for p in positions:
        print(f'{ckpt} {p}')
"
)

if [[ ${#lines[@]} -eq 0 ]]; then
    echo "[phase2] no (ckpt, position) pairs to run -- selected.json was empty." >&2
    exit 1
fi

echo "[phase2] $((${#lines[@]} * $(echo "$LAMBDA_GRID" | wc -w))) total runs:"
for line in "${lines[@]}"; do
    for lam in $LAMBDA_GRID; do
        echo "  $line $lam"
    done
done

for line in "${lines[@]}"; do
    tag=$(awk '{print $1}' <<<"$line")
    pos=$(awk '{print $2}' <<<"$line")
    ckpt=$(resolve_ckpt "$tag")
    if [[ -z "$ckpt" ]]; then
        echo "WARNING: unknown ckpt tag $tag -- skipping" >&2
        continue
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "WARNING: ckpt $ckpt not on disk -- skipping" >&2
        continue
    fi
    for lam in $LAMBDA_GRID; do
        run_one "$tag" "$ckpt" "$pos" "$lam" 2>&1 | tee -a "$LOG_FILE"
    done
done

echo ""
echo "============================================================"
echo "  Phase 2 decision gate"
echo "============================================================"
python scripts/seg_probe_decisions.py phase2 --root "$OUT_ROOT" 2>&1 | tee -a "$LOG_FILE"

touch "$DONE_MARKER"

t1=$(date +%s)
echo ""
echo "============================================================"
echo "  Phase 2 complete in $((t1 - t0))s -- $(date)"
echo "  Marker:  $DONE_MARKER"
echo "  Summary: $OUT_ROOT/SUMMARY.md"
echo "============================================================"
