#!/usr/bin/env bash
###############################################################################
# SPDNet Auxiliary Spatial Losses — overnight launcher
#
# Presets (see PRESET_LAMBDAS + PRESET_EXTRA_HYDRA below; the lambda format is
# "lambda_eq lambda_con lambda_distill distill_warmup con_warmup_start con_warmup_epochs"):
#
#   eq                 equivariance only                                  ~13 h
#   eq_con             equivariance + contrastive (no warmup)             ~14 h (old HEADLINE)
#   eq_con_distill     all three, distill warmup 10 ep                    ~16 h
#   eq_con_warmup      eq_con from scratch with L_con ramp (start=14,     ~14 h (F)
#                        ramp=7). Rationale: at epoch 14 the eq-only
#                        run reaches val/mAP ~0.6; classifier has enough
#                        spatial signal for L_con to shape without
#                        hijacking early convergence.
#   eq_con_warmstart   eq_con resumed from a converged eq-only ckpt,
#                        L_con ramp start=0, ramp=5. Short (25 epochs)
#                        by default -- classifier already converged.      ~5-7 h (C)
#
#   --- D1/D2/D3: diagnostic interventions from RESEARCH_CONTEXT.md §5.13.7 ---
#
#   d1_ac_warmstart          warmstart + L_ac (attention concentration
#                            regulariser, lambda_ac=0.5). Replaces L_eq
#                            entirely (lambda_eq=0). Breaks the uniform
#                            attention-map fixed point that L_eq cannot
#                            escape. Needs --from-checkpoint. ~4 h/40 ep
#
#   d2_mask_warmstart        warmstart + L_mask (pseudo-mask CAM supervision,
#                            lambda_mask=1.0) with chvar-based positives
#                            (alpha=0.25) intersected with the classifier's
#                            own top-alpha CAM; negatives = chvar bottom 50 %.
#                            All other aux losses off. ~4 h/40 ep
#
#   d3_d2plus_union_warmstart  d2_mask_warmstart + L_con with union anchors
#                            (anchor_source=union_cls_chvar, lambda_con=0.5,
#                            con warmup 0..5). Combines direct pseudo-mask
#                            supervision with contrastive learning that
#                            broadens anchor coverage beyond classifier-only.
#                            ~5 h/40 ep
#
#   --- D4: magnitude-rebalanced attention + union pseudo-mask (RQ1, RQ2, RQ5) ---
#
#   d4_main_warmstart        Headline D4: replace L_ac with L_marg_H
#                            (lambda_marg_H=0.15, beta=0.25) + L_mask with
#                            mask_combiner="union" (lambda_mask=0.10,
#                            alpha=0.25, beta=0.50). L_eq, L_con, L_dist
#                            all off. ~4 h/40 ep
#
#   d4_attn_only_warmstart   Ablation: L_marg_H alone (no L_mask). Isolates
#                            the attention-shaping contribution.
#
#   d4_ac_safe_warmstart     Pure-magnitude hypothesis (H1): use L_ac at a
#                            RQ1-calibrated weight (lambda_ac=0.05)
#                            alongside L_mask(union). Tests whether plain
#                            magnitude rebalancing without L_marg_H is
#                            enough.
#
#   d4_int_warmstart         A/B twin of d4_main_warmstart with
#                            mask_combiner="intersection" instead of
#                            "union". Isolates the RQ5 teacher-combiner
#                            hypothesis.
#
# Default dataset is PlantSeg + PlantVillage (33,877 train imgs, ~4.3x PS-only)
# so results are directly comparable to the spdnet_spatial_n1_ps_pv probe
# baseline (62% IoU upper bound). Set INCLUDE_PV=false for PS-only debugging
# (~3 h / preset). All presets default to fusion_mode=spatial, 80 epochs,
# batch=16, accum=2, refs=1, augmentation=heavy. Online localization metric is
# ON by default (val/cam_iou_best{,_thr,_auc} every epoch).
#
# Usage:
#   bash scripts/run_spdnet_aux_losses_experiments.sh                       # eq, eq_con, eq_con_distill (backward compat)
#   bash scripts/run_spdnet_aux_losses_experiments.sh --preset eq_con_warmup
#   bash scripts/run_spdnet_aux_losses_experiments.sh --preset eq_con_warmstart \
#       --from-checkpoint 'outputs/.../best.ckpt' \
#       MAX_EPOCHS=25
#   bash scripts/run_spdnet_aux_losses_experiments.sh --dry-run --preset eq_con_warmup
#
# NOTE on --from-checkpoint: the path is auto single-quoted before going to
# Hydra so that it survives '=' and '/' characters produced by our
# ModelCheckpoint filename template (e.g. ``epoch=72-val_mAP=val/mAP=0.8615``).
# Warmstart semantics are "weights only, fresh optimizer/scheduler/epoch" --
# see train_spdnet.py.
#
# Logs:    logs/spdnet_aux_losses_<preset>_<timestamp>.log
# Outputs: outputs/spdnet_aux_losses/spdnet_spatial_<preset>_<YYYYMMDD>/
#          MLflow run name = same.
###############################################################################

set -euo pipefail

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ---------------------------------------------------------------------------
# Defaults (override via env vars at the top of the call).
# ---------------------------------------------------------------------------

MAX_EPOCHS="${MAX_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
NUM_REFS="${NUM_REFS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-200}"
AUGMENTATION="${AUGMENTATION:-heavy}"
INCLUDE_PV="${INCLUDE_PV:-true}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# CLI flags.
DRY_RUN=0
PRESET=all
FROM_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift ;;
        --preset)          PRESET="$2"; shift 2 ;;
        --preset=*)        PRESET="${1#--preset=}"; shift ;;
        --from-checkpoint) FROM_CHECKPOINT="$2"; shift 2 ;;
        --from-checkpoint=*) FROM_CHECKPOINT="${1#--from-checkpoint=}"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

case "$PRESET" in
    eq|eq_con|eq_con_distill|eq_con_warmup|eq_con_warmstart) ;;
    d1_ac_warmstart|d2_mask_warmstart|d3_d2plus_union_warmstart) ;;
    d4_main_warmstart|d4_attn_only_warmstart|d4_ac_safe_warmstart|d4_int_warmstart) ;;
    all) ;;
    *)
        echo "Unknown --preset $PRESET" >&2
        echo "Valid presets:" >&2
        echo "  classic:    eq, eq_con, eq_con_distill, eq_con_warmup, eq_con_warmstart, all" >&2
        echo "  diagnostic: d1_ac_warmstart, d2_mask_warmstart, d3_d2plus_union_warmstart" >&2
        echo "  d4 ablation: d4_main_warmstart, d4_attn_only_warmstart, d4_ac_safe_warmstart, d4_int_warmstart" >&2
        exit 2
        ;;
esac

mkdir -p logs

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

# Format (6 space-separated fields):
#   LAMBDA_EQ LAMBDA_CON LAMBDA_DISTILL DISTILL_WARMUP CON_WARMUP_START CON_WARMUP_EPOCHS
#
# The last two are the linear-warmup schedule for L_con introduced in
# LossesConfig.con_warmup_{start_epoch,epochs}. Defaults of "0 0"
# reproduce the original no-warmup behaviour (lambda_con applied in full
# from epoch 0 onward).
declare -A PRESET_LAMBDAS=(
    ["eq"]="1.0 0.0 0.0 0 0 0"
    ["eq_con"]="1.0 0.5 0.0 0 0 0"
    ["eq_con_distill"]="1.0 0.5 0.1 10 0 0"
    # F: eq_con from scratch, L_con warmup starting after classifier has
    # reached val/mAP ~0.6 (around epoch 14 on the eq-only run). Ramp 7
    # means L_con reaches its full 0.5 weight at epoch 21.
    ["eq_con_warmup"]="1.0 0.5 0.0 0 14 7"
    # C: eq_con resumed from a converged eq-only ckpt. Classifier is
    # already at val/mAP ~0.86, so start the ramp immediately; keep it
    # short (5 epochs) to avoid spending training budget on a long warmup
    # that is no longer justified by the classifier being weak.
    ["eq_con_warmstart"]="1.0 0.5 0.0 0 0 5"

    # --- D1/D2/D3 diagnostic interventions (warmstart required) ---
    # D1: replace L_eq entirely with attention-concentration regulariser.
    #     lambda_eq=0 means the SCA's own attn_map is requested only for
    #     L_ac (cheaper forward path). L_con and L_mask stay off.
    ["d1_ac_warmstart"]="0.0 0.0 0.0 0 0 0"
    # D2: direct pseudo-mask supervision; no L_eq, L_ac, L_con, L_dist.
    ["d2_mask_warmstart"]="0.0 0.0 0.0 0 0 0"
    # D3: D2 + L_con with union anchors. L_con at 0.5 with short ramp
    #     (start=0, ramp=5) to avoid destabilising the warmstarted
    #     classifier in the first few steps. L_mask is added via
    #     PRESET_EXTRA_HYDRA below.
    ["d3_d2plus_union_warmstart"]="0.0 0.5 0.0 0 0 5"

    # --- D4 ablation: magnitude-rebalanced attention + union pseudo-mask ---
    # All four D4 presets: lambda_eq=0, lambda_con=0, lambda_distill=0,
    # no warmup. The attention / mask hyperparameters are set via
    # PRESET_EXTRA_HYDRA below.
    ["d4_main_warmstart"]="0.0 0.0 0.0 0 0 0"
    ["d4_attn_only_warmstart"]="0.0 0.0 0.0 0 0 0"
    ["d4_ac_safe_warmstart"]="0.0 0.0 0.0 0 0 0"
    ["d4_int_warmstart"]="0.0 0.0 0.0 0 0 0"
)

# Additional Hydra overrides specific to each preset (on top of the lambdas
# above). Absent key -> no extras. Keeps D1/D2/D3 readable without bloating
# the 6-field PRESET_LAMBDAS format.
declare -A PRESET_EXTRA_HYDRA=(
    ["d1_ac_warmstart"]=" \
        losses.lambda_ac=0.5"
    ["d2_mask_warmstart"]=" \
        losses.lambda_mask=1.0 \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.5 \
        losses.mask_use_intersection=true"
    ["d3_d2plus_union_warmstart"]=" \
        losses.lambda_mask=1.0 \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.5 \
        losses.mask_use_intersection=true \
        losses.con_anchor_source=union_cls_chvar"

    # D4 main: L_marg_H attention + L_mask(union). No L_ac, no L_eq/L_con/L_dist.
    #   lambda_marg_H=0.15 per RQ1/RQ2/D4-preflight; mask_combiner=union
    #   per RQ5 (chvar ∪ cam_D2 gave +3 pp IoU vs intersection).
    #   mask_use_intersection=null explicitly disables the deprecated
    #   alias so mask_combiner wins.
    ["d4_main_warmstart"]=" \
        losses.lambda_marg_H=0.15 \
        losses.marg_H_beta=0.25 \
        losses.lambda_mask=0.10 \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.50 \
        losses.mask_combiner=union \
        losses.mask_use_intersection=null \
        losses.mask_warmup_epochs=0"

    # D4 attention-only: L_marg_H without L_mask. Isolates attention
    # contribution to val/cam_iou.
    ["d4_attn_only_warmstart"]=" \
        losses.lambda_marg_H=0.15 \
        losses.marg_H_beta=0.25"

    # D4 ac-safe: pure-magnitude test (H1). Uses classical L_ac but at a
    # RQ1-calibrated weight of 0.05 (vs the blown-up 0.5 in D1) and pairs
    # it with L_mask(union). Should answer "does rescaling L_ac alone fix
    # the D1 collapse, even without the marginal term?".
    ["d4_ac_safe_warmstart"]=" \
        losses.lambda_ac=0.05 \
        losses.lambda_mask=0.10 \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.50 \
        losses.mask_combiner=union \
        losses.mask_use_intersection=null \
        losses.mask_warmup_epochs=0"

    # D4 int: A/B twin of d4_main with the old intersection combiner.
    # Isolates the RQ5 union-vs-intersection hypothesis (H3).
    ["d4_int_warmstart"]=" \
        losses.lambda_marg_H=0.15 \
        losses.marg_H_beta=0.25 \
        losses.lambda_mask=0.10 \
        losses.mask_alpha_pos=0.25 \
        losses.mask_beta_neg=0.50 \
        losses.mask_combiner=intersection \
        losses.mask_use_intersection=null \
        losses.mask_warmup_epochs=0"
)

# ---------------------------------------------------------------------------
# Launch one preset.
# ---------------------------------------------------------------------------

run_preset() {
    local preset="$1"
    local cfg="${PRESET_LAMBDAS[$preset]}"
    read -r LAMBDA_EQ LAMBDA_CON LAMBDA_DISTILL DISTILL_WARMUP \
        CON_WARMUP_START CON_WARMUP_EPOCHS <<< "$cfg"

    # Presets whose name contains "warmstart" REQUIRE --from-checkpoint.
    # Fail early with a clear message instead of silently training from a
    # fresh init.
    if [[ "$preset" == *"warmstart"* && -z "$FROM_CHECKPOINT" ]]; then
        echo "ERROR: preset '${preset}' requires --from-checkpoint <path>" >&2
        return 2
    fi

    local run_name="spdnet_spatial_${preset}_${DATE_TAG}"
    local log_path="logs/spdnet_aux_losses_${preset}_$(date +%Y%m%d_%H%M%S).log"

    # Hydra overrides shared by every preset.
    local -a HYDRA_ARGS=(
        "run_name=${run_name}"
        "experiment_name=spdnet_aux_losses"
        "model.fusion_mode=spatial"
        "trainer.max_epochs=${MAX_EPOCHS}"
        "trainer.log_every_n_steps=${LOG_EVERY}"
        "trainer.accumulate_grad_batches=${ACCUM_GRAD}"
        "data.batch_size=${BATCH_SIZE}"
        "data.num_references=${NUM_REFS}"
        "data.augmentation=${AUGMENTATION}"
        "data.include_plantvillage=${INCLUDE_PV}"
        "data.num_workers=${NUM_WORKERS}"
        "losses.lambda_eq=${LAMBDA_EQ}"
        "losses.lambda_con=${LAMBDA_CON}"
        "losses.lambda_distill=${LAMBDA_DISTILL}"
        "losses.distill_warmup_epochs=${DISTILL_WARMUP}"
        "losses.con_warmup_start_epoch=${CON_WARMUP_START}"
        "losses.con_warmup_epochs=${CON_WARMUP_EPOCHS}"
        "losses.online_loc_eval_enabled=true"
        "losses.online_loc_eval_every_n_epochs=1"
    )

    # Preset-specific extras (D1/D2/D3 turn on lambda_ac, lambda_mask,
    # mask_*, con_anchor_source here). Splits on whitespace so multi-line
    # associative-array values work.
    local extras="${PRESET_EXTRA_HYDRA[$preset]:-}"
    if [[ -n "${extras// /}" ]]; then  # non-empty after stripping spaces
        # shellcheck disable=SC2206  # word-splitting is intentional
        local extra_arr=( $extras )
        HYDRA_ARGS+=("${extra_arr[@]}")
    fi

    if [[ -n "$FROM_CHECKPOINT" ]]; then
        # Hydra doesn't have a stock `checkpoint=` field on SPDNetConfig --
        # forward as `+checkpoint=` so it appears in the resolved cfg
        # without forcing every preset to declare it. train_spdnet.py
        # then consumes it via OmegaConf.select(cfg, "checkpoint").
        # Single-quote the value: our ModelCheckpoint filename template
        # produces paths with '=' and '/' chars (e.g.
        # "epoch=72-val_mAP=val/mAP=0.8615.ckpt") that Hydra's override
        # parser refuses to parse unquoted.
        HYDRA_ARGS+=("+checkpoint='${FROM_CHECKPOINT}'")
    fi

    echo ""
    echo "================================================================"
    echo "  Preset: ${preset}"
    echo "    run_name:        ${run_name}"
    echo "    lambda_eq:       ${LAMBDA_EQ}"
    echo "    lambda_con:      ${LAMBDA_CON}  (warmup start=${CON_WARMUP_START} ramp=${CON_WARMUP_EPOCHS})"
    echo "    lambda_distill:  ${LAMBDA_DISTILL}  (warmup ${DISTILL_WARMUP} ep)"
    if [[ -n "${extras// /}" ]]; then
        echo "    extras:          ${extras}"
    fi
    echo "    epochs:          ${MAX_EPOCHS}"
    echo "    PS+PV:           ${INCLUDE_PV}"
    echo "    warmstart:       ${FROM_CHECKPOINT:-(none)}"
    echo "    log:             ${log_path}"
    echo "================================================================"

    if (( DRY_RUN )); then
        echo "[DRY RUN] python src/train_spdnet.py \\"
        for arg in "${HYDRA_ARGS[@]}"; do
            printf "    %s \\\n" "$arg"
        done
        echo "    --cfg job"
        # Resolve and print the Hydra config without launching.
        python src/train_spdnet.py "${HYDRA_ARGS[@]}" --cfg job
        return 0
    fi

    # Real launch: tee output to the per-preset log.
    set +e
    python src/train_spdnet.py "${HYDRA_ARGS[@]}" 2>&1 | tee "${log_path}"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "  Preset ${preset} FAILED (exit ${rc}); see ${log_path}"
        return $rc
    fi
    echo "  Preset ${preset} finished successfully."
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo "==============================================================="
echo "  SPDNet Auxiliary Spatial Losses overnight launcher"
echo "==============================================================="
echo "  CWD:           $(pwd)"
echo "  date_tag:      ${DATE_TAG}"
echo "  preset:        ${PRESET}"
echo "  max_epochs:    ${MAX_EPOCHS}"
echo "  batch:         ${BATCH_SIZE}  accum=${ACCUM_GRAD}  refs=${NUM_REFS}  workers=${NUM_WORKERS}"
echo "  augmentation:  ${AUGMENTATION}"
echo "  include_pv:    ${INCLUDE_PV}"
echo "  log_every:     ${LOG_EVERY} steps"
echo "  dry_run:       ${DRY_RUN}"
echo "  from_ckpt:     ${FROM_CHECKPOINT:-(none)}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if [[ "$PRESET" == "all" ]]; then
    for preset in eq eq_con eq_con_distill; do
        run_preset "$preset"
    done
else
    run_preset "$PRESET"
fi

echo ""
echo "==============================================================="
echo "  All requested presets done."
echo "==============================================================="
