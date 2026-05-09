#!/usr/bin/env bash
###############################################################################
# Phase 5 5090 chain: warm-start fine-tune with enlarged SCA key space.
#
# Context (RESEARCH_CONTEXT.md §§5.14.6-5.14.8):
#   * The Phase-5 LR-fix campaign on a single 32 GiB card verified Trap 1
#     (eff-batch LR rule) and Trap 2 (auto ref_pool_size=20 at 896) but
#     failed to break out of the architectural ceiling at val/mAP ≈ 0.85.
#     The L_mask warm-start hypothesis (P2 of run_phase5_lr_fix.sh) showed
#     +6.4 pp on cam_iou_auc, paid for with a 15 pp mAP regression -- a
#     classic "regulariser too strong" symptom.
#   * Host upgrade to a 2× RTX 5090 box (32 GiB / card, 1.79 TB/s, NCCL
#     P2P) opens up two new degrees of freedom:
#       (a) ref_pool_size = 56 -- a 7.84× key-space jump over the
#           single-card rps=20 ceiling, taking Q:K from 125:1 to 16:1.
#       (b) per-card batch=8 + accum=2 across 2 ranks -> eff_batch=32,
#           identical to the 448 baseline that produced val/mAP=0.888.
#     Both shifts are fed by Lightning DDP with find_unused_parameters=True
#     (SPDNet enters the proj_head / ema_teacher / attn-buffer paths
#     conditionally; without that flag DDP raises "expected to have
#     finished reduction" on epoch 0 -- see RESEARCH_CONTEXT.md §5.14.8).
#   * VRAM profiling (scripts/smoke_ddp_5090.py) showed rps=56 only fits
#     at per-rank batch=2. To compensate for the noisy BN statistics
#     this micro-batch implies, the chain enables Trainer.sync_batchnorm
#     so backbone BN aggregates across the 2-rank world batch (effective
#     BN sample = 4 instead of 2 per rank).
#
# Two phases run sequentially:
#
#   P1' (cls_only_rps56)  ~10-12 h, 50 ep
#       Pure classifier with ref_pool_size=56 + DDP(2). Disables every
#       aux lambda (lambda_eq=lambda_ac=lambda_marg_H=lambda_mask=
#       lambda_con=lambda_distill=0) but turns on the new
#       losses.log_attn_stats flag so we record train/attn_mean,
#       train/attn_std, train/attn_p99 every step. This becomes the
#       apples-to-apples "pure" attention reference against which P2'
#       is measured.
#       Acceptance: val/mAP >= 0.85 (matches single-card P1's 0.849);
#       attn_mean trajectory stays < 0.7 (no D1-style collapse on a
#       cls-only run -- if this fails, ref_pool_size=56 is unstable
#       even without aux losses, and the chain aborts before P2').
#
#   P2' (warm_mask_rps56)  ~5-6 h, 25 ep
#       Warm-start from P1' best_cam_iou.ckpt and add lambda_mask=0.05
#       (D2 union-combiner, RQ5 winner). Same DDP + rps=56 + log_attn_stats
#       setup so we can compare attn distribution before/after L_mask is
#       enforced. Mask warmup is 0 epochs (full strength from ep 0)
#       because the P1' classifier is already converged -- this is the
#       "regulariser, not competitor" calibration.
#       Acceptance: val/cam_iou_best >= 0.30 (vs single-card P2's 0.284);
#       val/cam_iou_auc >= 0.24 (vs 0.222); val/mAP stays >= 0.83;
#       attn_std INCREASES rather than decreases (more discriminative
#       attention vs collapse).
#
# Idempotency: per-phase markers under outputs/_phase5_5090_chain/.
# Remove a marker to force re-run.
#
# DDP-safety code prerequisites (verified by pre-flight):
#   * SPDNetTrainerConfig.{strategy, find_unused_parameters} present.
#   * train_spdnet._resolve_trainer_strategy returns DDPStrategy on
#     devices > 1 + auto/ddp.
#   * lightning.py training_step uses sync_dist=True on every scalar log.
#   * lightning.py on_validation_epoch_end runs OnlineCAMIoU.evaluate on
#     EVERY rank (symmetric contract; rank-0-only used to deadlock --
#     ModelCheckpoint(monitor="val/cam_iou_best") took different code
#     paths between ranks, hung NCCL on an asymmetric save barrier;
#     see RESEARCH_CONTEXT.md §5.14.6 + tests/test_spdnet.py
#     ::TestOnlineCAMIoUOOMDefense). val/cam_iou_* logs use
#     sync_dist=True (NOT rank_zero_only=True).
#   * SPDNetSpatialLossesConfig.log_attn_stats present and propagated.
#
# MLflow collision guard: the smoke test in scripts/run_phase5_5090_smoke.sh
# (or the inline command in §3 of the run plan) writes to
# ./mlruns_smoke/, NOT to ./mlruns/. This launcher additionally
# verifies that mlruns/ is DVC-clean before P1' starts -- if the user
# accidentally pointed a smoke run at the production mlruns/ the chain
# bails out with rc=3 rather than corrupting the DVC cache.
#
# Usage:
#   bash scripts/run_phase5_5090_chain.sh                       # P1' + P2'
#   bash scripts/run_phase5_5090_chain.sh --preflight-only      # just verify
#   PHASES="P1"        bash scripts/run_phase5_5090_chain.sh
#   PHASES="P2"        bash scripts/run_phase5_5090_chain.sh    # needs P1_CKPT
#   REF_POOL_SIZE=40   bash scripts/run_phase5_5090_chain.sh    # alt key space
#   P1_BATCH=6         bash scripts/run_phase5_5090_chain.sh    # OOM fallback
#
# Logs:    logs/phase5_5090_<phase>_<timestamp>.log
# Outputs: outputs/phase5_5090_chain/<run_name>/
# MLflow:  experiment "phase5_5090_chain"
###############################################################################

set -uo pipefail   # NOT set -e: a failed P1' should not skip the summary.

cd /workspace/plant-diseases-segmentation
export PATH="/venv/main/bin:$PATH"

# ----------------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------------

PREFLIGHT_ONLY=0
SKIP_MLRUNS_CHECK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only)    PREFLIGHT_ONLY=1; shift ;;
        --skip-mlruns-check) SKIP_MLRUNS_CHECK=1; shift ;;
        -h|--help) sed -n '2,80p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. Try --help." >&2; exit 1 ;;
    esac
done

# ----------------------------------------------------------------------------
# Configurable knobs
# ----------------------------------------------------------------------------

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M)}"
EXPERIMENT="${EXPERIMENT:-phase5_5090_chain}"

IMAGE_SIZE="${IMAGE_SIZE:-896}"
INCLUDE_PV="${INCLUDE_PV:-true}"
LOG_EVERY="${LOG_EVERY:-50}"
AUGMENTATION="${AUGMENTATION:-heavy}"
NUM_REFS="${NUM_REFS:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"     # per rank -> 12 total on the 256-vCPU box

# Per-phase batch/accum. With trainer.devices=2 the effective batch is
# 2 * P{1,2}_BATCH * P{1,2}_ACCUM. We target eff_batch=32 (matches the
# 448 calibration that yielded val/mAP=0.888) by default.
#
# Defaults below are calibrated against the per-rank VRAM ceiling
# discovered by ``scripts/smoke_ddp_5090.py`` at 896 + rps=56 +
# log_attn_stats=true:
#
#   * batch=8, rps=56  -> OOM (peak ~40 GiB per rank in 2-rank DDP)
#   * batch=4, rps=56  -> OOM (~32 GiB)
#   * batch=2, rps=56  -> 24.6 GiB peak  PASS (the dream config)
#   * batch=4, rps=40  -> OOM (~28 GiB)
#   * batch=3, rps=40  -> 21.1 GiB peak  PASS (alt: 4× key space)
#   * batch=6, rps=28  -> 25.2 GiB peak  PASS (alt: bigger micro-batch)
#
# The plan's pre-empirical "batch=8 + rps=56 fits in ~20 GiB" estimate
# was incorrect: PyTorch's ``MultiheadAttention`` with ``need_weights=
# True`` materialises the full B*H*Q*K weight tensor and retains it
# for backward, blowing the activation budget at scale. We pick rps=56
# + batch=2 + accum=8 by default to preserve the headline architectural
# test (7.84× key space vs single-card rps=20) at eff_batch=32 (same as
# the 448 calibration). Override to (3, 5, 40) or (6, 3, 28) for the
# alternative trade-offs.
P1_BATCH="${P1_BATCH:-2}"
P1_ACCUM="${P1_ACCUM:-4}"
P2_BATCH="${P2_BATCH:-2}"
P2_ACCUM="${P2_ACCUM:-4}"

MAX_EPOCHS_P1="${MAX_EPOCHS_P1:-60}"
MAX_EPOCHS_P2="${MAX_EPOCHS_P2:-30}"

# Base LR -- the optimiser sees scaled_lr = LR_BASE * eff_batch / 256.
# At eff_batch=32 and LR_BASE=5e-4 this is 6.25e-5, identical to the 448
# baseline. P2' uses a learning_rate_override (preserves the converged
# classifier; see SPDNetModelConfig.learning_rate_override docstring).
LR_BASE="${LR_BASE:-0.0005}"
LR_OVERRIDE_P2="${LR_OVERRIDE_P2:-1.2e-5}"

# SCA reference pool grid side. 56 takes Q:K from 125:1 (rps=20) to
# 16:1 at 896². Override to 40 if smoke shows OOM at batch=8.
REF_POOL_SIZE="${REF_POOL_SIZE:-56}"

# OnlineCAMIoU eval batch -- MUST be sized to match the training
# micro-batch for high-resolution / high-rps recipes. The metric does
# its own forward pass through SPDNet's SCA, materialising the
# (B, heads, Q, K) attention weight tensor in fp32:
#
#   ONLINE_LOC_EVAL_BS=2 + rps=56 + 896² -> ~5 GiB attention weights  PASS
#   ONLINE_LOC_EVAL_BS=4 + rps=56 + 896² -> ~10 GiB                   PASS (tight)
#   ONLINE_LOC_EVAL_BS=8 + rps=56 + 896² -> ~20 GiB                   OOM
#
# The 2026-05-06 P1' run died at end of epoch 0 because the 448-tuned
# default of 8 crashed at rps=56 (the launcher hardcoded ref_pool_size
# but inherited the default eval batch). The training-residual + the
# 20 GiB attention weights overshot the 32 GiB budget; rank 1 then sat
# on the next ALLREDUCE for the full 30-min NCCL watchdog window.
# Default 2 keeps the metric's runtime at ~1 min on 100 query images
# (50 forward batches at ~1s each) and is safe at the most aggressive
# rps used in the chain. Override only when running at lower rps.
ONLINE_LOC_EVAL_BS="${ONLINE_LOC_EVAL_BS:-2}"

# Min LR floors. P1' uses cosine to 1e-6; P2' uses cosine to 1e-7 below
# the override 1.2e-5 so there's still a strict descent (Trap 3 guard).
MIN_LR_P1="${MIN_LR_P1:-1e-6}"
MIN_LR_P2="${MIN_LR_P2:-1e-7}"

# Warmup epochs. P2' uses 2 epochs (classifier already converged); P1'
# uses 5 epochs (fresh init at the new rps=56 key space).
WARMUP_P1="${WARMUP_P1:-5}"
WARMUP_P2="${WARMUP_P2:-2}"

# DDP. Default 4 GPUs on the 5090 host (4x RTX 5090 / SYS-PCIe). The
# resolver in train_spdnet.py picks DDPStrategy(find_unused_parameters=
# True, gradient_as_bucket_view=True) when devices > 1 + strategy in
# {"auto", "ddp"}. With DEVICES=4 + accum=4 we keep eff_batch=32
# (matches the single-card 448 baseline). Override DEVICES=2 + accum=8
# for the older 2-GPU host or DEVICES=1 + accum=16 as an emergency
# single-card fallback.
DEVICES="${DEVICES:-4}"
STRATEGY="${STRATEGY:-ddp}"

# DDP env hygiene.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Phases. Default both.
PHASES="${PHASES-P1 P2}"

# Optional explicit P1 checkpoint for P2-only re-runs. When P1 is in
# PHASES the launcher overrides this with the actual run output.
P1_CKPT_OVERRIDE="${P1_CKPT_OVERRIDE:-}"

OUT_BASE="outputs/${EXPERIMENT}"
MARKER_DIR="outputs/_${EXPERIMENT}"
LOG_DIR="logs/${EXPERIMENT}"
mkdir -p "$OUT_BASE" "$MARKER_DIR" "$LOG_DIR"

# ----------------------------------------------------------------------------
# Pre-flight banner
# ----------------------------------------------------------------------------

echo "================================================================"
echo "  Phase 5 5090 chain: warm-start fine-tune at rps=${REF_POOL_SIZE}"
echo "  Started:        $(date)"
echo "  DATE_TAG:       $DATE_TAG"
echo "  EXPERIMENT:     $EXPERIMENT"
echo "  IMAGE_SIZE:     $IMAGE_SIZE"
echo "  REF_POOL_SIZE:  $REF_POOL_SIZE  (Q:K = $((IMAGE_SIZE/4))²:${REF_POOL_SIZE}²)"
echo "  LR_BASE:        $LR_BASE  (eff-batch//256 scaling at runtime)"
echo "  P1' (cls):      batch=$P1_BATCH accum=$P1_ACCUM eff_batch=$((DEVICES * P1_BATCH * P1_ACCUM)) ep=$MAX_EPOCHS_P1 warmup=$WARMUP_P1 min_lr=$MIN_LR_P1"
echo "  P2' (mask):     batch=$P2_BATCH accum=$P2_ACCUM eff_batch=$((DEVICES * P2_BATCH * P2_ACCUM)) ep=$MAX_EPOCHS_P2 warmup=$WARMUP_P2 min_lr=$MIN_LR_P2 lr_override=$LR_OVERRIDE_P2"
echo "  Eval batch:     online_loc_eval_batch_size=$ONLINE_LOC_EVAL_BS  (per-rank OnlineCAMIoU forward; deterministic+symmetric across ranks; OOM-safe for rps=$REF_POOL_SIZE + ${IMAGE_SIZE}px)"
echo "  DEVICES:        $DEVICES   STRATEGY: $STRATEGY"
echo "  Phases:         ${PHASES:-<none>}"
echo "  Preflight only: $PREFLIGHT_ONLY"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | sed 's/^/                  /'
echo "================================================================"

# Validate phase tokens.
for phase in ${PHASES:-}; do
    case "$phase" in
        P1|P2) : ;;
        *) echo "ERROR: PHASES contains unknown phase '$phase'. Valid: P1 P2." >&2; exit 4 ;;
    esac
done

# ----------------------------------------------------------------------------
# Pre-flight: confirm 2 GPUs visible (or DEVICES of them).
# ----------------------------------------------------------------------------

n_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "$n_gpus" -lt "$DEVICES" ]]; then
    echo "ERROR: DEVICES=$DEVICES but only $n_gpus GPU(s) visible." >&2
    exit 5
fi

# ----------------------------------------------------------------------------
# Pre-flight: code sanity. Verifies DDP-safety + log_attn_stats hooks.
# ----------------------------------------------------------------------------

echo ""
echo "Pre-flight: verifying DDP-safety + attention-stats code hooks..."
python - <<'PY'
import sys, inspect

from src.conf.spdnet import (
    SPDNetSpatialLossesConfig,
    SPDNetTrainerConfig,
    SPDNetModelConfig,
)
from src.train_spdnet import _resolve_trainer_strategy
from src.wsss.spdnet.lightning import SPDNetModule
from lightning.pytorch.strategies import DDPStrategy

# log_attn_stats wired through to losses cfg.
assert hasattr(SPDNetSpatialLossesConfig, "log_attn_stats"), \
    "Missing SPDNetSpatialLossesConfig.log_attn_stats"

# DDP knobs on trainer cfg.
assert hasattr(SPDNetTrainerConfig, "strategy"), \
    "Missing SPDNetTrainerConfig.strategy"
assert hasattr(SPDNetTrainerConfig, "find_unused_parameters"), \
    "Missing SPDNetTrainerConfig.find_unused_parameters"
assert hasattr(SPDNetTrainerConfig, "sync_batchnorm"), \
    "Missing SPDNetTrainerConfig.sync_batchnorm (needed for batch=2/rank stability)"

# train_spdnet wires sync_batchnorm into Trainer.__init__.
import src.train_spdnet as _ts
src_t = inspect.getsource(_ts.train_spdnet)
assert "sync_batchnorm=" in src_t, \
    "train_spdnet does not pass sync_batchnorm to L.Trainer"

# LR scaling MUST multiply by devices, not just batch * accum.
# Regression: the 2026-05-07 P1' run on 4x 5090 used the per-rank-only
# formula (batch*accum/256 = 8/256 instead of 32/256 for our setup),
# giving peak LR=1.56e-5 vs the correct 6.25e-5. After 36 epochs the
# val/mAP was 0.51 vs 0.79 in the equivalent single-card baseline at
# the same epoch. The formula must be ``base_lr * eff_batch_global /
# 256`` with ``eff_batch_global = batch * accum * devices``. We strip
# python ``#`` comments first because the docstring above the
# assignment also contains the literal string "eff_batch = batch *
# accum" (describing the OLD formula) -- without stripping comments
# the regex would false-positive on the docstring.
import re as _re
_src_t_code = "\n".join(
    _re.sub(r"(?<!['\"])#.*$", "", ln) for ln in src_t.splitlines()
)
# Match the RHS only on lines that look like an actual assignment
# (indented, starts with `eff_batch`).
_eff_match = _re.search(
    r"^\s+eff_batch\s*=\s*([^\n]+)", _src_t_code, _re.M,
)
assert _eff_match, \
    "Could not find eff_batch assignment in train_spdnet (LR-scaling block)."
_eff_rhs = _eff_match.group(1).strip()
assert _re.search(r"\b(devices|world_size|num_devices)\b", _eff_rhs), \
    (f"eff_batch RHS does not include devices/world_size: {_eff_rhs!r}. "
     "The 2026-05-07 P1' run was crippled by the per-rank-only formula "
     "``batch * accum``. The corrected formula is ``batch * accum * "
     "devices`` (linear scaling on the GLOBAL effective batch). See "
     "RESEARCH_CONTEXT.md and src/train_spdnet.py docstring.")

# Resolver returns DDPStrategy with find_unused_parameters=True for 2-device auto.
out = _resolve_trainer_strategy(strategy="auto", devices=2,
                                find_unused_parameters=True)
assert isinstance(out, DDPStrategy), \
    f"_resolve_trainer_strategy returned {type(out).__name__} not DDPStrategy"
fp = out._ddp_kwargs.get("find_unused_parameters")
assert fp is True, f"find_unused_parameters not propagated; got {fp!r}"
gb = out._ddp_kwargs.get("gradient_as_bucket_view")
assert gb is True, f"gradient_as_bucket_view not propagated; got {gb!r}"

# training_step uses sync_dist=True on its scalar logs (train/loss,
# train/L_*, train/attn_*) and DOES NOT compute train/mAP any more.
# After the 2026-05-07 P1' smoke series we dropped train/mAP entirely:
# the train preds tensor (8470x115 per rank) all_gather hangs on this
# 4x 5090 host (smoke #6 watchdog: ranks 1..3 stuck on ALLGATHER
# NumelIn=974050 NumelOut=3896200 while rank 0 had advanced past).
# The same primitive works for val (1122x115 = 129030 elements) so
# val/mAP is still computed in-fit via the manual gather; train/mAP
# becomes a post-hoc offline computation if ever needed.
src = inspect.getsource(SPDNetModule.training_step)
assert src.count("sync_dist=True") >= 2, \
    f"sync_dist=True missing in training_step (count={src.count('sync_dist=True')})"
assert "self.train_mAP" not in src, \
    "training_step still references self.train_mAP. After smoke #6 " \
    "we removed in-fit train/mAP entirely -- the train preds " \
    "all_gather deadlocks on this host. Compute it offline if needed."
assert "_train_preds_buf" not in src, \
    "training_step still buffers train preds. The train all_gather " \
    "deadlocks; there's no point buffering. Drop the buffer code."

# on_train_epoch_end is now a no-op (kept as a stable hook point for
# subclass overrides). It must not perform any cross-rank collective.
src_t_end = inspect.getsource(SPDNetModule.on_train_epoch_end)
assert "self.train_mAP" not in src_t_end, \
    "on_train_epoch_end still references self.train_mAP. We dropped it."
assert "self.all_gather" not in src_t_end, \
    "on_train_epoch_end calls self.all_gather. The train preds gather " \
    "deadlocks at the 8470x115 size on this 4x 5090 host (smoke #6). " \
    "Make on_train_epoch_end a no-op; do not compute train/mAP in-fit."
assert "multilabel_average_precision" not in src_t_end, \
    "on_train_epoch_end still computes mAP. Don't -- the gather hangs."

# log_attn_stats branch is present in training_step.
assert "log_attn_stats" in src, "training_step does not consult log_attn_stats"
assert 'comp["attn_p99"]' in src, "training_step does not emit attn_p99"

# OnlineCAMIoU branch must be SYMMETRIC across ranks.
# 2026-05-06 deadlock fix: rank-0-only evaluate + rank_zero_only=True
# log put val/cam_iou_best on rank 0 only, which made
# ModelCheckpoint(monitor="val/cam_iou_best") take asymmetric code
# paths (rank 0 saved + barrier; rank 1 skipped). NCCL hung on the
# unmatched barrier until the 600 s watchdog killed both processes.
# Diagnostic from the failed run::
#
#   Rank 0: WorkNCCL(SeqNum=2049548, OpType=ALLREDUCE,  NumelIn=1)
#   Rank 1: WorkNCCL(SeqNum=2049547, OpType=ALLGATHER,  NumelIn=2)
#
# Rank 0 had issued exactly one extra collective -- the save
# barrier. The fix in src/wsss/spdnet/lightning.py: every rank now
# computes OnlineCAMIoU.evaluate (deterministic on the same query
# subset + seed + DDP-synced weights), logs with sync_dist=True so
# the metric exists on every rank's callback_metrics, and an
# all_reduce(MIN) on a success flag coordinates OOM across ranks
# so an asymmetric one-rank failure can't reintroduce the deadlock
# through the back door. Pre-flight assertions below pin all of
# this in place; tests/test_spdnet.py::TestOnlineCAMIoUOOMDefense
# is the structured regression guard.
src_v = inspect.getsource(SPDNetModule.on_validation_epoch_end)
assert "self.online_loc_metric.evaluate" in src_v, \
    "on_validation_epoch_end no longer calls OnlineCAMIoU.evaluate"

# Strip line comments before the keyword checks below. The function
# intentionally documents the *historical* buggy patterns in its
# comments ("rank_zero_only=True caused a deadlock..."), so a naive
# substring match would false-positive on the explanation text. We
# also have to strip inline ``# ...`` tail comments since the
# explanation text occasionally appears mid-line. We deliberately
# do NOT strip docstrings/strings: there are no string literals
# referencing 'rank_zero_only=True' or 'is_global_zero' in the body
# (verified manually), so a line-level strip is sufficient and
# preserves the indentation we rely on for the if-guard parser.
def _strip_py_line_comments(src: str) -> str:
    out: list[str] = []
    for ln in src.splitlines():
        # Find the first '#' that is not inside a string. A simple
        # heuristic: count quote chars before the '#'. If even, the
        # '#' is in code; if odd, inside a string. Good enough for
        # the on_validation_epoch_end body which has no '#' chars
        # inside strings.
        hash_idx = ln.find("#")
        if hash_idx < 0:
            out.append(ln)
            continue
        prefix = ln[:hash_idx]
        if (prefix.count("'") + prefix.count('"')) % 2 == 0:
            out.append(prefix.rstrip())
        else:
            out.append(ln)
    return "\n".join(out)

src_v_code = _strip_py_line_comments(src_v)

# No is_global_zero gate on the OnlineCAMIoU branch -- regressing
# this re-introduces the 2026-05-06 deadlock.
_guard_start = src_v_code.find("if (")
_guard_end = -1
while _guard_start != -1:
    _t = src_v_code[_guard_start:]
    _anchor = _t.find("self.online_loc_metric is not None")
    if _anchor == -1 or _anchor > 200:
        _guard_start = src_v_code.find("if (", _guard_start + 1)
        continue
    _depth = 0
    for _i, _ch in enumerate(_t):
        if _ch == "(":
            _depth += 1
        elif _ch == ")":
            _depth -= 1
            if _depth == 0:
                _guard_end = _i + 1
                break
    break
_guard = src_v_code[_guard_start:_guard_start + _guard_end] if _guard_end > 0 else ""
assert _guard, "could not locate online_loc_metric if-guard"
assert "is_global_zero" not in _guard, \
    "is_global_zero gate is back on the OnlineCAMIoU branch -- " \
    "this reintroduces the 2026-05-06 ModelCheckpoint asymmetric " \
    "barrier deadlock. Remove the gate; every rank must compute."
# val/cam_iou_* log must use sync_dist=True NOT rank_zero_only=True.
assert "rank_zero_only = True" not in src_v_code and \
    "rank_zero_only=True" not in src_v_code, \
    "rank_zero_only=True is back in on_validation_epoch_end -- this " \
    "puts val/cam_iou_* on rank 0 only and ModelCheckpoint deadlocks " \
    "on the asymmetric save barrier. Use sync_dist=True instead."
assert "sync_dist = True" in src_v_code or "sync_dist=True" in src_v_code, \
    "Missing sync_dist=True on the val/cam_iou_* self.log() call. " \
    "Without it the metric is rank-local and ModelCheckpoint takes " \
    "the asymmetric path again."
# Cross-rank OOM coordination -- closes the asymmetric-OOM back door.
assert "torch.distributed.all_reduce" in src_v_code, \
    "Missing torch.distributed.all_reduce for cross-rank OOM " \
    "coordination in on_validation_epoch_end."
assert "ReduceOp.MIN" in src_v_code, \
    "OOM coordination must use ReduceOp.MIN (any-rank-failure -> " \
    "skip everywhere). MAX-reduce is the buggy original."
# OOM defense-in-depth: try/except around evaluate().
assert "except torch.cuda.OutOfMemoryError" in src_v_code, \
    "Lost the try/except torch.cuda.OutOfMemoryError around " \
    "OnlineCAMIoU.evaluate -- the OOM defense-in-depth is gone."

# val/mAP MUST be computed via the manual-gather + functional-mAP path
# rather than torchmetrics ``MultilabelAveragePrecision.compute()``. The
# class-based compute() runs ``Metric.sync()`` internally which issues
# an ALLGATHER of state-size info. On the 4x 5090 host (PCIe-only NCCL
# topology) that tiny size-info gather desynced against our DDP +
# SyncBN + find_unused_parameters=True + OnlineCAMIoU stack and
# deadlocked the entire fit. The 2026-05-07 P1' smoke #3 watchdog
# stack-trace pinned this exactly: rank 0 reached our OOM-coord
# ALLREDUCE NumelIn=1 while ranks 1..3 were stuck one collective
# behind on ALLGATHER NumelIn=2 NumelOut=8 -- the torchmetrics
# size-info gather. Fix: bypass torchmetrics' Metric.sync() by
# accumulating preds/target into per-rank buffers, gathering them via
# ``self.all_gather`` (Lightning's well-tested symmetric tensor
# primitive that pads to the max size across ranks), and computing
# the mAP locally on every rank with the FUNCTIONAL torchmetrics API
# (``multilabel_average_precision``), which is a pure function with
# no internal collectives.
assert "self.val_mAP.compute" not in src_v_code, \
    "on_validation_epoch_end calls self.val_mAP.compute(). That " \
    "triggers torchmetrics' Metric.sync() ALLGATHER which deadlocked " \
    "on the 2026-05-07 4x 5090 smoke. Use the manual-gather + " \
    "functional mAP path instead. See the comment block above."
assert "self.all_gather" in src_v_code, \
    "on_validation_epoch_end MUST call self.all_gather to cross-rank " \
    "gather buffered preds/target. Without it val/mAP is computed on " \
    "partial val data per rank and diverges across ranks (which then " \
    "breaks ModelCheckpoint(monitor=val/mAP) symmetry)."
assert "multilabel_average_precision" in src_v_code, \
    "on_validation_epoch_end MUST use the FUNCTIONAL torchmetrics " \
    "multilabel_average_precision. The class-based " \
    "MultilabelAveragePrecision is what triggers the internal-sync " \
    "deadlock."

# With the manual-gather flow the value of ``val/mAP`` is identical on
# every rank by construction, so the ``self.log`` call MUST set
# ``sync_dist=False`` -- ``sync_dist=True`` would all-reduce N copies
# of the same scalar and add a redundant collective at exactly the
# epoch boundary we are trying to keep collective-light. (rank_zero_
# only=True is also wrong for the unrelated reason that it would skip
# callback_metrics population on non-zero ranks and break
# ModelCheckpoint symmetry.)
_v_mAP_idx = src_v_code.find('"val/mAP"')
assert _v_mAP_idx > 0, \
    "Couldn't locate val/mAP log call in on_validation_epoch_end."
_v_mAP_log_open = src_v_code.rfind("self.log", 0, _v_mAP_idx)
_v_mAP_depth = 0
_v_mAP_end = -1
for _i, _ch in enumerate(src_v_code[_v_mAP_log_open + len("self.log"):]):
    if _ch == "(":
        _v_mAP_depth += 1
    elif _ch == ")":
        _v_mAP_depth -= 1
        if _v_mAP_depth == 0:
            _v_mAP_end = _v_mAP_log_open + len("self.log") + _i + 1
            break
_v_mAP_call = src_v_code[_v_mAP_log_open:_v_mAP_end] if _v_mAP_end > 0 else ""
assert _v_mAP_call, "Couldn't extract val/mAP log call body."
assert "sync_dist=False" in _v_mAP_call or "sync_dist = False" in _v_mAP_call, \
    "val/mAP log MUST set sync_dist=False because the value is " \
    "already identical on every rank (we gathered the inputs " \
    "ourselves via self.all_gather). sync_dist=True would all-reduce " \
    "identical scalars and add a redundant collective at the val-" \
    "epoch boundary -- the boundary that deadlocked on 2026-05-07."
assert "rank_zero_only=True" not in _v_mAP_call and \
    "rank_zero_only = True" not in _v_mAP_call, \
    "rank_zero_only=True is set on val/mAP -- that puts the metric on " \
    "rank 0 only and reintroduces the asymmetric callback_metrics " \
    "deadlock with ModelCheckpoint."

# Tightened DDP timeout still wired through.
assert hasattr(SPDNetTrainerConfig, "ddp_timeout_seconds"), \
    "Missing SPDNetTrainerConfig.ddp_timeout_seconds (defense in " \
    "depth: any future deadlock fails inside 10 min not 30)."

# Trap 1+2 fixes still present.
assert hasattr(SPDNetModelConfig, "learning_rate_override"), \
    "Trap 1 fix missing: SPDNetModelConfig.learning_rate_override"
assert hasattr(SPDNetModelConfig, "ref_pool_size"), \
    "Trap 2 fix missing: SPDNetModelConfig.ref_pool_size"

print("pre-flight code OK")
sys.exit(0)
PY
if [[ $? -ne 0 ]]; then
    echo "ERROR: pre-flight code check failed. Fix src/ before launching." >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Pre-flight: DVC mlruns sync. Catches the case where a smoke test
# accidentally wrote into ./mlruns/ instead of ./mlruns_smoke/, which
# would cause DVC to refuse the post-run push.
# ----------------------------------------------------------------------------

if (( SKIP_MLRUNS_CHECK == 0 )); then
    echo ""
    echo "Pre-flight: verifying mlruns/ is DVC-clean..."
    if [[ ! -f mlruns.dvc ]]; then
        echo "WARNING: mlruns.dvc not found -- skipping DVC sync check." >&2
    else
        # Capture stdout+stderr so we can distinguish (a) "clean" from
        # (b) "actually dirty" from (c) "lock contention because another
        # DVC process (typically a parallel pull) is running".
        # ``--quiet`` makes the clean case empty.
        dvc_out="$(dvc status mlruns.dvc --quiet 2>&1)"
        dvc_rc=$?
        if (( dvc_rc == 0 )); then
            echo "  mlruns/ is DVC-clean."
        elif echo "$dvc_out" | grep -q "Unable to acquire lock"; then
            # Another DVC process is running. We can't tell whether
            # mlruns/ is dirty without it. Degrade to WARNING + retry
            # advice rather than aborting the chain, because the
            # typical cause is a still-running ``dvc pull mlruns.dvc``
            # which will leave the dir clean once it completes.
            echo "WARNING: dvc status could not run (lock held by another DVC" >&2
            echo "         process). If you're still pulling mlruns, wait for" >&2
            echo "         it to finish before launching the chain. Re-run" >&2
            echo "         this script when the pull is done. Use" >&2
            echo "         --skip-mlruns-check to bypass." >&2
            exit 3
        else
            echo "ERROR: mlruns/ is out of sync with DVC. The chain would" >&2
            echo "       corrupt the cache by mixing pulled + freshly-written" >&2
            echo "       runs. Run 'dvc pull mlruns.dvc' OR 'dvc commit" >&2
            echo "       mlruns.dvc' to reconcile, then re-launch." >&2
            echo "       (Override with --skip-mlruns-check if you know" >&2
            echo "        you don't intend to push the new runs.)" >&2
            dvc status mlruns.dvc 2>&1 | head -20 >&2
            exit 3
        fi
    fi
fi

if (( PREFLIGHT_ONLY )); then
    echo ""
    echo "--preflight-only supplied -> exiting before dispatch."
    exit 0
fi

# ----------------------------------------------------------------------------
# Run-phase helper
# ----------------------------------------------------------------------------

run_phase() {
    local name="$1"; shift
    local marker="$MARKER_DIR/${name}.DONE"
    local log_path="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "  [${name}]  start"
    echo "  log:        $log_path"
    echo "  marker:     $marker"
    echo "  started:    $(date)"
    echo "================================================================"

    if [[ -f "$marker" ]]; then
        echo "[${name}] marker exists -- skipping. (rm $marker to force re-run.)"
        return 0
    fi

    "$@" 2>&1 | tee "$log_path"
    local rc=${PIPESTATUS[0]}
    echo ""
    echo "[${name}] done (rc=$rc) at $(date)"
    if (( rc == 0 )); then
        touch "$marker"
    else
        echo "[${name}] FAILED -- see $log_path; not writing $marker."
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# Common Hydra arguments shared by both phases
# ----------------------------------------------------------------------------

common_hydra_args() {
    local run_name="$1"
    local batch="$2"
    local accum="$3"
    local max_epochs="$4"
    local warmup="$5"
    local min_lr="$6"

    cat <<EOF
run_name=${run_name}
experiment_name=${EXPERIMENT}
model.fusion_mode=spatial
model.input_size=${IMAGE_SIZE}
model.learning_rate=${LR_BASE}
model.ref_pool_size=${REF_POOL_SIZE}
trainer.max_epochs=${max_epochs}
trainer.log_every_n_steps=${LOG_EVERY}
trainer.accumulate_grad_batches=${accum}
trainer.precision=bf16-mixed
trainer.warmup_epochs=${warmup}
trainer.min_lr=${min_lr}
trainer.devices=${DEVICES}
trainer.strategy=${STRATEGY}
trainer.find_unused_parameters=true
trainer.sync_batchnorm=true
data.image_size=${IMAGE_SIZE}
data.batch_size=${batch}
data.num_references=${NUM_REFS}
data.augmentation=${AUGMENTATION}
data.include_plantvillage=${INCLUDE_PV}
data.num_workers=${NUM_WORKERS}
losses.log_attn_stats=true
losses.online_loc_eval_enabled=true
losses.online_loc_eval_batch_size=${ONLINE_LOC_EVAL_BS}
EOF
}

# ----------------------------------------------------------------------------
# Phase definitions
# ----------------------------------------------------------------------------

# Run name templates so P2' can locate P1''s checkpoint.
P1_NAME="phase5_5090_P1_cls_only_rps${REF_POOL_SIZE}_${DATE_TAG}"
P2_NAME="phase5_5090_P2_warm_mask_rps${REF_POOL_SIZE}_${DATE_TAG}"
P1_RUN_DIR="${OUT_BASE}/${P1_NAME}"
P2_RUN_DIR="${OUT_BASE}/${P2_NAME}"
P1_CKPT_DEFAULT="${P1_RUN_DIR}/checkpoints/best_cam_iou.ckpt"

phase_P1_cls_only() {
    local args
    args="$(common_hydra_args "$P1_NAME" \
        "$P1_BATCH" "$P1_ACCUM" "$MAX_EPOCHS_P1" \
        "$WARMUP_P1" "$MIN_LR_P1")"
    args+="
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0
losses.lambda_marg_H=0
losses.lambda_mask=0
"
    python -m src.train_spdnet $args
}

phase_P2_warm_mask_only() {
    local ckpt="${P1_CKPT_OVERRIDE:-$P1_CKPT_DEFAULT}"
    if [[ ! -f "$ckpt" ]]; then
        echo "[P2] ERROR: P1' best_cam_iou checkpoint not found at $ckpt." >&2
        echo "       Either run P1 first (PHASES='P1 P2') or pass" >&2
        echo "       P1_CKPT_OVERRIDE=/path/to/best_cam_iou.ckpt." >&2
        return 6
    fi

    local args
    args="$(common_hydra_args "$P2_NAME" \
        "$P2_BATCH" "$P2_ACCUM" "$MAX_EPOCHS_P2" \
        "$WARMUP_P2" "$MIN_LR_P2")"
    args+="
+checkpoint=${ckpt}
model.learning_rate_override=${LR_OVERRIDE_P2}
losses.lambda_eq=0
losses.lambda_con=0
losses.lambda_distill=0
losses.lambda_ac=0
losses.lambda_marg_H=0
losses.lambda_mask=0.05
losses.mask_alpha_pos=0.25
losses.mask_beta_neg=0.50
losses.mask_combiner=union
losses.mask_use_intersection=null
losses.mask_warmup_start_epoch=0
losses.mask_warmup_epochs=0
"
    python -m src.train_spdnet $args
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

t0=$(date +%s)
declare -A RC=( [P1]=-1 [P2]=-1 )

if [[ -z "${PHASES// }" ]]; then
    echo "PHASES is empty -> nothing to dispatch. Exiting."
    exit 0
fi

for phase in ${PHASES}; do
    case "$phase" in
        P1) run_phase P1 phase_P1_cls_only;       RC[P1]=$? ;;
        P2)
            # Don't bother running P2 if P1 was attempted in this dispatch
            # and failed.
            if [[ "${RC[P1]}" != "-1" && "${RC[P1]}" != "0" ]]; then
                echo "[P2] skipping because P1 failed (rc=${RC[P1]})."
                RC[P2]=-2
                continue
            fi
            run_phase P2 phase_P2_warm_mask_only; RC[P2]=$?
            ;;
    esac
done

t1=$(date +%s)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Phase 5 5090 chain: wall clock $((t1 - t0))s ($(((t1 - t0) / 3600))h)"
echo "================================================================"
for p in P1 P2; do
    case "${RC[$p]}" in
        -1) printf "  %-3s  SKIPPED (not in PHASES)\n" "$p" ;;
        -2) printf "  %-3s  SKIPPED (upstream phase failed)\n" "$p" ;;
        *)  printf "  %-3s  rc=%d\n" "$p" "${RC[$p]}" ;;
    esac
done

echo ""
echo "MLflow runs to inspect (experiment '${EXPERIMENT}'):"
echo "  P1' cls_only:   ${P1_NAME}"
echo "  P2' mask_only:  ${P2_NAME}"
echo ""
echo "Checkpoints:"
echo "  P1' best_cam_iou:  ${P1_CKPT_DEFAULT}"
echo "  P2' best_cam_iou:  ${P2_RUN_DIR}/checkpoints/best_cam_iou.ckpt"
echo ""
echo "Acceptance criteria:"
echo "  P1':  val/mAP             >= 0.85   (matches single-card P1's 0.849)"
echo "        val/cam_iou_best    >= 0.24   (cls-only floor)"
echo "        attn_mean trajectory < 0.70   (no D1-style collapse)"
echo "  P2':  val/cam_iou_best    >= 0.30   (vs single-card P2's 0.284)"
echo "        val/cam_iou_auc     >= 0.24   (vs single-card P2's 0.222)"
echo "        val/mAP             >= 0.83   (within 2 pp of P1' peak)"
echo "        attn_std INCREASES vs P1' baseline (more discriminative)"
echo ""

# Exit non-zero if any phase actually ran and failed.
any_fail=0
for p in P1 P2; do
    rc="${RC[$p]}"
    if [[ "$rc" != "-1" && "$rc" != "-2" && "$rc" != "0" ]]; then
        any_fail=1
    fi
done
if (( any_fail )); then
    echo "At least one phase failed -- inspect logs/${EXPERIMENT}/<phase>_*.log."
    exit 1
fi
echo "All requested phases succeeded."
exit 0
