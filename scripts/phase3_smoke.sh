#!/usr/bin/env bash
#
# Fast end-to-end smoke for the multi-task pretraining pipeline.
# Runs all 3 phases on rel-f1 only at small scale; total wall-clock
# is typically <5 min on a single GPU. Use to catch phase-3 crashes
# (shard format, type remap, encoder OOM, NCCL init, etc.) without
# the ~80-min round trip on the AWS p4d.
#
# Usage:
#
#   ./scripts/phase3_smoke.sh                  # full smoke from scratch
#   SKIP_BUILD=1 ./scripts/phase3_smoke.sh     # skip phases 1+2 (need cached)
#
# What gets exercised:
#
#   * tools/build_tf_store.py (rel-f1, ~30 sec)
#   * tools/precompute_shards.py with --name_prefix (5 tasks, ~1-2 min)
#   * Phase 3 startup + first few training steps + first eval
#     (multi-task path, unified_type_map, shard remap, encoder
#     chunk + checkpoint, codebook sync). Detects:
#       - shard schema / type-id mismatch (fails inside step 1)
#       - encoder transformer kernel limit / OOM
#       - codebook all_reduce hang
#       - DDP setup / barrier deadlock
#
# Pre-flight pkill, watchdog, and MASTER_PORT override are inherited
# from pretrain_p4d.sh. Override any of these before invocation:
#
#   MASTER_PORT=29550 NPROC=1 ./scripts/phase3_smoke.sh
#
# This script is a thin wrapper -- it sets DATASETS / BATCH / K /
# EPOCHS / STEPS_PER_TASK / NPROC to small values and delegates to
# the main launcher. The launcher's existing failure modes (.done
# sentinel checks, watchdog, etc.) all apply.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Small-scale config. These are tuned to surface the multi-task
# code paths (multiple tasks per dataset, unified type map, shard
# remap) while staying fast on a single GPU.
export DATASETS="${DATASETS:-rel-f1}"
export BATCH="${BATCH:-16}"
export K="${K:-16}"
export CHANNELS="${CHANNELS:-64}"
export NUM_LAYERS="${NUM_LAYERS:-1}"
export HEADS="${HEADS:-2}"
export CENTROIDS="${CENTROIDS:-128}"
export EPOCHS="${EPOCHS:-1}"
export STEPS_PER_TASK="${STEPS_PER_TASK:-3}"
export NPROC="${NPROC:-1}"
export PARALLEL_TF_BUILDS="${PARALLEL_TF_BUILDS:-1}"
export PARALLEL_SHARD_BUILDS="${PARALLEL_SHARD_BUILDS:-1}"
export OUT_DIR="${OUT_DIR:-results/phase3_smoke}"
export RUN_NAME="${RUN_NAME:-phase3_smoke}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

# A different default rdzv port so a stuck production-port doesn't
# block the smoke run in parallel.
export MASTER_PORT="${MASTER_PORT:-29555}"

# Loosen the load-concurrency since we're single-rank (it's already
# bounded by NPROC=1).
export LOAD_CONCURRENCY="${LOAD_CONCURRENCY:-1}"

echo "================================================================"
echo "phase3_smoke: end-to-end smoke on rel-f1"
echo "  DATASETS=$DATASETS  NPROC=$NPROC"
echo "  BATCH=$BATCH  K=$K  CHANNELS=$CHANNELS  LAYERS=$NUM_LAYERS"
echo "  EPOCHS=$EPOCHS  STEPS_PER_TASK=$STEPS_PER_TASK"
echo "  OUT_DIR=$OUT_DIR  MASTER_PORT=$MASTER_PORT"
echo "  WANDB_MODE=$WANDB_MODE"
echo "================================================================"
echo

t0=$(date +%s)
bash "$REPO_ROOT/scripts/pretrain_p4d.sh" "$@"
rc=$?
t1=$(date +%s)
echo
echo "================================================================"
if [ "$rc" -eq 0 ]; then
    echo "phase3_smoke: PASSED in $(( t1 - t0 ))s"
else
    echo "phase3_smoke: FAILED (rc=$rc) in $(( t1 - t0 ))s"
    echo "  log: $OUT_DIR/train.log"
fi
echo "================================================================"
exit $rc
