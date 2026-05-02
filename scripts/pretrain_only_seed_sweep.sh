#!/usr/bin/env bash
# Pretrain-only N-seed sweep -- no cross-dataset adoption.
#
# Trains N independent multi-task backbones on the SOURCE datasets,
# one per seed in PRETRAIN_SEEDS, and aggregates per-task TEST
# metrics across the trials. Useful when you want pretrain-side
# variance bounds on a fixed task list (e.g. rel-f1 + rel-event)
# without doing any cross-dataset adoption.
#
# Per-trial seed isolation matches scripts/holdout_dataset_eval_pretrain_sweep.sh:
#   SEED=$PSEED                pretrain --seed (model init,
#                              DataLoader shuffle, multi-task
#                              sampler).
#   PYTHONHASHSEED=$PSEED      pins CPython's hash() so the per-row
#                              seed_val derivation in
#                              gfm_data/sampler.py is
#                              deterministically tied to the trial.
#   SHARDS_SUBDIR=pretrain_seedN
#                              namespaces the precomputed shards
#                              under $CACHE/shards[_full]/<sub>/ so
#                              each trial rebuilds its neighbor
#                              list from scratch instead of reusing
#                              trial 0's cache. tf_store and
#                              materialization stay shared.
#
# Pretrain artifacts per trial:
#   <OUT>/pretrain_seed<N>/multi_task/<N>.json  test metrics JSON
#   <OUT>/pretrain_seed<N>/multi_task/best_backbone.pt + meta + schema
#
# Aggregate:
#   <OUT>/<src_slug>/aggregate.json  per-task per-metric mean +/- SD
#
# Auto-enumeration in pretrain_p4d.sh now drops 5 RelBench v2 tasks
# whose supervised single-task GNN baseline (per the v2 paper) sits
# at or below random -- they're noise, not signal. The 5 are:
#   rel-event.event_interest-interested      AUC 0.4764 (below random)
#   rel-event.event_interest-not_interested  AUC 0.6040 (~random)
#   rel-event.users-birthyear                R^2 -0.030
#   rel-trial.site-success                   R^2 -0.483
#   rel-amazon.item-ltv                      R^2  0.032
# Defaults to dropping them via EXCLUDED_TASKS in pretrain_p4d.sh.
# Override EXCLUDED_TASKS="" to include every task; set TASKS_CSV
# explicitly to bypass the filter entirely.
#
# Recommended invocation -- 3 trials over rel-f1 + rel-event,
# auto-enumerated entity binary/regression tasks (low-quality
# tasks dropped by default):
#
#   NPROC=8 SHARD_WORKERS=10 \
#     PRETRAIN_SEEDS="0 1 2" \
#     SOURCE="rel-f1 rel-event" \
#     EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/pretrain_only_seed_sweep.sh
#
# Resolves to 8 tasks total: 5 rel-f1 (driver-position, driver-dnf,
# driver-top3, results-position, qualifying-position) + 3 rel-event
# (user-attendance, user-repeat, user-ignore). The 3 rel-event
# low-quality entries are excluded automatically.
#
# To override and include every task (the all-tasks ablation):
#
#   EXCLUDED_TASKS="" NPROC=8 SHARD_WORKERS=10 \
#     PRETRAIN_SEEDS="0 1 2" \
#     SOURCE="rel-f1 rel-event" \
#     EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/pretrain_only_seed_sweep.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-0 1 2}"
SOURCE="${SOURCE:-rel-f1 rel-event}"
EPOCHS="${EPOCHS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-500}"
NPROC="${NPROC:-8}"
SHARD_WORKERS="${SHARD_WORKERS:-10}"
TASKS_CSV="${TASKS_CSV:-}"
FULL_GRAPH="${FULL_GRAPH:-1}"
OUT_DIR_BASE="${OUT_DIR:-results/pretrain_only_seed_sweep}"
# Force-rebuild the per-trial precomputed shard tree under
# $CACHE/shards[_full]/pretrain_seed<N>/ before each trial. Default
# 0 (fast iteration): pretrain_p4d.sh's .done sentinel reuses
# shards from a prior invocation, which is what you want when only
# the training code changed. Set FORCE_REBUILD_SHARDS=1 when the
# sampler / seed_time / truncation policy moved between runs and
# you need neighbor lists rebuilt to match the new code.
FORCE_REBUILD_SHARDS="${FORCE_REBUILD_SHARDS:-0}"
# Honor an explicit CACHE_DIR override; pretrain_p4d.sh does the
# same so the wipe target matches whatever Phase 2 will write to.
CACHE_LOCAL="${CACHE_DIR:-$HOME/.cache/relbench_examples}"

# Build slug + comma-separated DATASETS filter for pretrain_p4d.sh.
SOURCE_LIST=( $SOURCE )
SOURCE_SLUG=$(IFS=+; echo "${SOURCE_LIST[*]}")
DATASETS_CSV=$(IFS=,; echo "${SOURCE_LIST[*]}")

SWEEP_DIR="$OUT_DIR_BASE/${SOURCE_SLUG}"
mkdir -p "$SWEEP_DIR"

echo "==================================================================="
echo "Pretrain-only seed sweep (no cross-dataset adoption)"
echo "  PRETRAIN_SEEDS : [$PRETRAIN_SEEDS]"
echo "  SOURCE         : $SOURCE"
echo "  TASKS_CSV      : ${TASKS_CSV:-<auto-enumerate>}"
echo "  EPOCHS         : $EPOCHS  STEPS_PER_TASK: $STEPS_PER_TASK"
echo "  NPROC          : $NPROC   SHARD_WORKERS : $SHARD_WORKERS"
echo "  FULL_GRAPH     : $FULL_GRAPH"
echo "  out            : $SWEEP_DIR"
echo "==================================================================="

# ---- Per-trial pretrain ----
PSEED_DIRS=()
for PSEED in $PRETRAIN_SEEDS; do
  PSEED_DIR="$SWEEP_DIR/pretrain_seed${PSEED}"
  mkdir -p "$PSEED_DIR"
  PSEED_DIRS+=("$PSEED_DIR")

  echo
  echo "------------------------------------------------------------------"
  echo "[$(date)] PRETRAIN_SEED=$PSEED  ->  $PSEED_DIR"
  echo "------------------------------------------------------------------"

  # Defensive shard wipe so a prior invocation in this container
  # cannot silently feed stale neighbor lists into this trial.
  # Wipes both shards/ and shards_full/ subtrees because either
  # FULL_GRAPH mode could have produced the leftover sentinel.
  if [ "$FORCE_REBUILD_SHARDS" = "1" ]; then
    for _shroot in "$CACHE_LOCAL/shards" "$CACHE_LOCAL/shards_full"; do
      _shpath="$_shroot/pretrain_seed${PSEED}"
      if [ -d "$_shpath" ]; then
        echo "[$(date)] [force-rebuild] wiping $_shpath"
        rm -rf "$_shpath"
      fi
    done
  fi

  # Optional TASKS_CSV. ${TASKS_CSV:+...} expands to the kvp only
  # when TASKS_CSV is non-empty so pretrain_p4d.sh's
  # auto-enumerate path stays the default.
  PYTHONHASHSEED="$PSEED" \
  SEED="$PSEED" \
  SHARDS_SUBDIR="pretrain_seed${PSEED}" \
  DATASETS="$DATASETS_CSV" \
  ${TASKS_CSV:+TASKS_CSV="$TASKS_CSV"} \
  NPROC="$NPROC" SHARD_WORKERS="$SHARD_WORKERS" \
  EPOCHS="$EPOCHS" STEPS_PER_TASK="$STEPS_PER_TASK" \
  FULL_GRAPH="$FULL_GRAPH" \
  OUT_DIR="$PSEED_DIR" \
  RUN_NAME="pretrain_${SOURCE_SLUG}_seed${PSEED}" \
  bash "$REPO_ROOT/scripts/pretrain_p4d.sh"
done

# ---- Aggregate per-task test metrics across trials ----
AGG="$SWEEP_DIR/aggregate.json"
PSEED_DIRS_STR=$(printf '%s\n' "${PSEED_DIRS[@]}")

python3 - <<PY
import json, os, statistics
pseed_dirs = """$PSEED_DIRS_STR""".strip().split("\n")
pretrain_seeds = "$PRETRAIN_SEEDS".split()

def _mean_sd(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return {"mean": vals[0], "sd": None, "n": 1, "values": vals}
    return {
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals),
        "n": len(vals),
        "values": vals,
    }

# train_multi_task.py writes test metrics to
# <OUT_DIR>/multi_task/<seed>.json
# (see train_multi_task.py:1008).
per_task_metric = {}
for pseed, pdir in zip(pretrain_seeds, pseed_dirs):
    json_path = os.path.join(pdir, "multi_task", f"{pseed}.json")
    if not os.path.exists(json_path):
        print(f"# WARN: no test json at {json_path}; "
              f"skipping pretrain_seed={pseed}")
        continue
    with open(json_path) as f:
        data = json.load(f)
    for task_name, metrics in (data.get("test_metrics") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            per_task_metric.setdefault(task_name, {}) \
                           .setdefault(metric_name, []).append({
                "pretrain_seed": pseed,
                "value": value,
            })

out = {
    "source": "$SOURCE",
    "pretrain_seeds": pretrain_seeds,
    "tasks_csv": ("$TASKS_CSV" or "<auto-enumerate>"),
    "per_task": {},
}
for task_name, metrics in per_task_metric.items():
    rolled = {}
    for metric_name, points in metrics.items():
        stats = _mean_sd([p["value"] for p in points])
        if stats is not None:
            stats["per_pretrain_seed"] = points
            rolled[metric_name] = stats
    out["per_task"][task_name] = rolled

with open("$AGG", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "==================================================================="
echo "Pretrain-only sweep done."
echo "  per-trial artifacts: $SWEEP_DIR/pretrain_seed*/multi_task/"
echo "  aggregate          : $AGG"
echo "==================================================================="
