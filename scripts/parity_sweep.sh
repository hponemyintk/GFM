#!/usr/bin/env bash
# ML3.5 parity sweep: dev-kyaw vs new pipeline on rel-f1 driver-position
# (regression) and rel-f1 driver-top3 (classification), N seeds each.
#
# Both pipelines run from their own git worktree so we don't have to
# stash/checkout in this one:
#
#   dev-kyaw branch:  /home/jedi/research_repos/GFM/.claude/worktrees/dev-kyaw
#   new branch:       /home/jedi/research_repos/GFM/.claude/worktrees/gfm-worktree
#
# Usage: ./scripts/parity_sweep.sh [num_seeds] [epochs]
#   num_seeds   default 5
#   epochs      default 5  (10 in plan, 5 for laptop time budget)
#
# CACHE BEHAVIOR (default: WIPE BEFORE RUN):
#   By default, both ``results/parity/{devkyaw,new}/rel-f1`` are wiped
#   at the start of every sweep. This is the safe default after we
#   discovered that stale .json files from a killed prior sweep were
#   silently being treated as "already done" by the existence-check
#   below, mixing broken old results into fresh metric tables.
#   To preserve prior results, set ``KEEP_CACHE=1``.
set -euo pipefail

NUM_SEEDS="${1:-5}"
EPOCHS="${2:-5}"
TASKS=("driver-position" "driver-top3")

NEW_DIR="/home/jedi/research_repos/GFM/.claude/worktrees/gfm-worktree"
OLD_DIR="/home/jedi/research_repos/GFM/.claude/worktrees/dev-kyaw"
RESULTS_ROOT="$NEW_DIR/results/parity"
mkdir -p "$RESULTS_ROOT/devkyaw" "$RESULTS_ROOT/new"

# Per-pipeline cache dirs. Even when both pipelines use the same
# ``upto_test_timestamp`` setting, scoping caches per branch is cheap
# defense-in-depth against subtle differences (encoder version skew,
# precomputed-shard format drift across PRs, etc.) silently poisoning
# the comparison.
DEVKYAW_CACHE="$HOME/.cache/relbench_examples_parity_devkyaw"
NEW_CACHE="$HOME/.cache/relbench_examples_parity_new"

# Wipe stale "new" results + cache. The dev-kyaw side is the
# historical baseline -- it doesn't change across PRs, so we keep
# its results + materialized/HDF5 cache across sweeps to save ~10
# runs per sweep (~15 min wallclock).
#
# Stale .json from a killed prior sweep (with broken code) on the
# "new" side would otherwise be skipped as "already done" and
# pollute the new comparison; same for HDF5/materialized. So we
# unconditionally wipe the "new" side by default.
#
# Two opt-out / opt-in env vars:
#   KEEP_CACHE=1            -- skip ALL wipes (dangerous; use only
#                              when manually iterating without code
#                              changes)
#   FORCE_REWIPE_DEVKYAW=1  -- also wipe dev-kyaw (rare; only when
#                              rebuilding the historical baseline,
#                              e.g. dev-kyaw branch advanced or the
#                              relbench data changed)
if [ "${KEEP_CACHE:-0}" != "1" ]; then
  echo "=== wiping NEW parity results + cache (dev-kyaw kept) ==="
  rm -rf "$RESULTS_ROOT/new/rel-f1"
  rm -f "$RESULTS_ROOT/new"/*.log
  rm -rf "$NEW_CACHE/precomputed/rel-f1" "$NEW_CACHE/rel-f1"
  if [ "${FORCE_REWIPE_DEVKYAW:-0}" = "1" ]; then
    echo "=== also wiping dev-kyaw (FORCE_REWIPE_DEVKYAW=1) ==="
    rm -rf "$RESULTS_ROOT/devkyaw/rel-f1"
    rm -f "$RESULTS_ROOT/devkyaw"/*.log
    rm -rf "$DEVKYAW_CACHE/precomputed/rel-f1" "$DEVKYAW_CACHE/rel-f1"
  fi
fi
mkdir -p "$DEVKYAW_CACHE" "$NEW_CACHE"

export WANDB_MODE=offline
export WANDB_SILENT=true

run_one() {
  local pipeline_dir="$1"
  local pipeline_tag="$2"
  local task="$3"
  local seed="$4"
  local out_dir="$RESULTS_ROOT/$pipeline_tag"

  # Pick the right per-pipeline cache. dev-kyaw and new branch
  # materialize different graphs (see DEVKYAW_CACHE / NEW_CACHE comment
  # above) so they MUST not share a cache root.
  local cache_dir
  if [ "$pipeline_tag" = "devkyaw" ]; then
    cache_dir="$DEVKYAW_CACHE"
  else
    cache_dir="$NEW_CACHE"
  fi

  local log="$out_dir/${task}_s${seed}.log"
  local result_path="$out_dir/rel-f1/${task}/${seed}.json"

  if [ -f "$result_path" ]; then
    echo "  [$pipeline_tag $task seed=$seed] already done -> $result_path"
    return 0
  fi

  echo "  [$pipeline_tag $task seed=$seed] running ..."
  ( cd "$pipeline_dir" && \
    torchrun --nproc_per_node 1 main_node_ddp.py \
      --dataset rel-f1 --task "$task" \
      --seed "$seed" --epochs "$EPOCHS" \
      --batch_size 128 --num_neighbors 64 \
      --channels 128 --num_layers 1 --num_heads 4 \
      --num_centroids 512 \
      --num_workers 2 \
      --cache_dir "$cache_dir" \
      --out_dir "$out_dir" \
      --run_name "${pipeline_tag}-${task}-s${seed}" \
      > "$log" 2>&1 ) || {
        echo "    FAILED -- see $log"
        return 1
      }
  echo "    OK -> $result_path"
}

echo "=== ML3.5 parity sweep: $NUM_SEEDS seeds x ${#TASKS[@]} tasks x 2 pipelines, ${EPOCHS} epochs each ==="
TOTAL=$((NUM_SEEDS * ${#TASKS[@]} * 2))
i=0
for task in "${TASKS[@]}"; do
  for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    i=$((i + 1)); echo "[$i/$TOTAL] dev-kyaw $task seed=$seed"
    run_one "$OLD_DIR" "devkyaw" "$task" "$seed" || true
    i=$((i + 1)); echo "[$i/$TOTAL] new $task seed=$seed"
    run_one "$NEW_DIR" "new" "$task" "$seed" || true
  done
done

echo
echo "=== Aggregating results ==="
python3 "$NEW_DIR/scripts/aggregate_parity.py" "$RESULTS_ROOT"
