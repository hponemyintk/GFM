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
set -euo pipefail

NUM_SEEDS="${1:-5}"
EPOCHS="${2:-5}"
TASKS=("driver-position" "driver-top3")

NEW_DIR="/home/jedi/research_repos/GFM/.claude/worktrees/gfm-worktree"
OLD_DIR="/home/jedi/research_repos/GFM/.claude/worktrees/dev-kyaw"
RESULTS_ROOT="$NEW_DIR/results/parity"
mkdir -p "$RESULTS_ROOT/devkyaw" "$RESULTS_ROOT/new"

export WANDB_MODE=offline
export WANDB_SILENT=true

run_one() {
  local pipeline_dir="$1"
  local pipeline_tag="$2"
  local task="$3"
  local seed="$4"
  local out_dir="$RESULTS_ROOT/$pipeline_tag"

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
