#!/usr/bin/env bash
# PR4 reference launcher: 6-task pretraining on rel-f1 + rel-event.
#
# Designed for the 8xA100 / 1TB box per docs/multi_task_refactor_plan.md.
# Loading rel-event TFs into RAM (~14 GB) is fine on that box; the laptop
# would have to use --max_rows_per_task or skip rel-event.
#
# Three phases:
#   1. Build per-dataset TF memmap stores once (offline, ~15 GB peak RAM
#      for rel-event materialization).
#   2. Build precomputed sample shards once per (dataset, task, K).
#   3. Launch DDP training across 8 GPUs.
#
# Total wall time on 8xA100: phase 1 ~5 min, phase 2 ~30-60 min, phase 3
# depends on epochs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"
TF_STORE="$CACHE/tf_store"
SHARDS="$CACHE/shards"
K="${K:-300}"

DATASETS=(rel-f1 rel-event)
# 11 tasks total: every regression + binary task across rel-f1 + rel-event.
# (driver-circuit-compete is link-prediction -- out of current scope.)
TASKS=(
  "rel-f1.driver-position:1.0"               # regression
  "rel-f1.driver-dnf:1.0"                    # binary
  "rel-f1.driver-top3:1.0"                   # binary
  "rel-f1.results-position:1.0"              # regression
  "rel-f1.qualifying-position:1.0"           # regression
  "rel-event.user-attendance:1.0"            # regression
  "rel-event.user-repeat:1.0"                # binary
  "rel-event.user-ignore:1.0"                # binary
  "rel-event.event_interest-interested:1.0"  # binary
  "rel-event.event_interest-not_interested:1.0"  # binary
  "rel-event.users-birthyear:1.0"            # regression
)

echo "=== [1/3] Building TF memmap stores ==="
for ds in "${DATASETS[@]}"; do
  if [ -f "$TF_STORE/$ds/.done" ]; then
    echo "  $ds: already built"
    continue
  fi
  echo "  $ds: building -> $TF_STORE/$ds"
  python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds"
  touch "$TF_STORE/$ds/.done"
done

echo
echo "=== [2/3] Building precomputed sample shards ==="
# One shards dir per (dataset, task). Only build if missing.
for spec in "${TASKS[@]}"; do
  full="${spec%%:*}"
  ds="${full%%.*}"
  task="${full#*.}"
  out="$SHARDS/$ds/$task"
  if [ -f "$out/.done" ]; then
    echo "  $ds.$task: already built"
    continue
  fi
  echo "  $ds.$task: building -> $out"
  python3 tools/precompute_shards.py \
    --dataset "$ds" --task "$task" \
    --K "$K" --shard_size 50000 \
    --out_dir "$out" \
    --splits train val test
  touch "$out/.done"
done

echo
echo "=== [3/3] Launching DDP training ==="
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")

# Tasks share K + tf_store_dir at the *root*; per-task shards_dir is
# selected by main_node_ddp -> train_multi_task at task-build time. (For
# now train_multi_task assumes a flat layout under shards_dir keyed by
# dataset/task; see _build_caches_and_tokens for the path convention.)
NPROC="${NPROC:-8}"
torchrun --nproc_per_node "$NPROC" main_node_ddp.py \
  --tasks "$TASKS_CSV" \
  --mode precomputed_shards \
  --shards_dir "$SHARDS" \
  --tf_store_dir "$TF_STORE" \
  --num_neighbors "$K" \
  --batch_size 256 \
  --channels 256 \
  --num_layers 2 \
  --num_heads 4 \
  --num_centroids 4096 \
  --epochs "${EPOCHS:-30}" \
  --max_steps_per_epoch "${MAX_STEPS:-3000}" \
  --num_workers 8 \
  --lr "${LR:-1e-4}" \
  --warmup_steps 1000 \
  --loss_balance "${LOSS_BALANCE:-none}" \
  --out_dir "${OUT_DIR:-results/pr4_6task}" \
  --run_name "${RUN_NAME:-pr4_6task}"
