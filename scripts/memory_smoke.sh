#!/usr/bin/env bash
# Memory smoke (MS1 / MS3 from docs/multi_task_refactor_plan.md §6.3.6).
#
# Runs main_node_ddp.py with the new memmap-TF + streaming pipeline on
# rel-f1, samples RSS every second, and reports peak / steady-state
# resident memory. Verifies that --max_rows_per_task + --mode streaming
# + --tf_store_dir bounds RAM usage as the plan claims.
#
# Usage: ./scripts/memory_smoke.sh [steps]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEPS="${1:-50}"
DATASET="rel-f1"
TASK="driver-top3"
TF_STORE_DIR="$HOME/.cache/relbench_examples/tf_store/$DATASET"
RESULTS_DIR="$REPO_ROOT/results/memory_smoke"
mkdir -p "$RESULTS_DIR"

echo "=== Memory smoke ($DATASET / $TASK, ${STEPS} train steps) ==="

# 1. Build TF store if missing.
if [ ! -f "$TF_STORE_DIR/drivers/meta.json" ]; then
  echo "[1/3] Building TF memmap store -> $TF_STORE_DIR"
  python3 tools/build_tf_store.py --dataset "$DATASET" --out_dir "$TF_STORE_DIR"
else
  echo "[1/3] TF store already exists at $TF_STORE_DIR"
fi

# 2. Launch training in the background, sample RSS while it runs.
echo "[2/3] Launching training (--mode streaming, --tf_store_dir, --max_rows_per_task 1000)"

# WANDB offline so it doesn't try to authenticate.
export WANDB_MODE=offline
export WANDB_SILENT=true

LOG="$RESULTS_DIR/run.log"
RSS_LOG="$RESULTS_DIR/rss.csv"
rm -f "$LOG" "$RSS_LOG"

torchrun --nproc_per_node 1 main_node_ddp.py \
    --dataset "$DATASET" --task "$TASK" \
    --mode streaming \
    --tf_store_dir "$TF_STORE_DIR" \
    --max_rows_per_task 1000 \
    --epochs 1 --batch_size 64 --num_neighbors 32 \
    --max_steps_per_epoch "$STEPS" \
    --num_workers 0 \
    --channels 64 --num_layers 1 --num_heads 2 \
    --num_centroids 256 \
    --out_dir "$RESULTS_DIR" \
    --run_name "ms_smoke" \
    > "$LOG" 2>&1 &
TRAIN_PID=$!

echo "ts,rss_mb" > "$RSS_LOG"
START=$(date +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  ts=$(($(date +%s) - START))
  # Sum RSS across the parent + any python children (DataLoader workers).
  rss_kb=$(ps -o rss= --pid "$TRAIN_PID" --ppid "$TRAIN_PID" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  rss_mb=$(awk "BEGIN {printf \"%.1f\", $rss_kb/1024}")
  echo "$ts,$rss_mb" >> "$RSS_LOG"
  sleep 1
done

wait "$TRAIN_PID" || true

# 3. Summarize.
echo "[3/3] Summary"
if [ -s "$RSS_LOG" ]; then
  python3 - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open("results/memory_smoke/rss.csv")))
if not rows:
    print("no samples"); raise SystemExit
xs = [float(r["rss_mb"]) for r in rows]
print(f"  samples:    {len(xs)}")
print(f"  peak RSS:   {max(xs):8.1f} MB")
print(f"  median RSS: {statistics.median(xs):8.1f} MB")
print(f"  final RSS:  {xs[-1]:8.1f} MB")
PY
else
  echo "  (no RSS samples collected)"
fi

echo
echo "Last 30 lines of training log:"
tail -30 "$LOG"
