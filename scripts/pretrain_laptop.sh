#!/usr/bin/env bash
# Laptop-scope multi-task pretraining (this RTX 5070, 12 GB VRAM).
#
# Trains across all regression/binary tasks in rel-f1 + rel-event using
# streaming mode + memmap-TF so we don't have to precompute shards or
# hold the full rel-event TF in RAM. Reports per-task test metrics at
# the end.
#
# 11 tasks: 5 from rel-f1 + 6 from rel-event.
#
# Memory math on this box (27 GB RAM, 12 GB VRAM):
#   rel-event materialization (one-time):  ~14 GB peak RAM
#   rel-event TF memmap on disk:           ~3 GB
#   rel-f1   TF memmap on disk:           ~50 MB
#   CSR adjacencies (both, in RAM):        ~2 GB
#   Per-step training memory:              ~3-4 GB VRAM at the configs below
#
# Total wall time estimate on this box:
#   rel-event memmap build: ~10-15 min (one-time)
#   training (5 epochs):    ~30-90 min depending on max_steps_per_epoch

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"
TF_STORE="$CACHE/tf_store"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-300}"
K="${K:-32}"
BATCH="${BATCH:-32}"
CHANNELS="${CHANNELS:-64}"
HEADS="${HEADS:-2}"
CENTROIDS="${CENTROIDS:-128}"
MAX_ROWS_TRAIN="${MAX_ROWS_TRAIN:-2000}"  # train-only cap (val/test stay full)
OUT_DIR="${OUT_DIR:-results/pretrain_laptop}"
RUN_NAME="${RUN_NAME:-laptop_alltasks}"

DATASETS=(rel-f1 rel-event)
TASKS=(
  "rel-f1.driver-position:1.0"
  "rel-f1.driver-dnf:1.0"
  "rel-f1.driver-top3:1.0"
  "rel-f1.results-position:1.0"
  "rel-f1.qualifying-position:1.0"
  "rel-event.user-attendance:1.0"
  "rel-event.user-repeat:1.0"
  "rel-event.user-ignore:1.0"
  "rel-event.event_interest-interested:1.0"
  "rel-event.event_interest-not_interested:1.0"
  "rel-event.users-birthyear:1.0"
)
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

export WANDB_MODE=offline
export WANDB_SILENT=true

echo "=== [1/2] Building TF memmap stores (one-time) ==="
for ds in "${DATASETS[@]}"; do
  if [ -f "$TF_STORE/$ds/.done" ]; then
    echo "  $ds: cached"
    continue
  fi
  echo "  $ds: building -> $TF_STORE/$ds (this can take a while for rel-event)"
  python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds"
  touch "$TF_STORE/$ds/.done"
done

echo
echo "=== [2/2] Training $EPOCHS epochs x $MAX_STEPS steps on $K-token subgraphs ==="
echo "  config: BATCH=$BATCH CHANNELS=$CHANNELS HEADS=$HEADS K=$K MAX_ROWS_TRAIN=$MAX_ROWS_TRAIN"
echo "  log:    $LOG"

torchrun --nproc_per_node 1 main_node_ddp.py \
  --tasks "$TASKS_CSV" \
  --mode streaming \
  --tf_store_dir "$TF_STORE" \
  --max_rows_per_task "$MAX_ROWS_TRAIN" \
  --num_neighbors "$K" \
  --batch_size "$BATCH" \
  --channels "$CHANNELS" \
  --num_layers 1 \
  --num_heads "$HEADS" \
  --num_centroids "$CENTROIDS" \
  --epochs "$EPOCHS" \
  --max_steps_per_epoch "$MAX_STEPS" \
  --num_workers 0 \
  --lr 1e-4 --warmup_steps 200 \
  --loss_balance "${LOSS_BALANCE:-none}" \
  --out_dir "$OUT_DIR" \
  --run_name "$RUN_NAME" \
  2>&1 | tee "$LOG"

echo
echo "=== Final test metrics ==="
ls -la "$OUT_DIR/multi_task/" 2>/dev/null | tail -5
cat "$OUT_DIR"/multi_task/*.json 2>/dev/null | head -100
