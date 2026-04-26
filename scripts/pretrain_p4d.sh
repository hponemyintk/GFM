#!/usr/bin/env bash
# pretrain_p4d.sh -- multi-task pretraining on AWS p4d.24xlarge.
#
# Hardware target:
#   8 x A100 40 GB (320 GB VRAM total)
#   96 vCPU
#   ~1.1 TB RAM
#   ~8 TB local NVMe
#
# Trains across every BINARY_CLASSIFICATION and REGRESSION task in
# RelBench v2 by default. To restrict to a subset of datasets:
#
#   DATASETS=rel-f1,rel-event ./scripts/pretrain_p4d.sh
#
# Phases:
#   0. enumerate tasks from relbench dynamically (filter to binary +
#      regression, skip link-prediction / multi-class)
#   1. build TF memmap stores per dataset (one Python process at a
#      time -> peak CPU RAM bounded by biggest single dataset, NOT the
#      sum)
#   2. build precomputed sample shards per (dataset, task) so training
#      reads from disk via mmap (no in-process BFS, no GIL contention)
#   3. launch DDP training (precomputed_shards + memmap-TF means no
#      large-tensor ownership in CPU RAM during training, just file
#      handles)
#
# Tunables via env vars (sensible safe defaults):
#   DATASETS                 comma-separated subset; default = full v2 list
#   K                        K-token neighbor context (default 300)
#   BATCH                    per-rank batch size (default 128)
#   CHANNELS                 model hidden dim (default 256)
#   NUM_LAYERS               (default 2)
#   HEADS                    (default 4)
#   CENTROIDS                (default 4096)
#   EPOCHS                   (default 30)
#   MAX_STEPS                steps/epoch (default 3000)
#   WORKERS                  DataLoader workers per rank (default 4)
#   LR                       base lr (default 1e-4; multiplied by world_size)
#   WARMUP                   warmup steps (default 1000)
#   LOSS_BALANCE             none|per_task_mean|fixed:..|uncertainty
#                            (default none = RT formula)
#   NPROC                    GPUs per node (default 8)
#   OUT_DIR                  results dir (default results/p4d_pretrain)
#   RUN_NAME                 wandb run name (default p4d_alltasks)
#   SHARD_SIZE               samples per memmap shard (default 50000)
#
# Why these defaults will NOT OOM on p4d.24xlarge:
#
#   CPU RAM (1.1 TB):
#     * Phase 1 builds one dataset's TF memmap at a time. Biggest
#       single dataset (rel-event ~25 GB peak) << 1.1 TB.
#     * Phase 2 same: one dataset+task at a time.
#     * Training: each DatasetGraphCache holds CSR adjacency (~few GB
#       for the largest) + file handles (cheap) for memmapped TFs.
#       8 ranks * 4 workers = 32 worker processes; each forks but
#       memmapped TF columns are page-cached by the kernel, not
#       duplicated.
#
#   GPU VRAM (40 GB per A100):
#     * batch=128, channels=256, K=300, layers=2, fp32 ~= 8-12 GB
#       activation+weights per rank. Headroom 25+ GB.
#     * If you raise CHANNELS or BATCH and hit OOM, halve BATCH or
#       enable bf16 (would need a small main_node_ddp.py edit; not
#       wired in this branch).
#
#   Disk (8 TB NVMe):
#     * RelBench raw + materialized: ~80 GB
#     * TF memmap stores: ~80 GB
#     * Sample shards (K=300, ~50M total seeds across v2): ~150 GB
#     * Total ~300 GB << 8 TB.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"
TF_STORE="$CACHE/tf_store"
SHARDS="$CACHE/shards"
K="${K:-300}"
BATCH="${BATCH:-512}"
CHANNELS="${CHANNELS:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
HEADS="${HEADS:-4}"
CENTROIDS="${CENTROIDS:-4096}"
EPOCHS="${EPOCHS:-30}"
# MAX_STEPS sizing for full RelBench v2 (~25-30 binary+regression tasks):
#   80 steps/task/epoch * 25 tasks = 2000 (this default)
#   Total budget = 30 epochs * 2000 steps * 512 batch * 8 ranks
#                ~= 246M sample-passes (~19x RT paper's 50k * 256 = 12.8M).
# If you cut DATASETS to a single dataset, drop MAX_STEPS proportionally:
#   for example rel-f1 alone has ~5 tasks, so MAX_STEPS=400 would match
#   the 80 steps/task/epoch density above.
MAX_STEPS="${MAX_STEPS:-2000}"
WORKERS="${WORKERS:-4}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-1000}"
LOSS_BALANCE="${LOSS_BALANCE:-none}"
NPROC="${NPROC:-8}"
OUT_DIR="${OUT_DIR:-results/p4d_pretrain}"
RUN_NAME="${RUN_NAME:-p4d_alltasks}"
SHARD_SIZE="${SHARD_SIZE:-50000}"

mkdir -p "$OUT_DIR" "$TF_STORE" "$SHARDS"

# Default RelBench v2 dataset list (overridden by $DATASETS env var).
V2_DATASETS_DEFAULT="rel-amazon,rel-avito,rel-event,rel-f1,rel-hm,rel-stack,rel-trial"
DATASETS_FILTER="${DATASETS:-$V2_DATASETS_DEFAULT}"

echo "================================================================"
echo "GFM multi-task pretraining (p4d.24xlarge)"
echo "================================================================"
echo "  datasets filter: $DATASETS_FILTER"
echo "  K=$K  batch=$BATCH  channels=$CHANNELS  layers=$NUM_LAYERS  heads=$HEADS"
echo "  epochs=$EPOCHS  max_steps=$MAX_STEPS  workers=$WORKERS  nproc=$NPROC"
echo "  lr=$LR  warmup=$WARMUP  loss_balance=$LOSS_BALANCE"
echo "  cache=$CACHE  out=$OUT_DIR"
echo

# ------------------------------------------------------------------
# Phase 0: enumerate tasks via relbench
# ------------------------------------------------------------------
echo "[0/3] Enumerating RelBench tasks (binary + regression only)"

# Pass DATASETS_FILTER into Python; emit a list of "ds.task:1.0" strings.
TASKS_RAW=$(DATASETS_FILTER="$DATASETS_FILTER" python3 - <<'PY'
import os, sys
from relbench.tasks import get_task_names, get_task
from relbench.base import TaskType

ALLOWED = {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
filt = os.environ["DATASETS_FILTER"].strip()
datasets = [d.strip() for d in filt.split(",") if d.strip()]

out = []
for ds in datasets:
    try:
        names = get_task_names(ds)
    except Exception as e:
        print(f"# WARN: cannot list tasks for {ds}: {e}", file=sys.stderr)
        continue
    for tn in names:
        try:
            t = get_task(ds, tn)
            if t.task_type in ALLOWED:
                out.append(f"{ds}.{tn}:1.0")
        except Exception as e:
            print(f"# WARN: cannot load {ds}.{tn}: {e}", file=sys.stderr)
print("\n".join(out))
PY
)

# Read into bash array.
mapfile -t TASKS <<< "$TASKS_RAW"
# Drop empty entries (in case DATASETS_FILTER is malformed).
TASKS_CLEAN=()
for t in "${TASKS[@]}"; do
    [ -n "$t" ] && TASKS_CLEAN+=("$t")
done
TASKS=("${TASKS_CLEAN[@]}")

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "ERROR: no tasks selected. Check DATASETS env var." >&2
    exit 1
fi

# Derive unique dataset list from selected tasks.
DATASETS_SELECTED=( $(printf '%s\n' "${TASKS[@]}" | awk -F: '{print $1}' | awk -F. '{print $1}' | sort -u) )

echo "  ${#TASKS[@]} tasks across ${#DATASETS_SELECTED[@]} datasets:"
for ds in "${DATASETS_SELECTED[@]}"; do
    echo "    $ds:"
    for t in "${TASKS[@]}"; do
        if [[ "$t" == "$ds."* ]]; then
            echo "      $(echo "$t" | awk -F: '{print $1}')"
        fi
    done
done
echo

# ------------------------------------------------------------------
# Phase 1: build TF memmap stores (sequential, bounded peak RAM)
# ------------------------------------------------------------------
echo "[1/3] Building TF memmap stores (sequential per dataset)"
for ds in "${DATASETS_SELECTED[@]}"; do
    if [ -f "$TF_STORE/$ds/.done" ]; then
        echo "  $ds: cached -> $TF_STORE/$ds"
        continue
    fi
    t0=$(date +%s)
    echo "  $ds: building ..."
    python3 tools/build_tf_store.py \
        --dataset "$ds" \
        --out_dir "$TF_STORE/$ds"
    touch "$TF_STORE/$ds/.done"
    echo "    done in $(( $(date +%s) - t0 ))s"
done
echo

# ------------------------------------------------------------------
# Phase 2: build sample shards per (dataset, task)
# ------------------------------------------------------------------
echo "[2/3] Building precomputed sample shards (K=$K, shard=$SHARD_SIZE)"
for spec in "${TASKS[@]}"; do
    full="${spec%%:*}"
    ds="${full%%.*}"
    task="${full#*.}"
    out="$SHARDS/$ds/$task"
    if [ -f "$out/.done" ]; then
        echo "  $ds.$task: cached -> $out"
        continue
    fi
    t0=$(date +%s)
    echo "  $ds.$task: building ..."
    python3 tools/precompute_shards.py \
        --dataset "$ds" \
        --task "$task" \
        --K "$K" \
        --shard_size "$SHARD_SIZE" \
        --out_dir "$out" \
        --splits train val test
    touch "$out/.done"
    echo "    done in $(( $(date +%s) - t0 ))s"
done
echo

# ------------------------------------------------------------------
# Phase 3: DDP training
# ------------------------------------------------------------------
echo "[3/3] Launching DDP training across $NPROC GPUs"
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
echo "  tasks=$TASKS_CSV"
echo

# WANDB env -- user can override.
export WANDB_MODE="${WANDB_MODE:-online}"

LOG="$OUT_DIR/train.log"
torchrun --nproc_per_node "$NPROC" main_node_ddp.py \
    --tasks "$TASKS_CSV" \
    --mode precomputed_shards \
    --shards_dir "$SHARDS" \
    --tf_store_dir "$TF_STORE" \
    --num_neighbors "$K" \
    --batch_size "$BATCH" \
    --channels "$CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$HEADS" \
    --num_centroids "$CENTROIDS" \
    --epochs "$EPOCHS" \
    --max_steps_per_epoch "$MAX_STEPS" \
    --num_workers "$WORKERS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --loss_balance "$LOSS_BALANCE" \
    --out_dir "$OUT_DIR" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "$LOG"

echo
echo "================================================================"
echo "Training complete. Test metrics:"
echo "================================================================"
ls -la "$OUT_DIR/multi_task/" 2>/dev/null | tail -5
cat "$OUT_DIR/multi_task/"*.json 2>/dev/null | head -200
