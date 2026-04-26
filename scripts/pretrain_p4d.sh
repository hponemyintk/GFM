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
# RelBench v2 by default. Restrict to a dataset subset via DATASETS env.
#
# Examples:
#
#   # Full RelBench v2, 3-day budget (~25 tasks, EPOCHS=20):
#   STEPS_PER_TASK=1000 EPOCHS=20 ./scripts/pretrain_p4d.sh
#
#   # rel-f1 + rel-event only (~11 tasks, ~30h wall):
#   DATASETS=rel-f1,rel-event STEPS_PER_TASK=1000 EPOCHS=20 \
#       ./scripts/pretrain_p4d.sh
#
#   # rel-f1 + rel-event quick shake-out (~1h wall):
#   DATASETS=rel-f1,rel-event STEPS_PER_TASK=200 EPOCHS=3 \
#       ./scripts/pretrain_p4d.sh
#
#   # Single dataset:
#   DATASETS=rel-f1 STEPS_PER_TASK=500 EPOCHS=10 \
#       ./scripts/pretrain_p4d.sh
#
# Offline / no-internet AWS pods:
#
# This script downloads two things on first run that you should
# pre-cache if the pod has no outbound internet:
#
#   1. RelBench raw data (~/.cache/relbench/<ds>/db/) -- triggered by
#      relbench's get_dataset(name, download=True). Pre-populate by
#      running once on a box with internet, then rsync/copy the
#      ~/.cache/relbench tree to the pod.
#
#   2. SentenceTransformer GloVe model (~/.cache/huggingface/hub/) --
#      downloaded by sentence_transformers on first SentenceTransformer
#      call. Pre-cache via:
#         huggingface-cli download \
#           sentence-transformers/average_word_embeddings_glove.6B.300d
#      then rsync ~/.cache/huggingface to the pod.
#
# Once both caches are populated, every subsequent run is fully offline.
# WANDB_MODE defaults to 'offline' so wandb.init never reaches out.
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
# Defaults match expts/run-large-base-experiments.sh per-task budget,
# adjusted for batch=512 (half of expts' 1024) on 8-GPU DDP.
BATCH="${BATCH:-512}"
CHANNELS="${CHANNELS:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
HEADS="${HEADS:-4}"
CENTROIDS="${CENTROIDS:-4096}"
FF_DROPOUT="${FF_DROPOUT:-0.3}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.3}"
EPOCHS="${EPOCHS:-10}"

# Per-task step budget. expts/run-large-base-experiments.sh uses 500
# (single-task, batch=1024 single-GPU). With our equal-weighted multi-
# task sampler, total MAX_STEPS scales linearly with task count so each
# task gets the same per-epoch density it would in the single-task run:
#
#   MAX_STEPS_total = STEPS_PER_TASK x N_tasks
#
# Auto-computed below. Override MAX_STEPS directly to bypass the auto.
STEPS_PER_TASK="${STEPS_PER_TASK:-500}"
WORKERS="${WORKERS:-4}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-1000}"
LOSS_BALANCE="${LOSS_BALANCE:-none}"
NPROC="${NPROC:-8}"
OUT_DIR="${OUT_DIR:-results/p4d_pretrain}"
RUN_NAME="${RUN_NAME:-p4d_alltasks}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
# Phase 1 (TF memmap) and Phase 2 (shards) have different bottlenecks:
#   * Phase 1 is GPU-bound (text embedding). Default concurrency
#     PARALLEL_TF_BUILDS = NPROC, one A100 per dataset.
#   * Phase 2 is RAM-bound. Each process loads make_pkey_fkey_graph
#     and even with the post-load TF drop, peak per-process RSS can
#     still hit ~5-10 GB on rel-event during seed-pass-build. With
#     PARALLEL_BUILDS=8 concurrent shard builders that's 40-80 GB --
#     fine on p4d's 1.1 TB but easy to OOM on smaller pods. Default
#     PARALLEL_SHARD_BUILDS = min(NPROC, 4) keeps it conservative.
#
# Both can be overridden independently. PARALLEL_BUILDS is kept as a
# back-compat catch-all that overrides BOTH if neither phase-specific
# var is set.
PARALLEL_BUILDS="${PARALLEL_BUILDS:-$NPROC}"
PARALLEL_TF_BUILDS="${PARALLEL_TF_BUILDS:-$PARALLEL_BUILDS}"
_default_shard_builds=$(( NPROC < 4 ? NPROC : 4 ))
PARALLEL_SHARD_BUILDS="${PARALLEL_SHARD_BUILDS:-$_default_shard_builds}"

mkdir -p "$OUT_DIR" "$TF_STORE" "$SHARDS"

# Validate concurrency knobs early (catch typos like PARALLEL_BUILDS=0).
for _name in PARALLEL_BUILDS PARALLEL_TF_BUILDS PARALLEL_SHARD_BUILDS NPROC; do
    _val="${!_name}"
    if ! [[ "$_val" =~ ^[0-9]+$ ]] || [ "$_val" -lt 1 ]; then
        echo "ERROR: $_name must be a positive integer (got '$_val')." >&2
        exit 1
    fi
done

# Default RelBench v2 dataset list (overridden by $DATASETS env var).
V2_DATASETS_DEFAULT="rel-amazon,rel-avito,rel-event,rel-f1,rel-hm,rel-stack,rel-trial"
DATASETS_FILTER="${DATASETS:-$V2_DATASETS_DEFAULT}"

echo "================================================================"
echo "GFM multi-task pretraining (p4d.24xlarge)"
echo "================================================================"
echo "  datasets filter: $DATASETS_FILTER"
echo "  K=$K  batch=$BATCH  channels=$CHANNELS  layers=$NUM_LAYERS  heads=$HEADS"
echo "  ff_dropout=$FF_DROPOUT  attn_dropout=$ATTN_DROPOUT"
echo "  epochs=$EPOCHS  steps_per_task=$STEPS_PER_TASK  workers=$WORKERS  nproc=$NPROC"
echo "  parallel_tf=$PARALLEL_TF_BUILDS  parallel_shard=$PARALLEL_SHARD_BUILDS"
echo "  lr=$LR  warmup=$WARMUP  loss_balance=$LOSS_BALANCE"
echo "  cache=$CACHE  out=$OUT_DIR"
echo

# ------------------------------------------------------------------
# Phase 0: enumerate tasks via relbench
# ------------------------------------------------------------------
echo "[0/3] Enumerating RelBench tasks (binary + regression only)"

# Pass DATASETS_FILTER into Python; emit a list of "ds.task:1.0" strings.
TASKS_RAW=$(DATASETS_FILTER="$DATASETS_FILTER" python3 - <<'PY'
import contextlib
import os
import sys

# CRITICAL: relbench prints "Loading Database... Done in X seconds." to
# STDOUT every time a dataset is loaded. We capture this heredoc's
# stdout into a bash array, so any stray relbench print contaminates
# the task list (the famous "File 'Done/db.zip' is not in the registry"
# error). Redirect ALL print output during loading to stderr; emit only
# the final clean list to stdout at the end.
with contextlib.redirect_stdout(sys.stderr):
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

# Only the final list goes to stdout (captured by the bash $(...)).
sys.stdout.write("\n".join(out) + ("\n" if out else ""))
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

# Compute total MAX_STEPS = STEPS_PER_TASK x N_tasks unless user pinned it.
N_TASKS=${#TASKS[@]}
if [ -z "${MAX_STEPS:-}" ]; then
    MAX_STEPS=$(( STEPS_PER_TASK * N_TASKS ))
fi
echo
echo "  per-task budget: ${STEPS_PER_TASK} steps/epoch, ${EPOCHS} epochs"
echo "  total MAX_STEPS = ${STEPS_PER_TASK} x ${N_TASKS} = ${MAX_STEPS} (override via MAX_STEPS env)"
echo "  total optimizer steps over the run: $((MAX_STEPS * EPOCHS))"
echo

# ------------------------------------------------------------------
# Helpers: bounded job pool with at most $PARALLEL_BUILDS concurrent
# children. wait -n returns when any one child exits; we then prune
# finished pids and continue spawning. Round-robin GPU assignment for
# phase 1; CPU-only (CUDA_VISIBLE_DEVICES="") for phase 2.
# ------------------------------------------------------------------
phase1_build_one() {
    local ds="$1"
    local gpu_id="$2"
    if [ -f "$TF_STORE/$ds/.done" ]; then
        echo "  [GPU $gpu_id] $ds: cached"
        return 0
    fi
    local log="$OUT_DIR/build_tf_${ds}.log"
    local t0
    t0=$(date +%s)
    echo "  [GPU $gpu_id] $ds: building ... (log: $log)"
    if CUDA_VISIBLE_DEVICES="$gpu_id" python3 tools/build_tf_store.py \
        --dataset "$ds" --out_dir "$TF_STORE/$ds" > "$log" 2>&1; then
        touch "$TF_STORE/$ds/.done"
        echo "  [GPU $gpu_id] $ds: done in $(( $(date +%s) - t0 ))s"
    else
        echo "  [GPU $gpu_id] $ds: FAILED -- see $log" >&2
        return 1
    fi
}

phase2_build_one() {
    local spec="$1"
    local full="${spec%%:*}"
    local ds="${full%%.*}"
    local task="${full#*.}"
    local out="$SHARDS/$ds/$task"
    if [ -f "$out/.done" ]; then
        echo "  $ds.$task: cached"
        return 0
    fi
    local log="$OUT_DIR/build_shard_${ds}_${task}.log"
    local t0
    t0=$(date +%s)
    echo "  $ds.$task: building ... (log: $log)"
    # CPU-bound -- explicitly hide GPUs so a stray torch.cuda call in
    # the offline script doesn't reserve VRAM uselessly.
    if CUDA_VISIBLE_DEVICES="" python3 tools/precompute_shards.py \
        --dataset "$ds" --task "$task" \
        --K "$K" --shard_size "$SHARD_SIZE" \
        --out_dir "$out" \
        --splits train val test > "$log" 2>&1; then
        touch "$out/.done"
        echo "  $ds.$task: done in $(( $(date +%s) - t0 ))s"
    else
        echo "  $ds.$task: FAILED -- see $log" >&2
        return 1
    fi
}

# Tracks running pids in the bounded pool. Drops finished ones.
prune_pids() {
    local -n arr=$1
    local kept=()
    local p
    for p in "${arr[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
            kept+=("$p")
        fi
    done
    arr=("${kept[@]}")
}

# ------------------------------------------------------------------
# Phase 1: TF memmap stores (parallel, GPU-pinned per dataset)
# ------------------------------------------------------------------
# Clamp phase-1 concurrency to the physical GPU count so we never set
# CUDA_VISIBLE_DEVICES to a non-existent device.
P1_CONCURRENCY=$(( PARALLEL_TF_BUILDS < NPROC ? PARALLEL_TF_BUILDS : NPROC ))
echo "[1/3] Building TF memmap stores ($P1_CONCURRENCY concurrent, one GPU each)"
declare -a P1_PIDS=()
P1_IDX=0
for ds in "${DATASETS_SELECTED[@]}"; do
    while [ ${#P1_PIDS[@]} -ge "$P1_CONCURRENCY" ]; do
        wait -n 2>/dev/null || true
        prune_pids P1_PIDS
    done
    GPU_ID=$(( P1_IDX % P1_CONCURRENCY ))
    phase1_build_one "$ds" "$GPU_ID" &
    P1_PIDS+=("$!")
    P1_IDX=$(( P1_IDX + 1 ))
done
wait

# Verify phase 1 outputs exist before moving on -- a silent build
# failure would otherwise propagate into phase 2 and torchrun.
P1_MISSING=()
for ds in "${DATASETS_SELECTED[@]}"; do
    if [ ! -f "$TF_STORE/$ds/.done" ]; then
        P1_MISSING+=("$ds")
    fi
done
if [ ${#P1_MISSING[@]} -gt 0 ]; then
    echo "ERROR: phase 1 failed to produce TF stores for: ${P1_MISSING[*]}" >&2
    echo "  see $OUT_DIR/build_tf_*.log for details" >&2
    exit 2
fi
echo

# ------------------------------------------------------------------
# Phase 2: precomputed sample shards (parallel, CPU-only)
# ------------------------------------------------------------------
echo "[2/3] Building precomputed sample shards ($PARALLEL_SHARD_BUILDS "\
"concurrent, K=$K, shard=$SHARD_SIZE)"
declare -a P2_PIDS=()
for spec in "${TASKS[@]}"; do
    while [ ${#P2_PIDS[@]} -ge "$PARALLEL_SHARD_BUILDS" ]; do
        wait -n 2>/dev/null || true
        prune_pids P2_PIDS
    done
    phase2_build_one "$spec" &
    P2_PIDS+=("$!")
done
wait

# Verify phase 2 outputs exist before launching the training run.
P2_MISSING=()
for spec in "${TASKS[@]}"; do
    full="${spec%%:*}"
    ds="${full%%.*}"
    task="${full#*.}"
    if [ ! -f "$SHARDS/$ds/$task/.done" ]; then
        P2_MISSING+=("$ds.$task")
    fi
done
if [ ${#P2_MISSING[@]} -gt 0 ]; then
    echo "ERROR: phase 2 failed to produce shards for: ${P2_MISSING[*]}" >&2
    echo "  see $OUT_DIR/build_shard_*.log for details" >&2
    exit 3
fi
echo

# ------------------------------------------------------------------
# Phase 3: DDP training
# ------------------------------------------------------------------
echo "[3/3] Launching DDP training across $NPROC GPUs"
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
echo "  tasks=$TASKS_CSV"
echo

# WANDB defaults to ONLINE -- this user has a WANDB_API_KEY configured
# in the shell. Override with WANDB_MODE=offline at launch if running
# on a pod without outbound internet (or set WANDB_API_KEY="" first).
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace + sentence_transformers can still ping the Hub for model
# metadata even when the model itself is cached locally. To avoid hangs
# on offline pods we auto-set the offline flags only WHEN the GloVe
# model is already cached -- otherwise the first run wouldn't be able
# to download. User can force either behavior:
#   HF_HUB_OFFLINE=0  -> allow Hub access even if cached
#   HF_HUB_OFFLINE=1  -> require cache (will error if missing)
_GLOVE_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--sentence-transformers--average_word_embeddings_glove.6B.300d"
if [ -d "$_GLOVE_CACHE" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    echo "[hf] GloVe cache present -> HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
else
    echo "[hf] GloVe cache NOT found at $_GLOVE_CACHE -- first run will download"
fi

LOG="$OUT_DIR/train.log"
# OOM mitigation: how many DDP ranks may pickle-load + materialize a
# dataset simultaneously. Default 1 = strict serialization (slowest
# startup, lowest peak RAM). Bump to 2 or 4 if pod has headroom.
LOAD_CONCURRENCY="${LOAD_CONCURRENCY:-1}"

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
    --ff_dropout "$FF_DROPOUT" \
    --attn_dropout "$ATTN_DROPOUT" \
    --epochs "$EPOCHS" \
    --max_steps_per_epoch "$MAX_STEPS" \
    --num_workers "$WORKERS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --loss_balance "$LOSS_BALANCE" \
    --load_concurrency "$LOAD_CONCURRENCY" \
    --out_dir "$OUT_DIR" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "$LOG"

echo
echo "================================================================"
echo "Training complete. Test metrics:"
echo "================================================================"
ls -la "$OUT_DIR/multi_task/" 2>/dev/null | tail -5
cat "$OUT_DIR/multi_task/"*.json 2>/dev/null | head -200
