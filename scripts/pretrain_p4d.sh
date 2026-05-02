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
#   WORKERS                  DataLoader workers per rank (default 2).
#                            Each worker is a forked Python process;
#                            CPython ref-counting breaks COW so each
#                            fork's anon RSS grows toward parent-rank
#                            size. With 8 ranks * 2 workers = 16 forks
#                            (vs 32 at the old default of 4) we cut
#                            anon overhead in half. Bump only if I/O
#                            is the bottleneck and you have RAM.
#   LR                       base lr (default 1e-4; multiplied by world_size)
#   WARMUP                   warmup steps (default 1000)
#   LOSS_BALANCE             none|per_task_mean|fixed:..|uncertainty
#                            (default none = RT formula)
#   NPROC                    GPUs per node (default 8)
#   OUT_DIR                  results dir (default results/p4d_pretrain)
#   RUN_NAME                 wandb run name (default p4d_alltasks)
#   SHARD_SIZE               samples per memmap shard (default 50000)
#   LOAD_CONCURRENCY         DDP ranks that may load a dataset at the
#                            same time (default 1 = strict serial; bump
#                            for faster startup if you have RAM headroom)
#   MEM_WATCHDOG_PCT         Memory threshold (% of cgroup limit) at
#                            which the watchdog SIGTERMs torchrun
#                            BEFORE the kubelet OOMKills the pod
#                            (default 92). Lower it if you keep getting
#                            killed -- the kubelet kills with SIGKILL
#                            and truncates train.log mid-write, but the
#                            watchdog gives python 30s to flush.
#   MEM_WATCHDOG_INTERVAL    Watchdog poll interval in seconds (default 3)
#   SKIP_BUILD               Skip phases 1 and 2 entirely; jump straight
#                            to phase 3 training. Use when TF stores
#                            and shards already exist on disk and you
#                            only want to re-run training (e.g., after
#                            a code change in the model / encoders).
#                            The launcher still verifies the per-
#                            dataset / per-task .done sentinels exist
#                            and aborts early if they don't, so a
#                            partial build won't silently break phase 3.
#                            (Equivalent: SKIP_PHASE_1=1 SKIP_PHASE_2=1)
#   SKIP_PHASE_1             Skip phase 1 (TF memmap build) only.
#   SKIP_PHASE_2             Skip phase 2 (sample shard build) only.
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

# FULL_GRAPH=1 builds with upto_test_timestamp=False (the materialization
# includes entities created after train_cutoff). Required for autocomplete
# tasks (results-position, qualifying-position, transactions-price,
# users-birthyear) whose val/test seeds reference such rows -- without it,
# their indptr lookup IndexErrors at graph_cache.py:288. Temporal leakage
# is still prevented by the per-neighbor seed_time filter at
# gfm_data/sampler.py:69. See docs/truncated_graph_caveat.md.
#
# We use mode-aware TF/shard cache paths so the truncated build is not
# overwritten by a full-graph rebuild (and vice-versa).
FULL_GRAPH="${FULL_GRAPH:-0}"
if [ "$FULL_GRAPH" = "1" ]; then
    TF_STORE="$CACHE/tf_store_full"
    SHARDS="$CACHE/shards_full"
    FULL_GRAPH_FLAG="--full_graph"
else
    TF_STORE="$CACHE/tf_store"
    SHARDS="$CACHE/shards"
    FULL_GRAPH_FLAG=""
fi
# Optional shard-tree namespace. Lets the backbone-variance sweep
# (scripts/holdout_dataset_eval_pretrain_sweep.sh) put each
# pretrain seed's shards in its own subdir so every trial rebuilds
# its own neighbor list rather than reusing the first-build cache.
# tf_store and the materialization cache stay shared (they're
# deterministic from raw data; no benefit from per-seed isolation).
SHARDS_SUBDIR="${SHARDS_SUBDIR:-}"
if [ -n "$SHARDS_SUBDIR" ]; then
    SHARDS="$SHARDS/$SHARDS_SUBDIR"
fi
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
WORKERS="${WORKERS:-2}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-1000}"
# Pretrain RNG seed -- forwarded to main_node_ddp.py --seed. Default
# 42 matches main_node_ddp.py's argparse default. Sweep over this
# (e.g. via scripts/holdout_dataset_eval_pretrain_sweep.sh) to bound
# backbone variance.
SEED="${SEED:-42}"
LOSS_BALANCE="${LOSS_BALANCE:-none}"
NPROC="${NPROC:-8}"
OUT_DIR="${OUT_DIR:-results/p4d_pretrain}"
RUN_NAME="${RUN_NAME:-p4d_alltasks}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
# Phase 1 (TF memmap) and Phase 2 (shards) have different bottlenecks:
#   * Phase 1 is GPU-bound (text embedding). Default concurrency
#     PARALLEL_TF_BUILDS = NPROC, one A100 per dataset.
#   * Phase 2 is CPU-bound (per-row neighbor sampling). Two-level
#     parallelism: PARALLEL_SHARD_BUILDS picks how many tasks to
#     build concurrently (each process owns one cache + steady-state
#     ~3-5 GB on rel-event after the TF drop), and SHARD_WORKERS
#     picks how many fork-workers each builder spins up to share
#     that cache via copy-on-write. Default 8 builders x 1 worker
#     matches the prior behavior; SHARD_WORKERS=10 saturates
#     p4d.24xlarge's 96 vCPUs (~12-16x speedup, see
#     speedup-precompute-shards.md).
#
# Both can be overridden independently. PARALLEL_BUILDS is kept as a
# back-compat catch-all that overrides BOTH if neither phase-specific
# var is set.
PARALLEL_BUILDS="${PARALLEL_BUILDS:-$NPROC}"
PARALLEL_TF_BUILDS="${PARALLEL_TF_BUILDS:-$PARALLEL_BUILDS}"
# Inter-task parallelism: number of shard-builder *processes*. Default
# raised to min(NPROC, 8) -- p4d.24xlarge has 1.1 TB RAM so even 8
# concurrent rel-event builds (~25 GB peak transient each) leave
# headroom. Override with PARALLEL_SHARD_BUILDS for tighter pods.
_default_shard_builds=$(( NPROC < 8 ? NPROC : 8 ))
PARALLEL_SHARD_BUILDS="${PARALLEL_SHARD_BUILDS:-$_default_shard_builds}"
# Intra-task parallelism: workers per builder for sample sampling.
# Default 10 -> 8 builders x 10 workers = 80 fork procs (~83% of
# p4d's 96 vCPUs), max throughput while leaving 16 cores for OS /
# page cache / DDP rendezvous. Drop to 1 (sequential, no intra-task
# fork) on memory-tight pods.
SHARD_WORKERS="${SHARD_WORKERS:-10}"

mkdir -p "$OUT_DIR" "$TF_STORE" "$SHARDS"

# Validate concurrency knobs early (catch typos like PARALLEL_BUILDS=0).
for _name in PARALLEL_BUILDS PARALLEL_TF_BUILDS PARALLEL_SHARD_BUILDS SHARD_WORKERS NPROC; do
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
echo "  parallel_tf=$PARALLEL_TF_BUILDS  parallel_shard=$PARALLEL_SHARD_BUILDS  shard_workers=$SHARD_WORKERS"
echo "  lr=$LR  warmup=$WARMUP  loss_balance=$LOSS_BALANCE"
echo "  cache=$CACHE  out=$OUT_DIR"
echo "  full_graph=$FULL_GRAPH  tf_store=$TF_STORE  shards=$SHARDS"
echo

# ------------------------------------------------------------------
# Phase 0: enumerate tasks via relbench (or take TASKS_CSV override)
# ------------------------------------------------------------------
# TASKS_CSV override: pre-built "ds.task:weight,..." list. When set,
# skip the relbench enumeration entirely. Used by
# scripts/holdout_task_dev.sh to drive a pretrain on the union of
# all-but-one tasks across multiple datasets without re-implementing
# the build phases / DDP launch.
# EXCLUDED_TASKS: comma-separated "ds.task" keys to drop from the
# auto-enumerate path. Default is the 5 RelBench v2 entity bcls/reg
# tasks whose supervised single-task GNN baseline (per the v2 paper)
# sits at-or-below random -- including them in pretrain just adds
# noise. See docs/holdout_results.md and the rationale in
# scripts/holdout_dataset_eval_clean.sh. Set EXCLUDED_TASKS="" to
# include every task; set TASKS_CSV explicitly to bypass this filter.
EXCLUDED_TASKS="${EXCLUDED_TASKS:-rel-event.event_interest-interested,rel-event.event_interest-not_interested,rel-event.users-birthyear,rel-trial.site-success,rel-amazon.item-ltv}"

if [ -n "${TASKS_CSV:-}" ]; then
    echo "[0/3] Using user-provided TASKS_CSV (skipping relbench enumeration)"
    TASKS_RAW=$(echo "$TASKS_CSV" | tr ',' '\n')
else
    echo "[0/3] Enumerating RelBench tasks (binary + regression only; excluding: ${EXCLUDED_TASKS:-<none>})"

    # Pass DATASETS_FILTER + EXCLUDED_TASKS into Python; emit a list
    # of "ds.task:1.0" strings.
    TASKS_RAW=$(DATASETS_FILTER="$DATASETS_FILTER" \
                EXCLUDED_TASKS="$EXCLUDED_TASKS" python3 - <<'PY'
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
    excluded = {
        x.strip()
        for x in (os.environ.get("EXCLUDED_TASKS") or "").split(",")
        if x.strip()
    }

    out = []
    for ds in datasets:
        try:
            names = get_task_names(ds)
        except Exception as e:
            print(f"# WARN: cannot list tasks for {ds}: {e}", file=sys.stderr)
            continue
        for tn in names:
            key = f"{ds}.{tn}"
            if key in excluded:
                print(f"# excluding low-quality task: {key}", file=sys.stderr)
                continue
            try:
                t = get_task(ds, tn)
                if t.task_type in ALLOWED:
                    out.append(f"{key}:1.0")
            except Exception as e:
                print(f"# WARN: cannot load {ds}.{tn}: {e}", file=sys.stderr)

# Only the final list goes to stdout (captured by the bash $(...)).
sys.stdout.write("\n".join(out) + ("\n" if out else ""))
PY
)
fi  # end TASKS_CSV override branch

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
        --dataset "$ds" --out_dir "$TF_STORE/$ds" $FULL_GRAPH_FLAG \
        > "$log" 2>&1; then
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
    # K-aware sentinel: tools/precompute_shards.py writes shards to
    # <out>/<K>/<split>/, so the per-K .done belongs at <out>/<K>/.
    # The earlier <out>/.done was K-blind -- a K=16 build would
    # falsely satisfy a K=300 launch's check, then training would
    # crash with "shards not found at .../300/train".
    local k_dir="$out/$K"
    if [ -f "$k_dir/.done" ]; then
        echo "  $ds.$task: cached (K=$K)"
        return 0
    fi
    local log="$OUT_DIR/build_shard_${ds}_${task}.log"
    local t0
    t0=$(date +%s)
    echo "  $ds.$task: building K=$K ... (log: $log)"
    # CPU-bound -- explicitly hide GPUs so a stray torch.cuda call in
    # the offline script doesn't reserve VRAM uselessly.
    # IMPORTANT: --name_prefix MUST match the prefix the multi-task
    # trainer uses at runtime (train_multi_task.py:148 passes
    # ``name_prefix=ds_name``). Without this, shard type ids are
    # un-prefixed (``users``, ``events``) while the runtime expects
    # prefixed (``rel-event::users``) and lookups KeyError.
    if CUDA_VISIBLE_DEVICES="" python3 tools/precompute_shards.py \
        --dataset "$ds" --task "$task" \
        --K "$K" --shard_size "$SHARD_SIZE" \
        --out_dir "$out" \
        --name_prefix "$ds" \
        --workers "$SHARD_WORKERS" \
        --splits train val test $FULL_GRAPH_FLAG \
        > "$log" 2>&1; then
        mkdir -p "$k_dir"
        touch "$k_dir/.done"
        echo "  $ds.$task: done in $(( $(date +%s) - t0 ))s (K=$K)"
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
# SKIP_BUILD or SKIP_PHASE_1 jumps straight past the build loop and
# only verifies the per-dataset .done sentinels. Phase 3 needs the
# memmap files to exist; aborting early here is much friendlier than
# letting torchrun crash with a cryptic FileNotFoundError later.
SKIP_PHASE_1="${SKIP_PHASE_1:-${SKIP_BUILD:-0}}"
SKIP_PHASE_2="${SKIP_PHASE_2:-${SKIP_BUILD:-0}}"

if [ "$SKIP_PHASE_1" = "1" ]; then
    echo "[1/3] SKIP_PHASE_1=1 -- skipping TF memmap build (verifying .done sentinels)"
else
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
fi

# Verify phase 1 outputs exist before moving on -- a silent build
# failure (or a SKIP_PHASE_1 with missing artifacts) would otherwise
# propagate into phase 2 and torchrun.
P1_MISSING=()
for ds in "${DATASETS_SELECTED[@]}"; do
    if [ ! -f "$TF_STORE/$ds/.done" ]; then
        P1_MISSING+=("$ds")
    fi
done
if [ ${#P1_MISSING[@]} -gt 0 ]; then
    if [ "$SKIP_PHASE_1" = "1" ]; then
        echo "ERROR: SKIP_PHASE_1=1 but TF stores are missing for: ${P1_MISSING[*]}" >&2
        echo "  rerun without SKIP_PHASE_1 / SKIP_BUILD to build them, or" >&2
        echo "  check that TF_STORE=$TF_STORE points at the right location" >&2
    else
        echo "ERROR: phase 1 failed to produce TF stores for: ${P1_MISSING[*]}" >&2
        echo "  see $OUT_DIR/build_tf_*.log for details" >&2
    fi
    exit 2
fi
echo

# ------------------------------------------------------------------
# Phase 2: precomputed sample shards (parallel, CPU-only)
# ------------------------------------------------------------------
if [ "$SKIP_PHASE_2" = "1" ]; then
    echo "[2/3] SKIP_PHASE_2=1 -- skipping shard build (verifying .done sentinels)"
else
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
fi

# Verify phase 2 outputs exist before launching the training run.
# Sentinel is K-aware (under <ds>/<task>/<K>/.done) so a stale
# .done from a different K won't pass.
P2_MISSING=()
for spec in "${TASKS[@]}"; do
    full="${spec%%:*}"
    ds="${full%%.*}"
    task="${full#*.}"
    if [ ! -f "$SHARDS/$ds/$task/$K/.done" ]; then
        P2_MISSING+=("$ds.$task")
    fi
done
if [ ${#P2_MISSING[@]} -gt 0 ]; then
    if [ "$SKIP_PHASE_2" = "1" ]; then
        echo "ERROR: SKIP_PHASE_2=1 but shards are missing for: ${P2_MISSING[*]}" >&2
        echo "  rerun without SKIP_PHASE_2 / SKIP_BUILD to build them, or" >&2
        echo "  check that SHARDS=$SHARDS points at the right location" >&2
    else
        echo "ERROR: phase 2 failed to produce shards for: ${P2_MISSING[*]}" >&2
        echo "  see $OUT_DIR/build_shard_*.log for details" >&2
    fi
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

# Pre-flight cleanup: kill any stale torchrun / worker processes from a
# prior aborted run (e.g., the watchdog SIGTERM'd a previous attempt
# but the C10d rendezvous server's listening socket lingered, leaving
# the next torchrun crashing with EADDRINUSE on the rdvz port). We
# don't want to wipe unrelated python jobs the user might have running,
# so we match precisely on this script's main_node_ddp.py and on
# torchrun launchers that target it. ``|| true`` guards against
# pkill rc=1 ("no processes matched") under set -e.
echo "[3/3] pre-flight: reaping any stale torchrun / main_node_ddp.py procs"
# Match python interpreters running main_node_ddp.py (NOT, e.g., a
# user's "vim main_node_ddp.py" in another terminal whose argv just
# contains the filename). Both patterns require python in argv[0].
pkill -TERM -f "python[0-9.]* .*main_node_ddp\.py" 2>/dev/null || true
pkill -TERM -f "python[0-9.]* .*torchrun.*main_node_ddp" 2>/dev/null || true
# Brief wait for sockets to clear (TIME_WAIT on the rdzv port can hold
# bind() for ~60s on default sysctls; SIGKILL after 5s if anything is
# still up).
sleep 5
pkill -KILL -f "python[0-9.]* .*main_node_ddp\.py" 2>/dev/null || true
pkill -KILL -f "python[0-9.]* .*torchrun.*main_node_ddp" 2>/dev/null || true

# Default rdzv port (overridable). Bumping this on a stuck-port retry
# is faster than waiting for TIME_WAIT to clear:
#   MASTER_PORT=29501 ./scripts/pretrain_p4d.sh
export MASTER_PORT="${MASTER_PORT:-29500}"
echo "[3/3] rdzv MASTER_PORT=$MASTER_PORT"


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

# Memory watchdog: in a Kubernetes pod, when total memory hits the
# cgroup limit the kubelet reaps the entire container with
# OOMKilled. The kernel oom-killer's SIGKILL gives the python
# processes zero time to flush stdout, so the train.log on the PVC
# is truncated mid-line and post-mortem is impossible. The watchdog
# samples ANONYMOUS memory only (excludes page cache, which is
# reclaimable) from cgroup memory.stat and SIGTERMs torchrun's process
# group when anon usage exceeds MEM_WATCHDOG_PCT of the limit -- giving
# python a chance to flush the [rss r<rank>] lines and write a clean
# error. We monitor anon (not memory.current) because our workload is
# memmap-heavy: TF stores + sample shards add tens of GiB of file-
# backed page cache that the kernel will evict before OOMing, so
# memory.current trips spuriously while real anonymous heap is fine.
MEM_WATCHDOG_PCT="${MEM_WATCHDOG_PCT:-92}"
MEM_WATCHDOG_INTERVAL="${MEM_WATCHDOG_INTERVAL:-3}"

# Locate cgroup memory.stat + memory.max (v2) / memory.limit (v1).
# We grep one line out of memory.stat per poll: cheap.
_MEM_STAT=""
_MEM_STAT_KEY=""   # "anon" on v2, "rss" on v1
_MEM_LIMIT=""
if [ -r /sys/fs/cgroup/memory.stat ] && [ -r /sys/fs/cgroup/memory.max ]; then
    # cgroup v2
    _MEM_STAT="/sys/fs/cgroup/memory.stat"
    _MEM_STAT_KEY="anon"
    _MEM_LIMIT_RAW=$(cat /sys/fs/cgroup/memory.max)
    if [ "$_MEM_LIMIT_RAW" = "max" ]; then
        _MEM_LIMIT=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
    else
        _MEM_LIMIT="$_MEM_LIMIT_RAW"
    fi
elif [ -r /sys/fs/cgroup/memory/memory.stat ] \
     && [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    # cgroup v1 -- "rss" in memory.stat is anon RSS (despite the name),
    # NOT including kernel page cache (which is "cache").
    _MEM_STAT="/sys/fs/cgroup/memory/memory.stat"
    _MEM_STAT_KEY="rss"
    _MEM_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    if [ "$_MEM_LIMIT" -gt $((1 << 62)) ]; then
        _MEM_LIMIT=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
    fi
else
    # No cgroup; we'll compute anon from /proc/meminfo at poll time
    # as Active(anon) + Inactive(anon).
    _MEM_LIMIT=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
fi

if [ -n "$_MEM_STAT" ]; then
    echo "[watchdog] anon source: $_MEM_STAT (key=$_MEM_STAT_KEY)"
else
    echo "[watchdog] anon source: /proc/meminfo Active(anon)+Inactive(anon)"
fi
echo "[watchdog] limit=$(numfmt --to=iec --suffix=B $_MEM_LIMIT 2>/dev/null || echo "$_MEM_LIMIT")"
echo "[watchdog] threshold=${MEM_WATCHDOG_PCT}% (anon only; page cache excluded)  poll=${MEM_WATCHDOG_INTERVAL}s"

# Compute byte threshold once.
_MEM_THRESHOLD=$(( _MEM_LIMIT * MEM_WATCHDOG_PCT / 100 ))

# Launch torchrun in its own process group so we can signal it
# cleanly (kill the whole tree, not just the bash subshell).
set -m
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
    --seed "$SEED" \
    --out_dir "$OUT_DIR" \
    --run_name "$RUN_NAME" $FULL_GRAPH_FLAG \
    > "$LOG" 2>&1 &
TORCHRUN_PID=$!
set +m

# Tail the log to the user's terminal in the background so they see
# progress live (same UX as the previous `| tee`).
tail -F "$LOG" &
TAIL_PID=$!

# Watchdog loop: sample memory; if over threshold, SIGTERM the
# torchrun process group, then escalate to SIGKILL after a grace.
(
    while kill -0 "$TORCHRUN_PID" 2>/dev/null; do
        if [ -n "$_MEM_STAT" ] && [ -r "$_MEM_STAT" ]; then
            # Read anon (cgroup v2) or rss (cgroup v1) -- both are
            # bytes of anonymous memory, EXCLUDING page cache.
            cur=$(awk -v key="$_MEM_STAT_KEY" '$1 == key { print $2; exit }' \
                  "$_MEM_STAT" 2>/dev/null || echo 0)
        else
            # No cgroup: anon ~= Active(anon) + Inactive(anon).
            cur=$(awk '
                /^Active\(anon\):/   { a = $2 }
                /^Inactive\(anon\):/ { i = $2 }
                END                  { print (a + i) * 1024 }
            ' /proc/meminfo)
        fi
        # Defensive: awk can produce empty output if the expected key
        # is missing (older kernels, exotic cgroup configs); the
        # subsequent ``[ "" -ge N ]`` test would syntax-error and
        # ``set -e`` (inherited by this subshell) would silently kill
        # the watchdog. Coerce empty/whitespace to 0.
        cur="${cur:-0}"
        case "$cur" in (''|*[!0-9]*) cur=0 ;; esac
        if [ "$cur" -ge "$_MEM_THRESHOLD" ]; then
            cur_h=$(numfmt --to=iec --suffix=B "$cur" 2>/dev/null || echo "$cur")
            lim_h=$(numfmt --to=iec --suffix=B "$_MEM_LIMIT" 2>/dev/null || echo "$_MEM_LIMIT")
            {
                echo
                echo "================================================================"
                echo "[watchdog] ANON MEMORY THRESHOLD EXCEEDED at $(date -Is)"
                echo "[watchdog]   anon=${cur_h}  limit=${lim_h}  threshold=${MEM_WATCHDOG_PCT}%"
                echo "[watchdog]   (page cache excluded -- this is real heap pressure)"
                echo "[watchdog] sending SIGTERM to torchrun pgid $TORCHRUN_PID"
                echo "[watchdog] (avoids kubelet OOMKilled which would truncate this log)"
                echo "================================================================"
            } >> "$LOG"
            # SIGTERM torchrun's process group + child tree. We try
            # three reach-mechanisms in order so that older bash
            # versions where ``set -m`` doesn't put the bg job in a
            # new pgid still get every DataLoader worker reaped:
            #   1. ``kill -TERM -PID`` (negative = pgid signal).
            #   2. ``pkill -TERM -P PID`` (every direct child by ppid).
            #   3. ``kill -TERM PID`` (the torchrun launcher itself).
            kill -TERM -"$TORCHRUN_PID" 2>/dev/null || true
            pkill -TERM -P "$TORCHRUN_PID" 2>/dev/null || true
            kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
            # Give python 30s to flush + cleanup, then escalate.
            for _ in $(seq 1 30); do
                kill -0 "$TORCHRUN_PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$TORCHRUN_PID" 2>/dev/null; then
                echo "[watchdog] grace expired; SIGKILL" >> "$LOG"
                kill -KILL -"$TORCHRUN_PID" 2>/dev/null || true
                pkill -KILL -P "$TORCHRUN_PID" 2>/dev/null || true
                kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
            fi
            exit 0
        fi
        sleep "$MEM_WATCHDOG_INTERVAL"
    done
) &
WATCHDOG_PID=$!

# Make sure the watchdog and tail die when this script exits for ANY
# reason (success, ctrl-c, watchdog-triggered termination).
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    kill "$TAIL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for torchrun; capture its exit code so the script propagates it.
# IMPORTANT: ``wait`` returns torchrun's exit code; with ``set -e`` a
# non-zero return would kill the script before we read $?. The
# watchdog explicitly SIGTERMs torchrun (rc != 0) on memory pressure
# -- if we don't guard the wait, the diagnostic banner below is never
# printed and the script exits with no message.
TORCHRUN_RC=0
wait "$TORCHRUN_PID" || TORCHRUN_RC=$?

# Stop the tail + watchdog now that torchrun is done.
cleanup
trap - EXIT INT TERM

if [ "$TORCHRUN_RC" -ne 0 ]; then
    echo
    echo "[watchdog] torchrun exited rc=$TORCHRUN_RC (see $LOG for details)"
fi

echo
echo "================================================================"
echo "Training complete. Test metrics:"
echo "================================================================"
ls -la "$OUT_DIR/multi_task/" 2>/dev/null | tail -5
cat "$OUT_DIR/multi_task/"*.json 2>/dev/null | head -200
