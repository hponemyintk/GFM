#!/usr/bin/env bash
# Phase-4 holdout-task development loop -- multi-dataset edition.
#
# Pretrain a SINGLE backbone on the union of (all-but-one) tasks
# from each of multiple datasets simultaneously. Then for each
# dataset's held-out task: freeze the backbone, extract embeddings,
# fine-tune a shallow head + run TabPFN. The frozen-backbone
# constraint is the central GFM claim -- pretrained representations
# should transfer to unseen task heads on already-seen schemas
# without further backbone training.
#
# Defaults (FULL_GRAPH=1; see docs/truncated_graph_caveat.md):
#   DATASETS    "rel-f1 rel-event"
#   HOLDOUTS    "rel-f1:driver-top3 rel-event:user-attendance"
#   PRETRAIN    union of all-but-holdout tasks across DATASETS
#                = 4 rel-f1 + 2 rel-event = 6 tasks
#   FULL_GRAPH  1   -- builds with upto_test_timestamp=False so
#                       autocomplete-task seeds (results-position,
#                       qualifying-position) don't IndexError on
#                       val/test.
#
# Holdouts are mixed-metric (one binary + one regression) so the
# frozen-backbone GFM claim is tested across both head kinds:
#   * rel-f1 / driver-top3       -- binary, AUROC. Paper-benchmarked
#                                   (expts/run-large-base-experiments,
#                                   expts/run-encoder-ablation).
#   * rel-event / user-attendance -- regression, MAE. Paper Table 1a
#                                   (MAE 0.2502); paper expts cover it.
#
# Pretrain composition with the default holdouts:
#   * rel-f1: driver-position (regression), driver-dnf (binary),
#             results-position (regression, autocomplete),
#             qualifying-position (regression, autocomplete)
#   * rel-event: user-repeat (binary), user-ignore (binary)
# = 3 regression + 3 binary tasks for pretrain. Mixed-metric pretrain
# matters: it forces the backbone to embed both classification and
# regression signal, so the frozen-backbone evaluation on the
# held-out regression head (rel-event:user-attendance) tests
# transfer of regression structure -- not just probing a backbone
# that only ever saw classification labels.
#
# Low-quality tasks dropped by default (RelBench v2 paper baselines
# at-or-below random; same exclusion set as the Phase-5 launcher
# scripts/holdout_dataset_eval.sh and scripts/pretrain_p4d.sh's
# EXCLUDED_TASKS default):
#   * rel-event.event_interest-interested      Table 3 GNN AUC 0.4764 -- below random
#   * rel-event.event_interest-not_interested  Table 3 GNN AUC 0.6040 -- ~random
#   * rel-event.users-birthyear                Table 5 GNN R^2 -0.030 -- negative
# Add them back to DEFAULT_ALL_TASKS[rel-event] for the all-tasks
# ablation. (rel-trial.site-success and rel-amazon.item-ltv are also
# in the standard low-quality set; they're not in this launcher's
# DATASETS today, so no change is needed here.)
#
# rel-hm is NOT in defaults but stays available via DATASETS override.
# Empirical wall-time on p4d.24xlarge for the full 9-task pretrain at
# EPOCHS=20 STEPS_PER_TASK=500: ~8h11m (commit e6aa7a3, see
# docs/truncated_graph_caveat.md for the test-metric snapshot).
#
# Memory profile:
#   * rel-f1 materialization:    ~50 MB (laptop-fine)
#   * rel-hm materialization:    ~600 MB (laptop-fine)
#   * rel-event materialization: ~25 GB peak RSS (laptop OOMs at 27 GB)
#
# Full 2-dataset default fits the laptop. The p4d backend (NPROC>=2)
# is still useful for paper-config (K=300 / 4-layer / channels=512).
#
# Usage:
#   bash scripts/holdout_task_dev.sh [EPOCHS] [STEPS_PER_TASK]
#
#     EPOCHS         default 5
#     STEPS_PER_TASK if set, forwarded to pretrain_p4d.sh (which
#                    computes MAX_STEPS = STEPS_PER_TASK x N_tasks
#                    so each task gets the same per-epoch density
#                    it would in a single-task run). If unset, the
#                    p4d backend uses its own default (500 in
#                    pretrain_p4d.sh) and the laptop backend uses
#                    50 (sized for laptop streaming smoke-tests).
#
# ---------------- Example invocations (copy-paste) ----------------
#
# AWS p4d.24xlarge (8 x A100, paper-config, default rel-f1+rel-event,
# FULL_GRAPH=1 so the rel-f1 autocomplete tasks build cleanly):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_task_dev.sh
#
# AWS p4d quick shake-out (~1 hour wall, sanity check end-to-end):
#
#   NPROC=8 EPOCHS=3 STEPS_PER_TASK=200 \
#     bash scripts/holdout_task_dev.sh
#
# AWS p4d, three-dataset (rel-f1 + rel-event + rel-hm):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 \
#     DATASETS="rel-f1 rel-event rel-hm" \
#     HOLDOUTS="rel-f1:driver-top3 rel-event:user-attendance rel-hm:user-churn" \
#     bash scripts/holdout_task_dev.sh
#
# AWS p4d, paper-strict (drop FULL_GRAPH so the build matches the
# RelGT paper guardrail exactly; this also forces dropping the
# autocomplete tasks):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 FULL_GRAPH=0 \
#     bash scripts/holdout_task_dev.sh
#
# Laptop (single GPU; rel-event materialization ~25 GB OOMs a 27 GB
# box, so override DATASETS to drop it):
#
#   DATASETS="rel-f1 rel-hm" \
#     HOLDOUTS="rel-f1:driver-top3 rel-hm:user-churn" \
#     bash scripts/holdout_task_dev.sh
#
# Output layout under $OUT_DIR (default results/holdout_task_dev/):
#   pretrain/                    # single shared pretrain
#     multi_task/<seed>.json     # per-task test metrics during pretrain
#     multi_task/best_full.pt
#     multi_task/best_backbone.pt
#     multi_task/backbone_meta.json
#     multi_task/backbone_schema.pt
#   <dataset>.<holdout>/         # one dir per held-out task
#     embeddings/{train,val,test}.pt
#     finetune_head/finetuned.pt
#     tabpfn/tabpfn.json
#   summary.json                 # cross-task verdict (this script writes it)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-${1:-5}}"
# STEPS_PER_TASK is the user-facing knob. When unset we leave it
# empty here so the p4d branch can fall back to pretrain_p4d.sh's
# own default (500); the laptop branch substitutes a smaller value
# inline below.
STEPS_PER_TASK="${STEPS_PER_TASK:-${2:-}}"
DATASETS="${DATASETS:-rel-f1 rel-event}"
# Per-dataset holdout map: "<dataset>:<task> <dataset>:<task> ..."
# Tasks cited in RelGT paper expts (see header docstring).
# Default holdouts: one binary (rel-f1:driver-top3) + one regression
# (rel-event:user-attendance), to evaluate frozen-backbone transfer
# to held-out heads of both kinds.
HOLDOUTS="${HOLDOUTS:-rel-f1:driver-top3 rel-event:user-attendance}"

# Default per-dataset full task lists. The launcher subtracts the
# HOLDOUTS map from these to produce the pretrain CSV.
#
# rel-f1 includes the autocomplete tasks (results-position,
# qualifying-position) -- these need FULL_GRAPH=1 (default) to build
# cleanly. Their test seeds reference rows added after train_cutoff
# (100% OOB against the truncated CSR adjacency). With FULL_GRAPH=1
# the materialization keeps all entities and the indptr lookup is
# in-bounds; the per-neighbor seed_time filter at sampler.py:69
# remains the leakage barrier.
#
# rel-event keeps the 3 paper-benchmarked user-* tasks
# (user-attendance / user-repeat / user-ignore). The 3 autocomplete
# tasks (event_interest-interested, event_interest-not_interested,
# users-birthyear) are dropped by default because their RelBench v2
# paper GNN baselines are at or below random -- including them in
# the pretrain mix just adds noise. To re-enable for an all-tasks
# ablation, append them back into the rel-event entry below; the
# commented lines preserve the rationale inline.
#
# rel-hm transactions-price is autocomplete (growing transaction
# seeds); needs FULL_GRAPH=1 if a user opts it in via DATASETS=rel-hm.
#
# Sources:
#   rel-f1:    expts/run-large-base-experiments.sh + paper Table 1.
#              Autocomplete: results-position, qualifying-position
#              (the paper does not benchmark these; we include them
#              to enrich the pretrain regression coverage).
#   rel-event: expts/run-hyperparam-sweep-small-experiments.sh +
#              paper Tables 1a (user-attendance MAE 0.2502) and 1b
#              (user-repeat AUC 0.7609, user-ignore AUC 0.8157).
#   rel-hm:    expts/run-large-base-experiments.sh -- user-churn,
#              item-sales. transactions-price is a RelBench v2
#              autocomplete regression and joins under FULL_GRAPH=1.
declare -A DEFAULT_ALL_TASKS=(
  [rel-f1]="driver-position driver-dnf driver-top3 results-position qualifying-position"
  # rel-event low-quality tasks dropped by default (uncomment to
  # re-include in the rel-event entry below):
  #   "event_interest-interested"      # paper Table 3 GNN AUC 0.4764 -- below random
  #   "event_interest-not_interested"  # paper Table 3 GNN AUC 0.6040 -- ~random
  #   "users-birthyear"                # paper Table 5 GNN R^2 -0.030 -- negative
  [rel-event]="user-attendance user-repeat user-ignore"
  [rel-hm]="user-churn item-sales transactions-price"
)

# Parse HOLDOUTS into an associative array: {dataset -> task}
declare -A HOLDOUT_OF
for pair in $HOLDOUTS; do
  ds="${pair%%:*}"
  tk="${pair##*:}"
  if [ -z "$ds" ] || [ -z "$tk" ] || [ "$ds" = "$tk" ]; then
    echo "ERR: malformed HOLDOUTS entry '$pair'; expected 'dataset:task'." >&2
    exit 2
  fi
  HOLDOUT_OF["$ds"]="$tk"
done

# Build the pretrain CSV unless the user supplied one. For each
# dataset in DATASETS, take its full task list minus the holdout.
if [ -z "${PRETRAIN_TASKS_CSV:-}" ]; then
  PRETRAIN_TASKS_ARR=()
  for ds in $DATASETS; do
    full="${DEFAULT_ALL_TASKS[$ds]:-}"
    if [ -z "$full" ]; then
      echo "ERR: no default task list for dataset '$ds'; set PRETRAIN_TASKS_CSV explicitly." >&2
      exit 2
    fi
    holdout="${HOLDOUT_OF[$ds]:-}"
    if [ -z "$holdout" ]; then
      echo "ERR: no holdout specified for dataset '$ds'; add it to HOLDOUTS." >&2
      exit 2
    fi
    # Sanity: the holdout must be in the full list.
    if ! echo " $full " | grep -q " $holdout "; then
      echo "ERR: holdout '$holdout' not in known tasks for '$ds': $full" >&2
      exit 2
    fi
    for tk in $full; do
      if [ "$tk" != "$holdout" ]; then
        PRETRAIN_TASKS_ARR+=("${ds}.${tk}:1.0")
      fi
    done
  done
  PRETRAIN_TASKS_CSV=$(IFS=,; echo "${PRETRAIN_TASKS_ARR[*]}")
fi

# Final defensive check: holdout tasks must NOT appear in the
# pretrain CSV (catches the easy mistake when PRETRAIN_TASKS_CSV is
# overridden).
for ds in $DATASETS; do
  holdout="${HOLDOUT_OF[$ds]:-}"
  if [ -n "$holdout" ] && echo "$PRETRAIN_TASKS_CSV" | grep -q "${ds}\.${holdout}"; then
    echo "ERR: holdout ${ds}.${holdout} present in PRETRAIN_TASKS_CSV; remove it." >&2
    exit 2
  fi
done

CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"

# FULL_GRAPH=1 (default): build with upto_test_timestamp=False so
# autocomplete-task seeds (results-position, qualifying-position,
# transactions-price) don't IndexError on val/test. Mode-aware
# TF_STORE path keeps truncated and full builds isolated; if the user
# flips FULL_GRAPH, the script reads from a separate cache rather
# than crashing on a stale build.
FULL_GRAPH="${FULL_GRAPH:-1}"
export FULL_GRAPH  # forward to pretrain_p4d.sh
if [ "$FULL_GRAPH" = "1" ]; then
    TF_STORE="$CACHE/tf_store_full"
    FULL_GRAPH_FLAG="--full_graph"
else
    TF_STORE="$CACHE/tf_store"
    FULL_GRAPH_FLAG=""
fi

SEED="${SEED:-0}"
OUT_DIR_BASE="${OUT_DIR:-results/holdout_task_dev}"
PRETRAIN_DIR="$OUT_DIR_BASE/pretrain"

# Detect run-target and pick the right pretrain backend.
#   p4d  (default if NPROC>=2): delegate to scripts/pretrain_p4d.sh --
#        gives DDP across NPROC GPUs + precomputed shards + memory
#        watchdog + parallel TF/shard build phases.
#   laptop (default if NPROC=1): inline single-GPU streaming pretrain.
#
# Force one or the other with PRETRAIN_BACKEND=p4d / PRETRAIN_BACKEND=laptop.
NPROC="${NPROC:-1}"
if [ -z "${PRETRAIN_BACKEND:-}" ]; then
  if [ "$NPROC" -ge 2 ]; then
    PRETRAIN_BACKEND=p4d
  else
    PRETRAIN_BACKEND=laptop
  fi
fi

# Per-backend training-config defaults. These are set BEFORE the
# backend dispatch because the adoption phase (extract_embeddings,
# finetune_head, tabpfn_eval) downstream also reads K + BATCH and
# must match what the pretrain used.
if [ "$PRETRAIN_BACKEND" = "p4d" ]; then
  # Match scripts/pretrain_p4d.sh paper-config defaults.
  K="${K:-300}"
  BATCH="${BATCH:-512}"
  CHANNELS="${CHANNELS:-512}"
  NUM_LAYERS="${NUM_LAYERS:-4}"
  HEADS="${HEADS:-4}"
  CENTROIDS="${CENTROIDS:-4096}"
else
  # Laptop-sized.
  K="${K:-64}"
  BATCH="${BATCH:-128}"
  CHANNELS="${CHANNELS:-128}"
  NUM_LAYERS="${NUM_LAYERS:-1}"
  HEADS="${HEADS:-4}"
  CENTROIDS="${CENTROIDS:-512}"
fi

mkdir -p "$PRETRAIN_DIR"

# WANDB_MODE is left to the caller's environment / wandb's own
# default. WANDB_SILENT defaults to true so wandb's own progress
# spam doesn't interleave with this script's stdout, but the user
# can override with WANDB_SILENT=false.
export WANDB_SILENT="${WANDB_SILENT:-true}"

echo "=============================================================="
echo "Phase-4 multi-dataset holdout-task"
echo "  DATASETS:        $DATASETS"
echo "  HOLDOUTS:        $HOLDOUTS"
echo "  PRETRAIN tasks:  $PRETRAIN_TASKS_CSV"
echo "  EPOCHS=$EPOCHS  STEPS_PER_TASK=${STEPS_PER_TASK:-<default>}  SEED=$SEED"
echo "  PRETRAIN_BACKEND=$PRETRAIN_BACKEND  NPROC=$NPROC"
echo "  FULL_GRAPH=$FULL_GRAPH  TF_STORE=$TF_STORE"
echo "  OUT_DIR:         $OUT_DIR_BASE"
echo "=============================================================="

# ----------------- Pretrain artifact paths (shared by both backends) -----------------
PRETRAIN_LOG="$PRETRAIN_DIR/run.log"
META="$PRETRAIN_DIR/multi_task/backbone_meta.json"
WEIGHTS="$PRETRAIN_DIR/multi_task/best_backbone.pt"
SCHEMA="$PRETRAIN_DIR/multi_task/backbone_schema.pt"

# ----------------- 1+2. TF stores + pretrain -----------------
if [ -f "$META" ] && [ -f "$WEIGHTS" ] && [ -f "$SCHEMA" ]; then
  echo
  echo "=== [1+2/4] Pretrain artifacts cached at $PRETRAIN_DIR/multi_task/ ==="
elif [ "$PRETRAIN_BACKEND" = "p4d" ]; then
  # Delegate to pretrain_p4d.sh. It handles:
  #   * parallel TF memmap build (one GPU per dataset)
  #   * parallel shard build (CPU-only, --mode precomputed_shards)
  #   * DDP launch across NPROC GPUs
  #   * memory watchdog (anon-only, cgroup-aware)
  #   * pre-flight stale-process cleanup, HF offline detection
  # We pass TASKS_CSV (added in the same commit) to bypass relbench's
  # full-task enumeration and use our all-but-holdout subset directly.
  # DATASETS gets passed through as the comma-joined dataset filter
  # so phase 1+2 only build the datasets we actually need.
  echo
  echo "=== [1+2/4] Delegating to pretrain_p4d.sh (NPROC=$NPROC) ==="
  echo "  log: $PRETRAIN_LOG"
  DS_CSV=$(echo "$DATASETS" | tr ' ' ',')
  # STEPS_PER_TASK is forwarded ONLY when set so an unset launcher
  # caller falls through to pretrain_p4d.sh's own default (500).
  # Forwarding STEPS_PER_TASK="" would clobber that default with an
  # empty string and pretrain_p4d.sh would compute MAX_STEPS=0.
  _STEPS_FWD=()
  if [ -n "${STEPS_PER_TASK:-}" ]; then
    _STEPS_FWD+=("STEPS_PER_TASK=$STEPS_PER_TASK")
  fi
  env \
    TASKS_CSV="$PRETRAIN_TASKS_CSV" \
    DATASETS="$DS_CSV" \
    NPROC="$NPROC" \
    EPOCHS="$EPOCHS" \
    "${_STEPS_FWD[@]}" \
    OUT_DIR="$PRETRAIN_DIR" \
    RUN_NAME="${RUN_NAME:-phase4_multi_holdout_pretrain}" \
    K="$K" \
    BATCH="$BATCH" \
    CHANNELS="$CHANNELS" \
    NUM_LAYERS="$NUM_LAYERS" \
    HEADS="$HEADS" \
    CENTROIDS="$CENTROIDS" \
    LR="${LR:-1e-4}" \
    WARMUP="${WARMUP:-1000}" \
    LOSS_BALANCE="${LOSS_BALANCE:-none}" \
    bash "$REPO_ROOT/scripts/pretrain_p4d.sh" \
    > "$PRETRAIN_LOG" 2>&1
else
  # Laptop backend: single-GPU streaming pretrain.
  MAX_ROWS_TRAIN="${MAX_ROWS_TRAIN:-2000}"

  echo
  echo "=== [1/4] Building TF stores (FULL_GRAPH=$FULL_GRAPH) ==="
  for ds in $DATASETS; do
    if [ -f "$TF_STORE/$ds/.done" ]; then
      echo "  $ds: cached at $TF_STORE/$ds"
    else
      echo "  $ds: building..."
      python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds" $FULL_GRAPH_FLAG
      touch "$TF_STORE/$ds/.done"
    fi
  done

  N_TASKS=$(echo "$PRETRAIN_TASKS_CSV" | tr ',' '\n' | wc -l)
  # Laptop default: 50 steps/task/epoch -- matches the previous
  # "MAX_STEPS=300 across ~6 tasks" feel without overwhelming a
  # single-GPU streaming run.
  _laptop_steps_per_task="${STEPS_PER_TASK:-50}"
  _max_steps_per_epoch=$(( _laptop_steps_per_task * N_TASKS ))
  echo
  echo "=== [2/4] Pretraining on $N_TASKS tasks (laptop streaming) ==="
  echo "  steps/task=$_laptop_steps_per_task -> max_steps_per_epoch=$_max_steps_per_epoch"
  echo "  log: $PRETRAIN_LOG"
  torchrun --nproc_per_node 1 main_node_ddp.py \
    --tasks "$PRETRAIN_TASKS_CSV" \
    --mode streaming \
    --tf_store_dir "$TF_STORE" \
    --max_rows_per_task "$MAX_ROWS_TRAIN" \
    --num_neighbors "$K" \
    --batch_size "$BATCH" \
    --channels "$CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$HEADS" \
    --num_centroids "$CENTROIDS" \
    --epochs "$EPOCHS" \
    --max_steps_per_epoch "$_max_steps_per_epoch" \
    --num_workers 0 \
    --lr 1e-4 --warmup_steps 200 \
    --loss_balance "${LOSS_BALANCE:-none}" \
    --seed "$SEED" \
    --out_dir "$PRETRAIN_DIR" \
    --run_name "${RUN_NAME:-phase4_multi_holdout_pretrain}" $FULL_GRAPH_FLAG \
    > "$PRETRAIN_LOG" 2>&1
fi

for f in "$META" "$WEIGHTS" "$SCHEMA"; do
  if [ ! -f "$f" ]; then
    echo "ERR: pretrain didn't write expected artifact: $f" >&2
    echo "     check $PRETRAIN_LOG" >&2
    exit 3
  fi
done

# ----------------- 3+4. Per-holdout adoption: extract -> head -> tabpfn -----------------
# 'auto': pass raw embeddings if channels<=500 (TabPFN v2's cap), else
# PCA-cap at 500. No tuning needed across backbone widths; the laptop
# (channels=128) and paper-config (channels=512) both pass raw, while a
# hypothetical channels=1024 backbone would auto-PCA to 500. Override
# with PROJECTOR=none / pca64 for ablations.
PROJECTOR="${PROJECTOR:-auto}"
# mlp2 by default: a 2-layer MLP head fits a richer decision boundary
# on the 128/512-d embeddings than Linear(C, 1) and adds negligible
# compute (a few extra GEMMs per epoch on a frozen backbone). Override
# with FT_HEAD=linear for the legacy ablation.
FT_HEAD="${FT_HEAD:-mlp2}"
# RUN_TABPFN=0 skips the TabPFN post-hoc step. Default 1 keeps the
# prior behavior (also surfaces a "tabpfn not installed" warning if
# the package is missing rather than silently noop'ing).
RUN_TABPFN="${RUN_TABPFN:-1}"

# Build the list of (ds, holdout) jobs. Each pair has isolated
# extract output + precompute paths (cache_dir keyed by dataset and
# task), so concurrent extracts cannot race on shared HDF5 files.
HOLDOUT_DIRS=()
HD_DS=()
HD_TASK=()
for ds in $DATASETS; do
  holdout="${HOLDOUT_OF[$ds]:-}"
  if [ -z "$holdout" ]; then
    continue
  fi
  TASK_DIR="$OUT_DIR_BASE/${ds}.${holdout}"
  mkdir -p "$TASK_DIR/embeddings" "$TASK_DIR/finetune_head" "$TASK_DIR/tabpfn"
  HOLDOUT_DIRS+=("$TASK_DIR")
  HD_DS+=("$ds")
  HD_TASK+=("$holdout")
done

# ---- Phase A: parallel extracts, one per holdout, on its own GPU ----
# GPU pool: 0..NPROC-1. Each parallel extract pins to a distinct GPU
# via CUDA_VISIBLE_DEVICES; finetune + tabpfn stay sequential below.
GPU_POOL=()
for ((_g=0; _g<NPROC; _g++)); do GPU_POOL+=("$_g"); done
declare -A PID_GPU
declare -A PID_DESC
declare -A PID_LOG

_reap_finished() {
  local pid
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      local rc=$?
      GPU_POOL+=("${PID_GPU[$pid]}")
      if [ "$rc" -ne 0 ]; then
        echo "  WARN: extract ${PID_DESC[$pid]} (gpu=${PID_GPU[$pid]}) FAILED rc=$rc; see ${PID_LOG[$pid]}" >&2
      else
        echo "  [extract] ${PID_DESC[$pid]} done (gpu=${PID_GPU[$pid]})"
      fi
      unset "PID_GPU[$pid]"
      unset "PID_DESC[$pid]"
      unset "PID_LOG[$pid]"
    fi
  done
}

_acquire_gpu() {
  # Direct invocation only -- $() would fork a subshell that mutates
  # its own copy of GPU_POOL. Returns the index in ACQUIRED_GPU.
  while [ ${#GPU_POOL[@]} -eq 0 ]; do
    _reap_finished
    [ ${#GPU_POOL[@]} -eq 0 ] && sleep 1 || true
  done
  ACQUIRED_GPU="${GPU_POOL[0]}"
  GPU_POOL=("${GPU_POOL[@]:1}")
}

_wait_all() {
  while [ ${#PID_GPU[@]} -gt 0 ]; do
    _reap_finished
    [ ${#PID_GPU[@]} -gt 0 ] && sleep 1 || true
  done
}

echo
echo "=== [3/4] Phase A: parallel extracts (NPROC=$NPROC GPUs, ${#HD_DS[@]} jobs) ==="
for _i in "${!HD_DS[@]}"; do
  ds="${HD_DS[$_i]}"
  holdout="${HD_TASK[$_i]}"
  TASK_DIR="${HOLDOUT_DIRS[$_i]}"
  EMB_DIR="$TASK_DIR/embeddings"
  EMB_LOG="$EMB_DIR/extract.log"
  if [ -f "$EMB_DIR/test.pt" ] && [ -f "$EMB_DIR/train.pt" ] && [ -f "$EMB_DIR/val.pt" ]; then
    echo "  [extract] ${ds}.${holdout} cached"
    continue
  fi
  _acquire_gpu
  GPU="$ACQUIRED_GPU"
  desc="${ds}.${holdout}"
  echo "  [extract] $desc launching on gpu=$GPU"
  (
    CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m tools.extract_embeddings \
      --backbone_meta "$META" \
      --backbone_weights "$WEIGHTS" \
      --backbone_schema "$SCHEMA" \
      --dataset "$ds" --task "$holdout" \
      --split all \
      --num_neighbors "$K" --batch_size "$BATCH" \
      --num_workers 0 \
      --cache_dir "$CACHE" \
      --out_dir "$EMB_DIR" $FULL_GRAPH_FLAG
  ) > "$EMB_LOG" 2>&1 &
  pid=$!
  PID_GPU[$pid]="$GPU"
  PID_DESC[$pid]="$desc"
  PID_LOG[$pid]="$EMB_LOG"
done
_wait_all
echo "=== Phase A complete ==="

# ---- Phase B: finetune_head + tabpfn_eval (sequential) ----
# Same dataset as pretrain, so the encoder already knows the prefixed
# types -- no --register_new_dataset needed. These steps are minutes
# each; sequential keeps the logs readable.
echo
echo "=== [4/4] Phase B: finetune_head ($FT_HEAD) + tabpfn_eval (sequential) ==="
for _i in "${!HD_DS[@]}"; do
  ds="${HD_DS[$_i]}"
  holdout="${HD_TASK[$_i]}"
  TASK_DIR="${HOLDOUT_DIRS[$_i]}"
  EMB_DIR="$TASK_DIR/embeddings"
  FT_DIR="$TASK_DIR/finetune_head"
  TABPFN_DIR="$TASK_DIR/tabpfn"
  echo
  echo "--- ${ds}.${holdout} ---"
  if [ ! -f "$EMB_DIR/test.pt" ]; then
    echo "    extract missing -- skipping (see $EMB_DIR/extract.log)"
    continue
  fi
  echo "  [finetune] $FT_HEAD head, frozen backbone"
  python3 -m tools.finetune_head \
    --embeddings_dir "$EMB_DIR" \
    --dataset "$ds" --task "$holdout" \
    --head "$FT_HEAD" --epochs 50 --lr 1e-3 \
    --out "$FT_DIR/finetuned.pt" \
    --device cpu \
    > "$FT_DIR/finetune.log" 2>&1 || \
      echo "    WARN: finetune_head failed; see $FT_DIR/finetune.log"

  if [ "$RUN_TABPFN" = "1" ]; then
    echo "  [tabpfn] post-hoc with $PROJECTOR projector"
    if ! python3 -m tools.tabpfn_eval \
        --embeddings_dir "$EMB_DIR" \
        --dataset "$ds" --task "$holdout" \
        --projector "$PROJECTOR" \
        --out "$TABPFN_DIR/tabpfn.json" \
        > "$TABPFN_DIR/tabpfn.log" 2>&1; then
      echo "    WARN: tabpfn_eval failed (likely tabpfn not installed); see $TABPFN_DIR/tabpfn.log"
    fi
  fi
done

# ----------------- 5. Aggregate verdict -----------------
SUMMARY="$OUT_DIR_BASE/summary.json"
# Export envs the python heredoc reads. HD_LIST is a tab-joined list
# of per-holdout directories so the heredoc can split it cleanly.
export DATASETS HOLDOUTS PRETRAIN_TASKS_CSV EPOCHS SEED OUT_DIR_BASE
export HD_LIST
HD_LIST=$(printf '%s\t' "${HOLDOUT_DIRS[@]:-}")

python3 - <<'PY'
import json, os
out = {
    "datasets":         os.environ["DATASETS"].split(),
    "holdouts":         os.environ["HOLDOUTS"].split(),
    "pretrain_tasks":   os.environ["PRETRAIN_TASKS_CSV"].split(","),
    "epochs":           int(os.environ["EPOCHS"]),
    "seed":             int(os.environ["SEED"]),
    "per_holdout":      {},
}
hd_list = os.environ.get("HD_LIST", "")
for d in hd_list.split("\t"):
    d = d.strip()
    if not d:
        continue
    name = os.path.basename(d)  # e.g. "rel-f1.driver-top3"
    entry = {}
    ft = os.path.join(d, "finetune_head", "finetuned.pt")
    if os.path.exists(ft):
        import torch
        f = torch.load(ft, map_location="cpu", weights_only=False)
        entry["finetune_head"] = {
            "head_kind":      f.get("head_kind"),
            "best_epoch":     f.get("best_epoch"),
            "best_val_loss":  f.get("best_val_loss"),
            "test_metrics":   f.get("test_metrics"),
        }
    tp = os.path.join(d, "tabpfn", "tabpfn.json")
    if os.path.exists(tp):
        with open(tp) as fp:
            entry["tabpfn"] = json.load(fp)
    out["per_holdout"][name] = entry
summary_path = os.path.join(os.environ["OUT_DIR_BASE"], "summary.json")
with open(summary_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "=============================================================="
echo "Phase-4 multi-dataset holdout-task done."
echo "  Summary: $SUMMARY"
echo "=============================================================="
