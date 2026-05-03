#!/usr/bin/env bash
# Phase-5 holdout-DATASET evaluation -- "clean" task subset.
#
# Identical to scripts/holdout_dataset_eval.sh except 5 RelBench v2
# tasks are dropped from both SOURCE pretrain and TARGET adoption
# because their supervised single-task GNN baselines are at-or-below
# random per the RelBench v2 paper -- including them just adds noise
# to the GFM pretrain mix and dilutes per-task error bars on the
# holdout side. Excluded:
#
#   rel-event.event_interest-interested      (paper Table 3 GNN AUC 0.4764, below random)
#   rel-event.event_interest-not_interested  (paper Table 3 GNN AUC 0.6040, ~random)
#   rel-event.users-birthyear                (paper Table 5 GNN R^2 -0.030)
#   rel-trial.site-success                   (paper Table 9 GNN R^2 -0.483)
#   rel-amazon.item-ltv                      (paper Table 9 GNN R^2  0.032, near zero)
#
# Resulting coverage: 35 entity bcls/reg tasks across 9 datasets
# (vs 40 in holdout_dataset_eval.sh):
#   rel-amazon: 3 (was 4)   rel-avito: 5    rel-event: 3 (was 6)
#   rel-f1:     5           rel-hm:    3    rel-stack: 3
#   rel-trial:  6 (was 7)   rel-arxiv: 2    rel-ratebeer: 5
#
# Use this variant when you want the GFM transfer claim on a
# higher-signal subset; use scripts/holdout_dataset_eval.sh for the
# unfiltered all-tasks setup.
#
# Pretrain on a SOURCE dataset's tasks; freeze the backbone; extract
# embeddings on a TARGET dataset's tasks (which the backbone never
# saw at training, so its prefixed types and column stats need to
# be registered at adoption time via PR 1.2's register_dataset --
# triggered by --register_new_dataset on extract_embeddings).
#
# Train fresh head + run TabPFN on each TARGET task.
#
# Defaults (FULL_GRAPH=1; p4d-ready -- see docs/truncated_graph_caveat.md):
#   TARGET       rel-event            6-task holdout: paper-benchmarked
#                                     (RelGT Tables 1a/1b) + 3 autocomplete
#                                     joined under FULL_GRAPH=1.
#   SOURCE       <SUPPORTED_DATASETS minus TARGET>
#                                     Leave-one-dataset-out across the 9
#                                     RelBench v2 datasets that have entity
#                                     binary/regression tasks supported by
#                                     the adoption pipeline.
#   SOURCE_TASKS Union of every entity binary + regression task across the
#                resolved SOURCE datasets.
#   TARGET_TASKS Every entity binary + regression task in TARGET.
#
# Excluded (unsupported by the adoption pipeline today):
#   * rel-mimic / rel-salt -- multiclass + autocomplete cls + recommendation
#     only. Add multiclass head support to finetune_head/tabpfn_eval to
#     unblock these.
#   * Per-dataset task exclusions: link-prediction / recommendation
#     (driver-circuit-compete, paper-paper-cocitation, user-beer-* etc.)
#     and multiclass (author-category) -- structural mismatches with the
#     ranking/multiclass-aware heads we don't have yet.
#
# SOURCE accepts a space-separated list so the backbone can see
# multiple schemas before adoption to TARGET. The script unions the
# per-source default task lists; user can override with
# SOURCE_TASKS_CSV / TARGET_TASKS_CSV at any time.
#
# FULL_GRAPH=1 (default) builds with upto_test_timestamp=False so
# autocomplete-task seeds (rel-event users-birthyear /
# event_interest-*; rel-arxiv paper-citation may also need it on test
# splits) don't IndexError on the truncated CSR adjacency. Override
# with FULL_GRAPH=0 only for paper-strict, forecasting-only runs.
#
# rel-event won't fit the laptop (~25 GB peak materialization); the
# default config above targets AWS p4d.24xlarge. For laptop dry-runs
# override DATASETS subsets (e.g. SOURCE=rel-f1 TARGET=rel-hm).
#
# Usage:
#   bash scripts/holdout_dataset_eval.sh [EPOCHS] [STEPS_PER_TASK]
#
#     EPOCHS         default 5
#     STEPS_PER_TASK if set, forwarded to pretrain_p4d.sh which
#                    computes MAX_STEPS = STEPS_PER_TASK x N_tasks.
#                    Unset -> p4d backend uses pretrain_p4d.sh's
#                    default (500); laptop backend uses 50.
#     SEEDS          space-separated adoption-phase seeds; default
#                    "0 1 2". The pretrain backbone is shared across
#                    seeds (only extract -> finetune -> tabpfn
#                    repeats). Per seed, the per-row neighbor cache
#                    at ~/.cache/relbench_examples/precomputed/
#                    <TARGET>/<task>/<K>/ is wiped so the sampler
#                    regenerates it under the new RNG. SEEDS=""
#                    falls back to the legacy single-seed behavior
#                    keyed off $SEED.
#
# Output layout: results/holdout_dataset_eval/<src>_to_<tgt>/
#
# ---------------- Example invocations (copy-paste) ----------------
#
# Backend dispatch mirrors holdout_task_dev.sh:
#   NPROC>=2  -> delegate SOURCE pretrain to scripts/pretrain_p4d.sh
#                (DDP across NPROC GPUs + parallel TF/shard build +
#                 memory watchdog + pre-flight cleanup)
#   NPROC=1   -> inline single-GPU streaming pretrain (default)
#
# AWS p4d.24xlarge (default LOO: TARGET=rel-event, SOURCE = rest;
# paper-config, 3-seed adoption sweep):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d, hold out a different dataset (e.g. rel-arxiv):
#
#   NPROC=8 TARGET=rel-arxiv \
#     EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d, restrict the SOURCE pool (e.g. only rel-f1 + rel-hm):
#
#   NPROC=8 SOURCE="rel-f1 rel-hm" TARGET=rel-arxiv \
#     EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d, single-seed adoption only (skip the seed-variance sweep):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 SEEDS="0" \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d quick shake-out (default datasets, lighter budget ~1-2h):
#
#   NPROC=8 EPOCHS=3 STEPS_PER_TASK=200 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d, paper-strict (drop FULL_GRAPH so the build matches the
# RelGT paper guardrail; this also forces dropping the autocomplete
# tasks unless their seeds happen to fit the truncated CSR):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 FULL_GRAPH=0 \
#     bash scripts/holdout_dataset_eval.sh
#
# Laptop dry-run (override away from rel-event, which OOMs a 27 GB box):
#
#   SOURCE=rel-f1 TARGET=rel-hm EPOCHS=3 STEPS_PER_TASK=30 \
#     bash scripts/holdout_dataset_eval.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-${1:-5}}"
# STEPS_PER_TASK is the user-facing knob. Unset -> p4d branch falls
# through to pretrain_p4d.sh's default (500); laptop branch uses 50.
STEPS_PER_TASK="${STEPS_PER_TASK:-${2:-}}"
# Default holdout TARGET = rel-event. Picked over rel-arxiv because
# it has 6 entity binary/regression tasks (rel-arxiv only has 2),
# giving a richer per-task error-bar story for the GFM
# generalization claim, and because it's paper-benchmarked in RelGT
# Tables 1a/1b. Override TARGET=rel-arxiv (or any other supported
# dataset) to flip the holdout.
TARGET="${TARGET:-rel-event}"
# SUPPORTED_DATASETS lists every RelBench v2 dataset whose entity
# binary/regression tasks the adoption pipeline (extract_embeddings
# -> finetune_head -> tabpfn_eval) handles today. rel-mimic and
# rel-salt are excluded -- they only expose multiclass / autocomplete
# classification / recommendation tasks, and the launcher's adoption
# heads don't support those yet. When SOURCE is unset, SOURCE
# defaults to "all SUPPORTED_DATASETS except TARGET" (LOO).
SUPPORTED_DATASETS="${SUPPORTED_DATASETS:-rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial rel-arxiv rel-ratebeer}"
if [ -z "${SOURCE:-}" ]; then
  _src_auto=()
  for _ds in $SUPPORTED_DATASETS; do
    if [ "$_ds" != "$TARGET" ]; then
      _src_auto+=("$_ds")
    fi
  done
  SOURCE="${_src_auto[*]}"
fi

# SOURCE accepts a space-separated list of pretraining datasets so the
# backbone can see multiple schemas before we adopt to TARGET. Parse
# into an array; downstream code treats each source uniformly.
SOURCE_LIST=( $SOURCE )
# Slug used for output paths -- spaces don't survive a directory name.
SOURCE_SLUG=$(IFS=+; echo "${SOURCE_LIST[*]}")

for _s in "${SOURCE_LIST[@]}"; do
  if [ "$_s" = "$TARGET" ]; then
    echo "ERR: SOURCE and TARGET must differ (Phase-5 is cross-dataset)." \
         " '$_s' appears in both. For same-dataset holdout-task, use" \
         " scripts/holdout_task_dev.sh." >&2
    exit 2
  fi
done

# Source pretraining task lists. Coverage rule: every entity-level
# binary classification + regression task supported by RelBench v2 is
# included by default. Excluded by structural reason only:
#   * link-prediction / recommendation tasks (driver-circuit-compete,
#     paper-paper-cocitation) -- adoption pipeline has no ranking head
#   * multiclass tasks (author-category) -- finetune_head /
#     tabpfn_eval only have binary + regression branches
# Autocomplete tasks (results-position, qualifying-position,
# transactions-price, users-birthyear, event_interest-*) build
# cleanly under FULL_GRAPH=1 (the default below). Override the lists
# with SOURCE_TASKS_CSV / TARGET_TASKS_CSV for ablations.
DEFAULT_RELF1_ALL=(
  "rel-f1.driver-position:1.0"
  "rel-f1.driver-dnf:1.0"
  "rel-f1.driver-top3:1.0"
  "rel-f1.results-position:1.0"
  "rel-f1.qualifying-position:1.0"
)
DEFAULT_RELHM_ALL=(
  "rel-hm.user-churn:1.0"
  "rel-hm.item-sales:1.0"
  "rel-hm.transactions-price:1.0"
)
# rel-event (clean): 3 user-* tasks only. event_interest-* and
# users-birthyear are commented out below with their RelBench v2
# baseline numbers -- uncomment if you want to ablate.
DEFAULT_RELEVENT_ALL=(
  "rel-event.user-attendance:1.0"
  "rel-event.user-repeat:1.0"
  "rel-event.user-ignore:1.0"
  # "rel-event.event_interest-interested:1.0"      # paper Table 3 GNN AUC 0.4764 -- below random
  # "rel-event.event_interest-not_interested:1.0"  # paper Table 3 GNN AUC 0.6040 -- ~random
  # "rel-event.users-birthyear:1.0"                # paper Table 5 GNN R^2 -0.030 -- negative
)
# rel-arxiv: paper-citation (binary) + author-publication (regression).
# author-category is multiclass and paper-paper-cocitation is link-pred;
# neither is supported by the launcher's adoption pipeline today.
DEFAULT_RELARXIV_ALL=(
  "rel-arxiv.paper-citation:1.0"
  "rel-arxiv.author-publication:1.0"
)
# rel-amazon (clean): 3 entity tasks (2 binary churn + 1 reg LTV).
# item-ltv is commented out below with its baseline number --
# uncomment to ablate. review-rating is autocomplete classification
# and not supported by the adoption pipeline today.
DEFAULT_RELAMAZON_ALL=(
  "rel-amazon.user-churn:1.0"
  "rel-amazon.item-churn:1.0"
  "rel-amazon.user-ltv:1.0"
  # "rel-amazon.item-ltv:1.0"  # paper Table 9 GNN R^2 0.032 -- near zero signal
)
# rel-avito: 5 entity tasks. ad-ctr (reg) + user-* (binary
# forecasting) + 2 autocomplete binaries (searchstream-click,
# searchinfo-isuserloggedon) joined under FULL_GRAPH=1.
DEFAULT_RELAVITO_ALL=(
  "rel-avito.ad-ctr:1.0"
  "rel-avito.user-visits:1.0"
  "rel-avito.user-clicks:1.0"
  "rel-avito.searchstream-click:1.0"
  "rel-avito.searchinfo-isuserloggedon:1.0"
)
# rel-stack: 3 entity tasks (RelBench v1 forecasting only; no v2
# autocomplete on this dataset).
DEFAULT_RELSTACK_ALL=(
  "rel-stack.user-engagement:1.0"
  "rel-stack.user-badge:1.0"
  "rel-stack.post-votes:1.0"
)
# rel-trial (clean): 6 entity tasks. site-success is commented out
# below with its baseline -- uncomment to ablate. Keeps the rest:
# study-outcome (binary forecasting), study-adverse (reg), plus the
# 4 autocomplete tasks (studies-enrollment reg, studies-has_dmc bin,
# eligibilities-{adult,child} bin).
DEFAULT_RELTRIAL_ALL=(
  "rel-trial.study-outcome:1.0"
  "rel-trial.study-adverse:1.0"
  # "rel-trial.site-success:1.0"  # paper Table 9 GNN R^2 -0.483 -- strongly negative
  "rel-trial.studies-enrollment:1.0"
  "rel-trial.studies-has_dmc:1.0"
  "rel-trial.eligibilities-adult:1.0"
  "rel-trial.eligibilities-child:1.0"
)
# rel-ratebeer: 5 entity tasks (3 churn binaries forecasting,
# user-count reg, beer_ratings-total_score reg autocomplete).
# user-beer-favorite / user-beer-liked / user-place-liked are
# recommendation tasks and stay out of the launcher's defaults.
DEFAULT_RELRATEBEER_ALL=(
  "rel-ratebeer.beer-churn:1.0"
  "rel-ratebeer.user-churn:1.0"
  "rel-ratebeer.brewer-dormant:1.0"
  "rel-ratebeer.user-count:1.0"
  "rel-ratebeer.beer_ratings-total_score:1.0"
)

# Per-dataset default-task lookup so multi-source pretrain (and any
# TARGET) can resolve auto-defaults without a chain of if/elif.
_default_tasks_for() {
  case "$1" in
    rel-f1)       printf '%s\n' "${DEFAULT_RELF1_ALL[@]}" ;;
    rel-event)    printf '%s\n' "${DEFAULT_RELEVENT_ALL[@]}" ;;
    rel-hm)       printf '%s\n' "${DEFAULT_RELHM_ALL[@]}" ;;
    rel-arxiv)    printf '%s\n' "${DEFAULT_RELARXIV_ALL[@]}" ;;
    rel-amazon)   printf '%s\n' "${DEFAULT_RELAMAZON_ALL[@]}" ;;
    rel-avito)    printf '%s\n' "${DEFAULT_RELAVITO_ALL[@]}" ;;
    rel-stack)    printf '%s\n' "${DEFAULT_RELSTACK_ALL[@]}" ;;
    rel-trial)    printf '%s\n' "${DEFAULT_RELTRIAL_ALL[@]}" ;;
    rel-ratebeer) printf '%s\n' "${DEFAULT_RELRATEBEER_ALL[@]}" ;;
    *)            return 1 ;;
  esac
}

if [ -z "${SOURCE_TASKS_CSV:-}" ]; then
  SRC_ARR=()
  for _s in "${SOURCE_LIST[@]}"; do
    _ds_tasks=$(_default_tasks_for "$_s") || {
      echo "ERR: no default SOURCE_TASKS for source dataset '$_s'; " \
           "set SOURCE_TASKS_CSV explicitly." >&2
      exit 2
    }
    while IFS= read -r _t; do SRC_ARR+=("$_t"); done <<< "$_ds_tasks"
  done
  SOURCE_TASKS_CSV=$(IFS=,; echo "${SRC_ARR[*]}")
fi

if [ -z "${TARGET_TASKS_CSV:-}" ]; then
  _ds_tasks=$(_default_tasks_for "$TARGET") || {
    echo "ERR: no default TARGET_TASKS for TARGET=$TARGET; " \
         "set TARGET_TASKS_CSV explicitly." >&2
    exit 2
  }
  TGT_ARR=()
  while IFS= read -r _t; do TGT_ARR+=("$_t"); done <<< "$_ds_tasks"
  TARGET_TASKS_CSV=$(IFS=,; echo "${TGT_ARR[*]}")
fi

CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"
# FULL_GRAPH=1 (default) builds with upto_test_timestamp=False so
# autocomplete-task seeds (rel-event users-birthyear /
# event_interest-*; potentially rel-arxiv as TARGET) don't IndexError
# on the truncated CSR adjacency. Mode-aware TF_STORE keeps the
# truncated and full builds isolated. FULL_GRAPH is exported so
# pretrain_p4d.sh inherits it.
FULL_GRAPH="${FULL_GRAPH:-1}"
export FULL_GRAPH
if [ "$FULL_GRAPH" = "1" ]; then
    TF_STORE="$CACHE/tf_store_full"
    FULL_GRAPH_FLAG="--full_graph"
else
    TF_STORE="$CACHE/tf_store"
    FULL_GRAPH_FLAG=""
fi
SEED="${SEED:-0}"
# Adoption-time sweep: run extract -> finetune -> tabpfn once per
# seed in $SEEDS. The pretrain backbone is shared across seeds (it's
# expensive and the user-facing question is "how stable is the
# adoption pipeline" not "how stable is pretraining"). Per-seed each
# repeat wipes ~/.cache/relbench_examples/precomputed/<TARGET>/<task>
# so the per-row neighbor sampling done by gfm_data/sampler.py
# (random.sample on the neighbor list) is regenerated. Set SEEDS=""
# (or " ") to fall back to single-seed legacy behavior keyed off
# $SEED.
SEEDS="${SEEDS:-0 1 2}"
OUT_DIR_BASE="${OUT_DIR:-results/holdout_dataset_eval}"
RUN_DIR="$OUT_DIR_BASE/${SOURCE_SLUG}_to_${TARGET}"
PRETRAIN_DIR="$RUN_DIR/pretrain"

# ---- backend dispatch (mirror of holdout_task_dev.sh) ----
# NPROC>=2 -> delegate pretrain to scripts/pretrain_p4d.sh (DDP +
# parallel build phases + memory watchdog + pre-flight cleanup).
# NPROC=1  -> inline single-GPU streaming pretrain (default).
NPROC="${NPROC:-1}"
if [ -z "${PRETRAIN_BACKEND:-}" ]; then
  if [ "$NPROC" -ge 2 ]; then
    PRETRAIN_BACKEND=p4d
  else
    PRETRAIN_BACKEND=laptop
  fi
fi

# Per-backend training-config defaults. Set BEFORE the dispatch so
# the adoption phase (extract_embeddings + finetune_head + tabpfn)
# downstream reads matching K + BATCH.
if [ "$PRETRAIN_BACKEND" = "p4d" ]; then
  K="${K:-300}"
  BATCH="${BATCH:-512}"
  CHANNELS="${CHANNELS:-512}"
  NUM_LAYERS="${NUM_LAYERS:-4}"
  HEADS="${HEADS:-4}"
  CENTROIDS="${CENTROIDS:-4096}"
  # The adoption-phase extract is single-process today (no DDP), so
  # the dataloader's per-row neighbor sampling is the bottleneck.
  # Bump num_workers on p4d to parallelize that across CPU cores.
  EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-8}"
else
  K="${K:-64}"
  BATCH="${BATCH:-128}"
  CHANNELS="${CHANNELS:-128}"
  NUM_LAYERS="${NUM_LAYERS:-1}"
  HEADS="${HEADS:-4}"
  CENTROIDS="${CENTROIDS:-512}"
  EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-0}"
fi
MAX_ROWS_TRAIN="${MAX_ROWS_TRAIN:-2000}"

mkdir -p "$PRETRAIN_DIR"

# WANDB_MODE left to the caller's environment / wandb's own default.
# WANDB_SILENT default true to avoid interleaving wandb spam with
# script stdout; override with WANDB_SILENT=false if desired.
export WANDB_SILENT="${WANDB_SILENT:-true}"

echo "=============================================================="
echo "Phase-5 cross-dataset adoption: SOURCE=[${SOURCE_LIST[*]}] -> TARGET=$TARGET"
echo "  SOURCE_TASKS: $SOURCE_TASKS_CSV"
echo "  TARGET_TASKS: $TARGET_TASKS_CSV"
echo "  EPOCHS=$EPOCHS  STEPS_PER_TASK=${STEPS_PER_TASK:-<default>}  SEED=$SEED"
echo "  FULL_GRAPH=$FULL_GRAPH  TF_STORE=$TF_STORE"
echo "  RUN_DIR: $RUN_DIR"
echo "=============================================================="

# ----------------- Pretrain artifact paths (shared by both backends) -----------------
PRETRAIN_LOG="$PRETRAIN_DIR/run.log"
META="$PRETRAIN_DIR/multi_task/backbone_meta.json"
WEIGHTS="$PRETRAIN_DIR/multi_task/best_backbone.pt"
SCHEMA="$PRETRAIN_DIR/multi_task/backbone_schema.pt"

# ----------------- 1+2. TF stores + SOURCE pretrain -----------------
# The TARGET TF store is also built so step-3 extract_embeddings has
# memmap shards to read from (`--use_tf_store` below).
if [ -f "$WEIGHTS" ] && [ -f "$META" ] && [ -f "$SCHEMA" ]; then
  echo "[pretrain] artifacts cached at $PRETRAIN_DIR/multi_task/"
elif [ "$PRETRAIN_BACKEND" = "p4d" ]; then
  # Delegate to pretrain_p4d.sh. It handles parallel TF/shard
  # builds, DDP launch, memory watchdog, etc. We pass TASKS_CSV to
  # bypass relbench's full-task enumeration. NOTE: pretrain_p4d.sh
  # only builds TF stores for datasets in $DATASETS, so we ALSO
  # pre-build TARGET's TF store here for the post-pretrain
  # extract_embeddings step (which reads it via --use_tf_store).
  echo
  echo "=== Pre-building TARGET TF store for adoption phase ==="
  if [ ! -f "$TF_STORE/$TARGET/.done" ]; then
    python3 tools/build_tf_store.py --dataset "$TARGET" --out_dir "$TF_STORE/$TARGET" $FULL_GRAPH_FLAG
    touch "$TF_STORE/$TARGET/.done"
  else
    echo "  $TARGET: cached at $TF_STORE/$TARGET"
  fi

  echo
  echo "=== Delegating SOURCE pretrain to pretrain_p4d.sh (NPROC=$NPROC) ==="
  echo "  log: $PRETRAIN_LOG"
  # STEPS_PER_TASK forwarded only when set; unset = pretrain_p4d.sh
  # default (500). Forwarding "" would zero out MAX_STEPS downstream.
  _STEPS_FWD=()
  if [ -n "${STEPS_PER_TASK:-}" ]; then
    _STEPS_FWD+=("STEPS_PER_TASK=$STEPS_PER_TASK")
  fi
  _src_csv=$(IFS=,; echo "${SOURCE_LIST[*]}")
  env \
    TASKS_CSV="$SOURCE_TASKS_CSV" \
    DATASETS="$_src_csv" \
    NPROC="$NPROC" \
    EPOCHS="$EPOCHS" \
    "${_STEPS_FWD[@]}" \
    OUT_DIR="$PRETRAIN_DIR" \
    RUN_NAME="phase5_${SOURCE_SLUG}_to_${TARGET}_pretrain" \
    K="$K" \
    BATCH="$BATCH" \
    CHANNELS="$CHANNELS" \
    NUM_LAYERS="$NUM_LAYERS" \
    HEADS="$HEADS" \
    CENTROIDS="$CENTROIDS" \
    LR="${LR:-1e-4}" \
    WARMUP="${WARMUP:-1000}" \
    LOSS_BALANCE="${LOSS_BALANCE:-none}" \
    SEED="$SEED" \
    SHARDS_SUBDIR="${SHARDS_SUBDIR:-}" \
    bash "$REPO_ROOT/scripts/pretrain_p4d.sh" \
    > "$PRETRAIN_LOG" 2>&1
else
  # Laptop backend: single-GPU streaming pretrain, with both
  # SOURCE and TARGET TF stores built sequentially.
  for ds in "${SOURCE_LIST[@]}" "$TARGET"; do
    if [ ! -f "$TF_STORE/$ds/.done" ]; then
      echo
      echo "=== Building TF store for $ds ==="
      python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds" $FULL_GRAPH_FLAG
      touch "$TF_STORE/$ds/.done"
    else
      echo "[setup] TF store cached at $TF_STORE/$ds"
    fi
  done

  N_SOURCE_TASKS=$(echo "$SOURCE_TASKS_CSV" | tr ',' '\n' | wc -l)
  _laptop_steps_per_task="${STEPS_PER_TASK:-50}"
  _max_steps_per_epoch=$(( _laptop_steps_per_task * N_SOURCE_TASKS ))
  echo
  echo "=== Pretraining on SOURCE=[${SOURCE_LIST[*]}] ($SOURCE_TASKS_CSV) ==="
  echo "  steps/task=$_laptop_steps_per_task -> max_steps_per_epoch=$_max_steps_per_epoch"
  echo "  log: $PRETRAIN_LOG"
  torchrun --nproc_per_node 1 main_node_ddp.py \
    --tasks "$SOURCE_TASKS_CSV" \
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
    --run_name "phase5_${SOURCE_SLUG}_to_${TARGET}_pretrain" $FULL_GRAPH_FLAG \
    > "$PRETRAIN_LOG" 2>&1
fi

for f in "$META" "$WEIGHTS" "$SCHEMA"; do
  if [ ! -f "$f" ]; then
    echo "ERR: pretrain didn't write expected artifact: $f" >&2
    echo "     check $PRETRAIN_LOG" >&2
    exit 3
  fi
done

# ----------------- 3. For each TARGET task x SEED, extract + finetune + tabpfn -----------------
# 'auto': raw if channels<=500, else PCA-cap at 500. Adapts to whatever
# channels the upstream backbone produces. Override with PROJECTOR=none /
# pca64 for ablations.
PROJECTOR="${PROJECTOR:-auto}"

# Resolve the seed list. Empty SEEDS = legacy single-seed behavior at
# $SEED, keyed off the same per-task path layout.
if [ -z "${SEEDS// /}" ]; then
  SEED_LIST=( "$SEED" )
else
  read -ra SEED_LIST <<< "$SEEDS"
fi

# Parse TARGET_TASKS_CSV "ds.task:weight,..." -> just the task names.
IFS=',' read -ra TGT_PAIRS <<< "$TARGET_TASKS_CSV"
TASK_SUMMARIES=()
TASK_NAMES=()
for pair in "${TGT_PAIRS[@]}"; do
  pair_no_weight="${pair%%:*}"
  task_name="${pair_no_weight#${TARGET}.}"
  TASK_DIR="$RUN_DIR/${task_name}"
  mkdir -p "$TASK_DIR"
  TASK_SUMMARIES+=("$TASK_DIR")
  TASK_NAMES+=("$task_name")
  for ADOPT_SEED in "${SEED_LIST[@]}"; do
    mkdir -p \
      "$TASK_DIR/seed${ADOPT_SEED}/embeddings" \
      "$TASK_DIR/seed${ADOPT_SEED}/finetune_head" \
      "$TASK_DIR/seed${ADOPT_SEED}/tabpfn"
  done
done

# ---- Phase A: parallel extracts, one per (task, seed), each on its own GPU ----
# GPU pool: 0..NPROC-1. Each parallel extract picks a free GPU via
# CUDA_VISIBLE_DEVICES. Per-(task, seed) precomputed_dir keeps the
# HDF5 caches isolated so concurrent jobs never race.
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
  # IMPORTANT: this function MUST be invoked directly, not via
  # ``$(_acquire_gpu)`` -- a $() subshell would mutate its own copy
  # of GPU_POOL / PID_GPU and the parent's pool would never drain,
  # so every job would land on gpu=0. The function returns the
  # acquired index in the global ``ACQUIRED_GPU``.
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
echo "=== Phase A: parallel extracts (NPROC=$NPROC GPUs, $((${#TGT_PAIRS[@]} * ${#SEED_LIST[@]})) jobs) ==="
for task_name in "${TASK_NAMES[@]}"; do
  for ADOPT_SEED in "${SEED_LIST[@]}"; do
    SEED_DIR="$RUN_DIR/${task_name}/seed${ADOPT_SEED}"
    EMB_DIR="$SEED_DIR/embeddings"
    EMB_LOG="$EMB_DIR/extract.log"
    if [ -f "$EMB_DIR/test.pt" ]; then
      echo "  [extract] ${task_name} seed=$ADOPT_SEED cached"
      continue
    fi
    _acquire_gpu
    GPU="$ACQUIRED_GPU"
    PRECOMP_DIR="$SEED_DIR/precomputed"
    desc="${task_name} seed=$ADOPT_SEED"
    echo "  [extract] $desc launching on gpu=$GPU"
    (
      CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m tools.extract_embeddings \
        --backbone_meta "$META" \
        --backbone_weights "$WEIGHTS" \
        --backbone_schema "$SCHEMA" \
        --dataset "$TARGET" --task "$task_name" \
        --register_new_dataset \
        --split all \
        --num_neighbors "$K" --batch_size "$BATCH" \
        --num_workers "$EXTRACT_NUM_WORKERS" \
        --cache_dir "$CACHE" \
        --use_tf_store \
        --seed "$ADOPT_SEED" \
        --precomputed_dir "$PRECOMP_DIR" \
        --out_dir "$EMB_DIR" $FULL_GRAPH_FLAG
    ) > "$EMB_LOG" 2>&1 &
    pid=$!
    PID_GPU[$pid]="$GPU"
    PID_DESC[$pid]="$desc"
    PID_LOG[$pid]="$EMB_LOG"
  done
done
_wait_all
echo "=== Phase A complete ==="

# ---- Phase B: finetune_head + tabpfn_eval (cheap, sequential) ----
# Each finetune is ~minutes and tabpfn ~10-20 min on this scale; the
# bottleneck was the extract above. Stay sequential here so we don't
# fight the multi-GPU phase A and so logs stay readable.
#
# Default to mlp2 (2-layer MLP w/ GELU) -- a richer head than Linear
# fits the 128/512-d frozen embeddings better at negligible compute
# cost (a few extra GEMMs per epoch on a frozen backbone). Override
# with FT_HEAD=linear for the legacy linear-probe ablation.
FT_HEAD="${FT_HEAD:-mlp2}"
# RUN_TABPFN=0 skips the TabPFN post-hoc step (each call is ~10-20
# min on this scale). Default 1 keeps the prior behavior.
RUN_TABPFN="${RUN_TABPFN:-1}"
echo
echo "=== Phase B: finetune_head ($FT_HEAD)$([ "$RUN_TABPFN" = "1" ] && echo " + tabpfn_eval") (sequential) ==="
for task_name in "${TASK_NAMES[@]}"; do
  echo
  echo "  ${TARGET}.${task_name}"
  for ADOPT_SEED in "${SEED_LIST[@]}"; do
    SEED_DIR="$RUN_DIR/${task_name}/seed${ADOPT_SEED}"
    EMB_DIR="$SEED_DIR/embeddings"
    FT_DIR="$SEED_DIR/finetune_head"
    TABPFN_DIR="$SEED_DIR/tabpfn"
    if [ ! -f "$EMB_DIR/test.pt" ]; then
      echo "    seed=$ADOPT_SEED: extract missing -- skipping (see $EMB_DIR/extract.log)"
      continue
    fi
    echo "    seed=$ADOPT_SEED finetune ($FT_HEAD)"
    python3 -m tools.finetune_head \
      --embeddings_dir "$EMB_DIR" \
      --dataset "$TARGET" --task "$task_name" \
      --head "$FT_HEAD" --epochs 50 --lr 1e-3 \
      --seed "$ADOPT_SEED" \
      --out "$FT_DIR/finetuned.pt" \
      > "$FT_DIR/finetune.log" 2>&1 || \
        echo "      WARN: finetune_head failed; see $FT_DIR/finetune.log"
    if [ "$RUN_TABPFN" = "1" ]; then
      echo "    seed=$ADOPT_SEED tabpfn"
      python3 -m tools.tabpfn_eval \
        --embeddings_dir "$EMB_DIR" \
        --dataset "$TARGET" --task "$task_name" \
        --projector "$PROJECTOR" \
        --seed "$ADOPT_SEED" \
        --out "$TABPFN_DIR/tabpfn.json" \
        > "$TABPFN_DIR/tabpfn.log" 2>&1 || \
          echo "      WARN: tabpfn_eval failed; see $TABPFN_DIR/tabpfn.log"
    fi
  done
done

# ----------------- 4. Aggregate (per-task mean +/- SD across seeds) -----------------
SUMMARY="$RUN_DIR/summary.json"
SEED_LIST_STR="${SEED_LIST[*]}"
python3 - <<PY
import json, os, statistics
out = {
    "source": "${SOURCE_LIST[*]}",
    "target": "$TARGET",
    "source_tasks": "$SOURCE_TASKS_CSV".split(","),
    "target_tasks": "$TARGET_TASKS_CSV".split(","),
    "epochs": int($EPOCHS),
    "seeds": [int(s) for s in "$SEED_LIST_STR".split() if s.strip()],
    "per_task": {},
}

def _mean_sd(values):
    """Return (mean, std) for a numeric list. SD undefined for n<2."""
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

for d in """$(IFS=$'\n'; echo "${TASK_SUMMARIES[*]}")""".split():
    if not d.strip():
        continue
    name = os.path.basename(d.strip())
    seed_dirs = sorted(
        sd for sd in os.listdir(d)
        if sd.startswith("seed") and os.path.isdir(os.path.join(d, sd))
    )
    finetune_runs = []
    tabpfn_runs = []
    for sd in seed_dirs:
        seed_label = sd[len("seed"):]
        ft_path = os.path.join(d, sd, "finetune_head", "finetuned.pt")
        if os.path.exists(ft_path):
            import torch
            f = torch.load(ft_path, map_location="cpu", weights_only=False)
            finetune_runs.append({
                "seed": seed_label,
                "head_kind": f.get("head_kind"),
                "best_epoch": f.get("best_epoch"),
                "best_val_loss": f.get("best_val_loss"),
                "test_metrics": f.get("test_metrics"),
            })
        tp_path = os.path.join(d, sd, "tabpfn", "tabpfn.json")
        if os.path.exists(tp_path):
            with open(tp_path) as fp:
                tabpfn_json = json.load(fp)
            tabpfn_json["seed"] = seed_label
            tabpfn_runs.append(tabpfn_json)

    entry = {"finetune_head_seeds": finetune_runs,
             "tabpfn_seeds": tabpfn_runs}

    # Roll up the per-seed test metrics into mean +/- SD per metric.
    # Same shape for both adoption methods.
    def _roll(runs, metrics_field):
        agg = {}
        if not runs:
            return agg
        # Collect metric keys from the first run that has them.
        keys = set()
        for r in runs:
            m = r.get(metrics_field) or {}
            keys.update(m.keys())
        for k in sorted(keys):
            stats = _mean_sd([
                (r.get(metrics_field) or {}).get(k) for r in runs
            ])
            if stats is not None:
                agg[k] = stats
        return agg

    entry["finetune_head_summary"] = _roll(finetune_runs, "test_metrics")
    entry["tabpfn_summary"] = _roll(tabpfn_runs, "test_metrics")
    out["per_task"][name] = entry

with open("$SUMMARY", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "=============================================================="
echo "Phase-5 cross-dataset done. Summary: $SUMMARY"
echo "=============================================================="
