#!/usr/bin/env bash
# Phase-5 holdout-DATASET evaluation.
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
#   SOURCE       "rel-f1 rel-event"   space-separated multi-source pretrain
#   TARGET       rel-arxiv            unseen schema -- adoption via
#                                     register_new_dataset
#   SOURCE_TASKS union of paper-benchmarked rel-f1 + rel-event tasks
#                (3 rel-f1 + 6 rel-event = 9 tasks)
#   TARGET_TASKS rel-arxiv.paper-citation (binary) +
#                rel-arxiv.author-publication (regression)
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
# AWS p4d.24xlarge (default rel-f1 + rel-event -> rel-arxiv, paper-config):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d quick shake-out (default datasets, lighter budget ~1-2h):
#
#   NPROC=8 EPOCHS=3 STEPS_PER_TASK=200 \
#     bash scripts/holdout_dataset_eval.sh
#
# AWS p4d, single-source override (rel-f1 -> rel-hm):
#
#   NPROC=8 SOURCE=rel-f1 TARGET=rel-hm \
#     EPOCHS=10 STEPS_PER_TASK=500 \
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
SOURCE="${SOURCE:-rel-f1 rel-event}"
TARGET="${TARGET:-rel-arxiv}"

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

# Source pretraining task lists. Restricted to the paper-benchmarked
# subset per docs/truncated_graph_caveat.md -- the omitted tasks
# (driver-circuit-compete, results-position, qualifying-position,
# transactions-price) IndexError on val/test seeds against the
# truncated CSR adjacency. Override SOURCE_TASKS_CSV /
# TARGET_TASKS_CSV to opt in (and accept the crash unless
# upto_test_timestamp is also flipped).
DEFAULT_RELF1_ALL=(
  "rel-f1.driver-position:1.0"
  "rel-f1.driver-dnf:1.0"
  "rel-f1.driver-top3:1.0"
)
DEFAULT_RELHM_ALL=(
  "rel-hm.user-churn:1.0"
  "rel-hm.item-sales:1.0"
)
# rel-event: 6 entity tasks. user-* are paper-benchmarked
# (RelGT paper Table 1a/1b); event_interest-* and users-birthyear
# are autocomplete and need FULL_GRAPH=1 to build cleanly.
DEFAULT_RELEVENT_ALL=(
  "rel-event.user-attendance:1.0"
  "rel-event.user-repeat:1.0"
  "rel-event.user-ignore:1.0"
  "rel-event.event_interest-interested:1.0"
  "rel-event.event_interest-not_interested:1.0"
  "rel-event.users-birthyear:1.0"
)
# rel-arxiv: paper-citation (binary) + author-publication (regression).
# author-category is multiclass and paper-paper-cocitation is link-pred;
# neither is supported by the launcher's adoption pipeline today.
DEFAULT_RELARXIV_ALL=(
  "rel-arxiv.paper-citation:1.0"
  "rel-arxiv.author-publication:1.0"
)

# Per-dataset default-task lookup so multi-source pretrain (and any
# TARGET) can resolve auto-defaults without a chain of if/elif.
_default_tasks_for() {
  case "$1" in
    rel-f1)    printf '%s\n' "${DEFAULT_RELF1_ALL[@]}" ;;
    rel-event) printf '%s\n' "${DEFAULT_RELEVENT_ALL[@]}" ;;
    rel-hm)    printf '%s\n' "${DEFAULT_RELHM_ALL[@]}" ;;
    rel-arxiv) printf '%s\n' "${DEFAULT_RELARXIV_ALL[@]}" ;;
    *)         return 1 ;;
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
else
  K="${K:-64}"
  BATCH="${BATCH:-128}"
  CHANNELS="${CHANNELS:-128}"
  NUM_LAYERS="${NUM_LAYERS:-1}"
  HEADS="${HEADS:-4}"
  CENTROIDS="${CENTROIDS:-512}"
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

# ----------------- 3. For each TARGET task, extract + finetune + tabpfn -----------------
# 'auto': raw if channels<=500, else PCA-cap at 500. Adapts to whatever
# channels the upstream backbone produces. Override with PROJECTOR=none /
# pca64 for ablations.
PROJECTOR="${PROJECTOR:-auto}"

# Parse TARGET_TASKS_CSV "ds.task:weight,..." -> just the task names.
IFS=',' read -ra TGT_PAIRS <<< "$TARGET_TASKS_CSV"
TASK_SUMMARIES=()
for pair in "${TGT_PAIRS[@]}"; do
  # Strip optional weight, then leading "<ds>." prefix.
  pair_no_weight="${pair%%:*}"
  task_name="${pair_no_weight#${TARGET}.}"

  TASK_DIR="$RUN_DIR/${task_name}"
  EMB_DIR="$TASK_DIR/embeddings"
  FT_DIR="$TASK_DIR/finetune_head"
  TABPFN_DIR="$TASK_DIR/tabpfn"
  mkdir -p "$EMB_DIR" "$FT_DIR" "$TABPFN_DIR"

  echo
  echo "=== TARGET task: ${TARGET}.${task_name} ==="

  EMB_LOG="$EMB_DIR/extract.log"
  if [ ! -f "$EMB_DIR/test.pt" ]; then
    echo "  [extract] register_new_dataset on $TARGET, then forward"
    python3 -m tools.extract_embeddings \
      --backbone_meta "$META" \
      --backbone_weights "$WEIGHTS" \
      --backbone_schema "$SCHEMA" \
      --dataset "$TARGET" --task "$task_name" \
      --register_new_dataset \
      --split all \
      --num_neighbors "$K" --batch_size "$BATCH" \
      --num_workers 0 \
      --cache_dir "$CACHE" \
      --use_tf_store \
      --out_dir "$EMB_DIR" $FULL_GRAPH_FLAG \
      > "$EMB_LOG" 2>&1
  else
    echo "  [extract] embeddings cached"
  fi

  echo "  [finetune] linear head"
  python3 -m tools.finetune_head \
    --embeddings_dir "$EMB_DIR" \
    --dataset "$TARGET" --task "$task_name" \
    --head linear --epochs 50 --lr 1e-3 \
    --out "$FT_DIR/finetuned.pt" \
    --device cpu \
    > "$FT_DIR/finetune.log" 2>&1 || \
      echo "  WARN: finetune_head failed; see $FT_DIR/finetune.log"

  echo "  [tabpfn] post-hoc"
  python3 -m tools.tabpfn_eval \
    --embeddings_dir "$EMB_DIR" \
    --dataset "$TARGET" --task "$task_name" \
    --projector "$PROJECTOR" \
    --out "$TABPFN_DIR/tabpfn.json" \
    > "$TABPFN_DIR/tabpfn.log" 2>&1 || \
      echo "  WARN: tabpfn_eval failed; see $TABPFN_DIR/tabpfn.log"

  TASK_SUMMARIES+=("$TASK_DIR")
done

# ----------------- 4. Aggregate -----------------
SUMMARY="$RUN_DIR/summary.json"
python3 - <<PY
import json, os
out = {
    "source": "${SOURCE_LIST[*]}",
    "target": "$TARGET",
    "source_tasks": "$SOURCE_TASKS_CSV".split(","),
    "target_tasks": "$TARGET_TASKS_CSV".split(","),
    "epochs": int($EPOCHS),
    "seed": int($SEED),
    "per_task": {},
}
for d in """$(IFS=$'\n'; echo "${TASK_SUMMARIES[*]}")""".split():
    if not d.strip():
        continue
    name = os.path.basename(d.strip())
    entry = {}
    ft = os.path.join(d, "finetune_head", "finetuned.pt")
    if os.path.exists(ft):
        import torch
        f = torch.load(ft, map_location="cpu", weights_only=False)
        entry["finetune_head"] = {
            "head_kind": f.get("head_kind"),
            "best_epoch": f.get("best_epoch"),
            "best_val_loss": f.get("best_val_loss"),
            "test_metrics": f.get("test_metrics"),
        }
    tp = os.path.join(d, "tabpfn", "tabpfn.json")
    if os.path.exists(tp):
        with open(tp) as fp:
            entry["tabpfn"] = json.load(fp)
    out["per_task"][name] = entry
with open("$SUMMARY", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "=============================================================="
echo "Phase-5 cross-dataset done. Summary: $SUMMARY"
echo "=============================================================="
