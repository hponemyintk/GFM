#!/usr/bin/env bash
# Phase-4 holdout-task development loop.
#
# Pretrain on N-1 of a dataset's tasks; freeze the backbone; extract
# embeddings on the held-out task; train a fresh head + run TabPFN;
# compare against the from-scratch single-task baseline (the existing
# scripts/pretrain_laptop.sh single-task path on the held-out task).
#
# This is the laptop-scale validation that the embedding-extraction +
# adoption tooling actually produces useful representations for tasks
# the backbone never saw the head for. Same dataset → same schema →
# no Phase-5 cross-dataset register_dataset path; just exercises the
# Phase 1-3 pipeline end-to-end.
#
# Defaults:
#   DATASET     rel-f1
#   HOLDOUT     driver-top3   (binary; AUROC is the cleanest signal)
#   TASKS_KEEP  the other 5 rel-f1 tasks
#
# Override DATASET=rel-hm HOLDOUT=user-churn TASKS_KEEP="..." to run
# the rel-hm version. The PRETRAINED tasks must NOT include HOLDOUT.
#
# Usage:
#   bash scripts/holdout_task_dev.sh [EPOCHS] [MAX_STEPS]
#
#     EPOCHS        default 5  (same budget as parity sweep)
#     MAX_STEPS     default 300
#
# Output layout under $OUT_DIR (default results/holdout_task_dev/):
#   <ds>_<holdout>/
#     pretrain/                # multi-task pretrain on N-1 tasks
#       multi_task/<seed>.json
#       best_full.pt
#       best_backbone.pt
#       backbone_meta.json
#       backbone_schema.pt
#     embeddings/
#       train.pt val.pt test.pt
#     finetune_head/
#       finetuned.pt
#     tabpfn/
#       tabpfn.json
#     summary.json             # holdout-task verdict (this script writes it)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-${1:-5}}"
MAX_STEPS="${MAX_STEPS:-${2:-300}}"
DATASET="${DATASET:-rel-f1}"
HOLDOUT="${HOLDOUT:-driver-top3}"

# Tasks to KEEP for pretraining (the holdout task must NOT be here).
# Default sets cover rel-f1 (5/6) and rel-hm (2/3). Override
# TASKS_KEEP_CSV='ds.task:1.0,ds.task:1.0' for custom mixes.
DEFAULT_RELF1_TASKS=(
  "rel-f1.driver-position:1.0"
  "rel-f1.driver-dnf:1.0"
  "rel-f1.driver-circuit-compete:1.0"
  "rel-f1.results-position:1.0"
  "rel-f1.qualifying-position:1.0"
)
DEFAULT_RELHM_TASKS=(
  "rel-hm.item-sales:1.0"
  "rel-hm.transactions-price:1.0"
)

if [ -z "${TASKS_KEEP_CSV:-}" ]; then
  if [ "$DATASET" = "rel-f1" ]; then
    TASKS_KEEP_ARR=("${DEFAULT_RELF1_TASKS[@]}")
  elif [ "$DATASET" = "rel-hm" ]; then
    TASKS_KEEP_ARR=("${DEFAULT_RELHM_TASKS[@]}")
  else
    echo "ERR: no default TASKS_KEEP for DATASET=$DATASET; set TASKS_KEEP_CSV." >&2
    exit 2
  fi
  TASKS_KEEP_CSV=$(IFS=,; echo "${TASKS_KEEP_ARR[*]}")
fi

# Sanity: holdout task must NOT appear in TASKS_KEEP_CSV.
if echo "$TASKS_KEEP_CSV" | grep -q "${DATASET}\.${HOLDOUT}"; then
  echo "ERR: HOLDOUT task ${DATASET}.${HOLDOUT} is in TASKS_KEEP_CSV; remove it." >&2
  exit 2
fi

CACHE="${CACHE_DIR:-$HOME/.cache/relbench_examples}"
TF_STORE="$CACHE/tf_store"
SEED="${SEED:-0}"
K="${K:-64}"
BATCH="${BATCH:-128}"
CHANNELS="${CHANNELS:-128}"
HEADS="${HEADS:-4}"
CENTROIDS="${CENTROIDS:-512}"
MAX_ROWS_TRAIN="${MAX_ROWS_TRAIN:-2000}"
OUT_DIR_BASE="${OUT_DIR:-results/holdout_task_dev}"
RUN_DIR="$OUT_DIR_BASE/${DATASET}_${HOLDOUT}"
PRETRAIN_DIR="$RUN_DIR/pretrain"
EMB_DIR="$RUN_DIR/embeddings"
FT_DIR="$RUN_DIR/finetune_head"
TABPFN_DIR="$RUN_DIR/tabpfn"

mkdir -p "$PRETRAIN_DIR" "$EMB_DIR" "$FT_DIR" "$TABPFN_DIR"

export WANDB_MODE=offline
export WANDB_SILENT=true

echo "=============================================================="
echo "Phase-4 holdout-task: DATASET=$DATASET HOLDOUT=$HOLDOUT"
echo "  TASKS_KEEP: $TASKS_KEEP_CSV"
echo "  EPOCHS=$EPOCHS  MAX_STEPS=$MAX_STEPS  SEED=$SEED"
echo "  RUN_DIR: $RUN_DIR"
echo "=============================================================="

# ----------------- 1. TF store (one-time) -----------------
if [ ! -f "$TF_STORE/$DATASET/.done" ]; then
  echo
  echo "=== [1/4] Building TF store for $DATASET ==="
  python3 tools/build_tf_store.py --dataset "$DATASET" --out_dir "$TF_STORE/$DATASET"
  touch "$TF_STORE/$DATASET/.done"
else
  echo "[1/4] TF store cached at $TF_STORE/$DATASET"
fi

# ----------------- 2. Pretrain on TASKS_KEEP -----------------
PRETRAIN_LOG="$PRETRAIN_DIR/run.log"
if [ -f "$PRETRAIN_DIR/multi_task/best_full.pt" ]; then
  echo "[2/4] Pretrain artifacts cached at $PRETRAIN_DIR/multi_task/"
else
  echo
  echo "=== [2/4] Pretraining on $(echo "$TASKS_KEEP_CSV" | tr ',' '\n' | wc -l) tasks ==="
  echo "  log: $PRETRAIN_LOG"
  torchrun --nproc_per_node 1 main_node_ddp.py \
    --tasks "$TASKS_KEEP_CSV" \
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
    --seed "$SEED" \
    --out_dir "$PRETRAIN_DIR" \
    --run_name "holdout_${HOLDOUT}_pretrain" \
    > "$PRETRAIN_LOG" 2>&1
fi

# Locate the four artifacts written by PR 2.1's _save_best_checkpoint.
META="$PRETRAIN_DIR/multi_task/backbone_meta.json"
WEIGHTS="$PRETRAIN_DIR/multi_task/best_backbone.pt"
SCHEMA="$PRETRAIN_DIR/multi_task/backbone_schema.pt"
for f in "$META" "$WEIGHTS" "$SCHEMA"; do
  if [ ! -f "$f" ]; then
    echo "ERR: pretrain didn't write expected artifact: $f" >&2
    echo "     check $PRETRAIN_LOG" >&2
    exit 3
  fi
done

# ----------------- 3. Extract embeddings on held-out task -----------------
EMB_LOG="$EMB_DIR/extract.log"
if [ -f "$EMB_DIR/train.pt" ] && [ -f "$EMB_DIR/val.pt" ] && [ -f "$EMB_DIR/test.pt" ]; then
  echo "[3/4] Embeddings cached at $EMB_DIR/"
else
  echo
  echo "=== [3/4] Extracting embeddings on $DATASET.$HOLDOUT ==="
  echo "  log: $EMB_LOG"
  python3 -m tools.extract_embeddings \
    --backbone_meta "$META" \
    --backbone_weights "$WEIGHTS" \
    --backbone_schema "$SCHEMA" \
    --dataset "$DATASET" --task "$HOLDOUT" \
    --split all \
    --num_neighbors "$K" --batch_size "$BATCH" \
    --num_workers 0 \
    --cache_dir "$CACHE" \
    --out_dir "$EMB_DIR" \
    > "$EMB_LOG" 2>&1
fi

# ----------------- 4a. Finetune head -----------------
FT_LOG="$FT_DIR/finetune.log"
echo
echo "=== [4a] Finetuning linear head on extracted embeddings ==="
echo "  log: $FT_LOG"
python3 -m tools.finetune_head \
  --embeddings_dir "$EMB_DIR" \
  --dataset "$DATASET" --task "$HOLDOUT" \
  --head linear --epochs 50 --lr 1e-3 \
  --out "$FT_DIR/finetuned.pt" \
  --device cpu \
  > "$FT_LOG" 2>&1

# ----------------- 4b. TabPFN post-hoc -----------------
TABPFN_LOG="$TABPFN_DIR/tabpfn.log"
echo "=== [4b] TabPFN post-hoc evaluation ==="
echo "  log: $TABPFN_LOG"
PROJECTOR="${PROJECTOR:-pca64}"
if ! python3 -m tools.tabpfn_eval \
    --embeddings_dir "$EMB_DIR" \
    --dataset "$DATASET" --task "$HOLDOUT" \
    --projector "$PROJECTOR" \
    --out "$TABPFN_DIR/tabpfn.json" \
    > "$TABPFN_LOG" 2>&1; then
  echo "  WARN: tabpfn_eval failed (likely tabpfn not installed); see $TABPFN_LOG"
fi

# ----------------- 5. Aggregate -----------------
SUMMARY="$RUN_DIR/summary.json"
python3 - <<PY
import json, os
out = {
    "dataset": "$DATASET",
    "holdout": "$HOLDOUT",
    "pretrained_tasks": "$TASKS_KEEP_CSV".split(","),
    "epochs": int($EPOCHS),
    "seed": int($SEED),
}
ft_path = "$FT_DIR/finetuned.pt"
if os.path.exists(ft_path):
    import torch
    ft = torch.load(ft_path, map_location="cpu", weights_only=False)
    out["finetune_head"] = {
        "head_kind": ft.get("head_kind"),
        "best_epoch": ft.get("best_epoch"),
        "best_val_loss": ft.get("best_val_loss"),
        "test_metrics": ft.get("test_metrics"),
    }
tp_path = "$TABPFN_DIR/tabpfn.json"
if os.path.exists(tp_path):
    with open(tp_path) as f:
        out["tabpfn"] = json.load(f)
with open("$SUMMARY", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "=============================================================="
echo "Phase-4 holdout-task done. Summary: $SUMMARY"
echo "=============================================================="
