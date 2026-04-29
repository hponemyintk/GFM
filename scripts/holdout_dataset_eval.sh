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
# Defaults:
#   SOURCE       rel-f1        (~50 MB, fits laptop)
#   TARGET       rel-hm        (~600 MB, fits laptop)
#   SOURCE_TASKS all 6 rel-f1 tasks (multi-task pretrain)
#   TARGET_TASKS all 3 rel-hm tasks (extract + finetune + tabpfn each)
#
# rel-event won't fit the laptop; the AWS p4d run uses
# scripts/pretrain_p4d.sh for source, then this script with
# SOURCE=rel-event TARGET=<other> on the box.
#
# Usage:
#   bash scripts/holdout_dataset_eval.sh [EPOCHS] [MAX_STEPS]
#
# Output layout: results/holdout_dataset_eval/<src>_to_<tgt>/

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-${1:-5}}"
MAX_STEPS="${MAX_STEPS:-${2:-300}}"
SOURCE="${SOURCE:-rel-f1}"
TARGET="${TARGET:-rel-hm}"

if [ "$SOURCE" = "$TARGET" ]; then
  echo "ERR: SOURCE and TARGET must differ (Phase-5 is cross-dataset). " \
       "For same-dataset holdout-task, use scripts/holdout_task_dev.sh." >&2
  exit 2
fi

# Source pretraining task lists. All tasks of the source dataset.
DEFAULT_RELF1_ALL=(
  "rel-f1.driver-position:1.0"
  "rel-f1.driver-dnf:1.0"
  "rel-f1.driver-top3:1.0"
  "rel-f1.driver-circuit-compete:1.0"
  "rel-f1.results-position:1.0"
  "rel-f1.qualifying-position:1.0"
)
DEFAULT_RELHM_ALL=(
  "rel-hm.user-churn:1.0"
  "rel-hm.item-sales:1.0"
  "rel-hm.transactions-price:1.0"
)

if [ -z "${SOURCE_TASKS_CSV:-}" ]; then
  if [ "$SOURCE" = "rel-f1" ]; then
    SRC_ARR=("${DEFAULT_RELF1_ALL[@]}")
  elif [ "$SOURCE" = "rel-hm" ]; then
    SRC_ARR=("${DEFAULT_RELHM_ALL[@]}")
  else
    echo "ERR: no default SOURCE_TASKS for SOURCE=$SOURCE; set SOURCE_TASKS_CSV." >&2
    exit 2
  fi
  SOURCE_TASKS_CSV=$(IFS=,; echo "${SRC_ARR[*]}")
fi

if [ -z "${TARGET_TASKS_CSV:-}" ]; then
  if [ "$TARGET" = "rel-hm" ]; then
    TGT_ARR=("${DEFAULT_RELHM_ALL[@]}")
  elif [ "$TARGET" = "rel-f1" ]; then
    TGT_ARR=("${DEFAULT_RELF1_ALL[@]}")
  else
    echo "ERR: no default TARGET_TASKS for TARGET=$TARGET; set TARGET_TASKS_CSV." >&2
    exit 2
  fi
  TARGET_TASKS_CSV=$(IFS=,; echo "${TGT_ARR[*]}")
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
OUT_DIR_BASE="${OUT_DIR:-results/holdout_dataset_eval}"
RUN_DIR="$OUT_DIR_BASE/${SOURCE}_to_${TARGET}"
PRETRAIN_DIR="$RUN_DIR/pretrain"

mkdir -p "$PRETRAIN_DIR"

# WANDB_MODE left to the caller's environment / wandb's own default.
# WANDB_SILENT default true to avoid interleaving wandb spam with
# script stdout; override with WANDB_SILENT=false if desired.
export WANDB_SILENT="${WANDB_SILENT:-true}"

echo "=============================================================="
echo "Phase-5 cross-dataset adoption: SOURCE=$SOURCE -> TARGET=$TARGET"
echo "  SOURCE_TASKS: $SOURCE_TASKS_CSV"
echo "  TARGET_TASKS: $TARGET_TASKS_CSV"
echo "  EPOCHS=$EPOCHS  MAX_STEPS=$MAX_STEPS  SEED=$SEED"
echo "  RUN_DIR: $RUN_DIR"
echo "=============================================================="

# ----------------- 1. TF stores for SOURCE + TARGET -----------------
for ds in "$SOURCE" "$TARGET"; do
  if [ ! -f "$TF_STORE/$ds/.done" ]; then
    echo
    echo "=== Building TF store for $ds ==="
    python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds"
    touch "$TF_STORE/$ds/.done"
  else
    echo "[setup] TF store cached at $TF_STORE/$ds"
  fi
done

# ----------------- 2. Pretrain on SOURCE -----------------
PRETRAIN_LOG="$PRETRAIN_DIR/run.log"
META="$PRETRAIN_DIR/multi_task/backbone_meta.json"
WEIGHTS="$PRETRAIN_DIR/multi_task/best_backbone.pt"
SCHEMA="$PRETRAIN_DIR/multi_task/backbone_schema.pt"

if [ -f "$WEIGHTS" ] && [ -f "$META" ] && [ -f "$SCHEMA" ]; then
  echo "[pretrain] artifacts cached at $PRETRAIN_DIR/multi_task/"
else
  echo
  echo "=== Pretraining on SOURCE=$SOURCE ($SOURCE_TASKS_CSV) ==="
  echo "  log: $PRETRAIN_LOG"
  torchrun --nproc_per_node 1 main_node_ddp.py \
    --tasks "$SOURCE_TASKS_CSV" \
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
    --run_name "phase5_${SOURCE}_to_${TARGET}_pretrain" \
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
PROJECTOR="${PROJECTOR:-pca64}"

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
      --out_dir "$EMB_DIR" \
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
    "source": "$SOURCE",
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
