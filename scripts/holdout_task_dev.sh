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
# Defaults (full-scale, AWS p4d-targeted):
#   DATASETS    "rel-f1 rel-event rel-hm"
#   HOLDOUTS    "rel-f1:driver-top3 rel-event:user-attendance rel-hm:user-churn"
#   PRETRAIN    union of all-but-holdout tasks across DATASETS
#                = 5 rel-f1 + 5 rel-event + 2 rel-hm = 12 tasks
#
# Holdout task choices match the RelGT paper's benchmark task list.
# Metric mix is intentional (paper-aligned, not AUROC-consistent):
#   * rel-f1 / driver-top3      -- binary, AUROC. Cited in
#                                  expts/run-large-base-experiments
#                                  and expts/run-encoder-ablation.
#   * rel-event / user-attendance -- REGRESSION, MAE. Cited in
#                                  expts/run-encoder-ablation
#                                  (paper benchmarks user-attendance,
#                                  user-repeat, user-ignore on
#                                  rel-event; user-attendance is the
#                                  regression of the three).
#   * rel-hm / user-churn       -- binary, AUROC. Cited in
#                                  expts/run-large-base-experiments
#                                  and the paper's main results table.
#
# Per-holdout metric will appear in summary.json under the relevant
# RelBench-evaluated key ('roc_auc' for binary, 'mae' for
# regression). Cross-holdout aggregation is left to the consumer.
#
# Memory profile:
#   * rel-f1 materialization:    ~50 MB (laptop-fine)
#   * rel-hm materialization:    ~600 MB (laptop-fine)
#   * rel-event materialization: ~25 GB peak RSS (laptop OOMs at 27 GB)
#
# To run on the laptop: override DATASETS to drop rel-event:
#   DATASETS="rel-f1 rel-hm" \
#   HOLDOUTS="rel-f1:driver-top3 rel-hm:user-churn" \
#   bash scripts/holdout_task_dev.sh
#
# Full 3-dataset run targets the 8x A100 / 1 TB RAM box (same as
# scripts/pretrain_p4d.sh).
#
# Usage:
#   bash scripts/holdout_task_dev.sh [EPOCHS] [MAX_STEPS]
#
#     EPOCHS        default 5
#     MAX_STEPS     default 300
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
MAX_STEPS="${MAX_STEPS:-${2:-300}}"
DATASETS="${DATASETS:-rel-f1 rel-event rel-hm}"
# Per-dataset holdout map: "<dataset>:<task> <dataset>:<task> ..."
# Tasks cited in RelGT paper expts (see header docstring).
HOLDOUTS="${HOLDOUTS:-rel-f1:driver-top3 rel-event:user-attendance rel-hm:user-churn}"

# Default per-dataset full task lists. The launcher subtracts the
# HOLDOUTS map from these to produce the pretrain CSV. Sources:
#   rel-f1:    relbench.tasks.get_task_names('rel-f1')
#   rel-event: relbench.tasks.get_task_names('rel-event')  (matches
#              scripts/pretrain_alltasks.sh task list verbatim)
#   rel-hm:    relbench.tasks.get_task_names('rel-hm')
#              (user-item-purchase is recommendation, omitted)
declare -A DEFAULT_ALL_TASKS=(
  [rel-f1]="driver-position driver-dnf driver-top3 driver-circuit-compete results-position qualifying-position"
  [rel-event]="user-attendance user-repeat user-ignore event_interest-interested event_interest-not_interested users-birthyear"
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
TF_STORE="$CACHE/tf_store"
SEED="${SEED:-0}"
K="${K:-64}"
BATCH="${BATCH:-128}"
CHANNELS="${CHANNELS:-128}"
HEADS="${HEADS:-4}"
CENTROIDS="${CENTROIDS:-512}"
MAX_ROWS_TRAIN="${MAX_ROWS_TRAIN:-2000}"
OUT_DIR_BASE="${OUT_DIR:-results/holdout_task_dev}"
PRETRAIN_DIR="$OUT_DIR_BASE/pretrain"

mkdir -p "$PRETRAIN_DIR"

export WANDB_MODE=offline
export WANDB_SILENT=true

echo "=============================================================="
echo "Phase-4 multi-dataset holdout-task"
echo "  DATASETS:       $DATASETS"
echo "  HOLDOUTS:       $HOLDOUTS"
echo "  PRETRAIN tasks: $PRETRAIN_TASKS_CSV"
echo "  EPOCHS=$EPOCHS  MAX_STEPS=$MAX_STEPS  SEED=$SEED"
echo "  OUT_DIR:        $OUT_DIR_BASE"
echo "=============================================================="

# ----------------- 1. TF stores (one-time per dataset) -----------------
echo
echo "=== [1/4] TF stores ==="
for ds in $DATASETS; do
  if [ -f "$TF_STORE/$ds/.done" ]; then
    echo "  $ds: cached at $TF_STORE/$ds"
  else
    echo "  $ds: building..."
    python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds"
    touch "$TF_STORE/$ds/.done"
  fi
done

# ----------------- 2. Multi-dataset pretrain on all-but-holdout -----------------
PRETRAIN_LOG="$PRETRAIN_DIR/run.log"
META="$PRETRAIN_DIR/multi_task/backbone_meta.json"
WEIGHTS="$PRETRAIN_DIR/multi_task/best_backbone.pt"
SCHEMA="$PRETRAIN_DIR/multi_task/backbone_schema.pt"

if [ -f "$META" ] && [ -f "$WEIGHTS" ] && [ -f "$SCHEMA" ]; then
  echo
  echo "=== [2/4] Pretrain artifacts cached at $PRETRAIN_DIR/multi_task/ ==="
else
  N_TASKS=$(echo "$PRETRAIN_TASKS_CSV" | tr ',' '\n' | wc -l)
  echo
  echo "=== [2/4] Pretraining on $N_TASKS tasks across $(echo $DATASETS | wc -w) datasets ==="
  echo "  log: $PRETRAIN_LOG"
  torchrun --nproc_per_node 1 main_node_ddp.py \
    --tasks "$PRETRAIN_TASKS_CSV" \
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
    --run_name "phase4_multi_holdout_pretrain" \
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
PROJECTOR="${PROJECTOR:-pca64}"
HOLDOUT_DIRS=()

echo
echo "=== [3/4] Per-holdout adoption (frozen backbone) ==="

for ds in $DATASETS; do
  holdout="${HOLDOUT_OF[$ds]:-}"
  if [ -z "$holdout" ]; then
    continue
  fi
  TASK_DIR="$OUT_DIR_BASE/${ds}.${holdout}"
  EMB_DIR="$TASK_DIR/embeddings"
  FT_DIR="$TASK_DIR/finetune_head"
  TABPFN_DIR="$TASK_DIR/tabpfn"
  mkdir -p "$EMB_DIR" "$FT_DIR" "$TABPFN_DIR"
  HOLDOUT_DIRS+=("$TASK_DIR")

  echo
  echo "--- ${ds}.${holdout} ---"

  # Extract: same dataset as pretrain, so register_dataset has
  # already populated the encoder with this dataset's prefixed
  # types -- no --register_new_dataset needed.
  EMB_LOG="$EMB_DIR/extract.log"
  if [ -f "$EMB_DIR/test.pt" ] && [ -f "$EMB_DIR/train.pt" ] && [ -f "$EMB_DIR/val.pt" ]; then
    echo "  [extract] embeddings cached"
  else
    echo "  [extract] backbone forward(task_id=None) on ${ds}.${holdout}"
    python3 -m tools.extract_embeddings \
      --backbone_meta "$META" \
      --backbone_weights "$WEIGHTS" \
      --backbone_schema "$SCHEMA" \
      --dataset "$ds" --task "$holdout" \
      --split all \
      --num_neighbors "$K" --batch_size "$BATCH" \
      --num_workers 0 \
      --cache_dir "$CACHE" \
      --out_dir "$EMB_DIR" \
      > "$EMB_LOG" 2>&1
  fi

  # Frozen-backbone fine-tune: only the head sees gradients. This
  # is the GFM-claim test -- can a Linear(channels, 1) on top of
  # the frozen backbone hit reasonable test metric on a held-out
  # task head?
  echo "  [finetune] linear head, frozen backbone"
  python3 -m tools.finetune_head \
    --embeddings_dir "$EMB_DIR" \
    --dataset "$ds" --task "$holdout" \
    --head linear --epochs 50 --lr 1e-3 \
    --out "$FT_DIR/finetuned.pt" \
    --device cpu \
    > "$FT_DIR/finetune.log" 2>&1 || \
      echo "    WARN: finetune_head failed; see $FT_DIR/finetune.log"

  # Pure post-hoc TabPFN -- no gradient anywhere.
  echo "  [tabpfn] post-hoc with $PROJECTOR projector"
  if ! python3 -m tools.tabpfn_eval \
      --embeddings_dir "$EMB_DIR" \
      --dataset "$ds" --task "$holdout" \
      --projector "$PROJECTOR" \
      --out "$TABPFN_DIR/tabpfn.json" \
      > "$TABPFN_DIR/tabpfn.log" 2>&1; then
    echo "    WARN: tabpfn_eval failed (likely tabpfn not installed); see $TABPFN_DIR/tabpfn.log"
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
