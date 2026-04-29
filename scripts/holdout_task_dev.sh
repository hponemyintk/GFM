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
# Defaults (paper-safe subset; see docs/truncated_graph_caveat.md):
#   DATASETS    "rel-f1 rel-event"
#   HOLDOUTS    "rel-f1:driver-top3 rel-event:user-attendance"
#   PRETRAIN    union of all-but-holdout tasks across DATASETS
#                = 2 rel-f1 + 2 rel-event = 4 tasks
#
# Holdouts are deliberately mixed-metric (one binary classification +
# one regression) to test the frozen-backbone GFM claim across head
# types:
#   * rel-f1 / driver-top3       -- binary, AUROC. Paper-benchmarked
#                                   in expts/run-large-base-experiments
#                                   and expts/run-encoder-ablation.
#   * rel-event / user-attendance -- regression, MAE. Paper-benchmarked
#                                   (Table 1a, MAE 0.2502) and in
#                                   expts/run-hyperparam-sweep-small.
#
# Pretrain composition:
#   * rel-f1: driver-position (regression), driver-dnf (binary)
#   * rel-event: user-repeat (binary), user-ignore (binary)
# = 1 regression + 3 binary tasks. The regression-vs-binary mix in
# pretrain (driver-position) plus the regression holdout
# (user-attendance) lets us evaluate transfer to a held-out
# regression head specifically.
#
# rel-hm is NOT in defaults but stays available via DATASETS override.
# users-birthyear / event_interest-* / results-position /
# qualifying-position / transactions-price (autocomplete tasks with
# growing seeds) are excluded from defaults and require --full_graph
# to opt in -- see docs/truncated_graph_caveat.md.
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
#   bash scripts/holdout_task_dev.sh [EPOCHS] [MAX_STEPS]
#
#     EPOCHS        default 5
#     MAX_STEPS     default 300
#
# ---------------- Example invocations (copy-paste) ----------------
#
# AWS p4d.24xlarge (8 x A100, paper-config, default rel-f1+rel-event):
#
#   NPROC=8 EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_task_dev.sh
#
# AWS p4d quick shake-out (~1 hour wall, sanity that the full
# pipeline runs end-to-end):
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
# Laptop (single GPU, smaller config -- override to drop rel-event,
# whose materialization peak ~25 GB OOMs a 27 GB box):
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
MAX_STEPS="${MAX_STEPS:-${2:-300}}"
DATASETS="${DATASETS:-rel-f1 rel-event}"
# Per-dataset holdout map: "<dataset>:<task> <dataset>:<task> ..."
# Tasks cited in RelGT paper expts (see header docstring).
# Default holdouts: one binary (rel-f1:driver-top3) + one regression
# (rel-event:user-attendance), to evaluate frozen-backbone transfer
# to held-out heads of both kinds.
HOLDOUTS="${HOLDOUTS:-rel-f1:driver-top3 rel-event:user-attendance}"

# Default per-dataset full task lists. The launcher subtracts the
# HOLDOUTS map from these to produce the pretrain CSV. Each list is
# restricted to the paper-benchmarked tasks whose val/test seeds
# index within the truncated CSR adjacency built under the RelGT
# paper guardrail (upto_test_timestamp=True). Empirically all tasks
# below have val OOB = 0 and test OOB = 0 against the truncated
# entity table -- safe out of the box.
#
# Tasks excluded from defaults (autocomplete / growing-seed):
#   * rel-f1.results-position, rel-f1.qualifying-position --
#     test seeds 100% reference rows past train_cutoff.
#   * rel-hm.transactions-price -- growing transaction seeds.
#   * rel-event.users-birthyear, rel-event.event_interest-* --
#     val OK but test-split materialization itself OOMs in
#     RelBench's pd.date_range (separate bug from the CSR-OOB).
# To opt these in, override PRETRAIN_TASKS_CSV explicitly AND build
# shards/TF stores with --full_graph (see docs/truncated_graph_caveat.md).
#
# Sources:
#   rel-f1:    expts/run-large-base-experiments.sh + paper Table 1.
#   rel-event: expts/run-hyperparam-sweep-small-experiments.sh +
#              paper Tables 1a (user-attendance MAE 0.2502) and 1b
#              (user-repeat AUC 0.7609, user-ignore AUC 0.8157).
#   rel-hm:    expts/run-large-base-experiments.sh -- user-churn,
#              item-sales.
declare -A DEFAULT_ALL_TASKS=(
  [rel-f1]="driver-position driver-dnf driver-top3"
  [rel-event]="user-attendance user-repeat user-ignore"
  [rel-hm]="user-churn item-sales"
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
echo "  EPOCHS=$EPOCHS  MAX_STEPS=$MAX_STEPS  SEED=$SEED"
echo "  PRETRAIN_BACKEND=$PRETRAIN_BACKEND  NPROC=$NPROC"
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
  TASKS_CSV="$PRETRAIN_TASKS_CSV" \
    DATASETS="$DS_CSV" \
    NPROC="$NPROC" \
    EPOCHS="$EPOCHS" \
    MAX_STEPS="$MAX_STEPS" \
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
  echo "=== [1/4] Building TF stores ==="
  for ds in $DATASETS; do
    if [ -f "$TF_STORE/$ds/.done" ]; then
      echo "  $ds: cached at $TF_STORE/$ds"
    else
      echo "  $ds: building..."
      python3 tools/build_tf_store.py --dataset "$ds" --out_dir "$TF_STORE/$ds"
      touch "$TF_STORE/$ds/.done"
    fi
  done

  N_TASKS=$(echo "$PRETRAIN_TASKS_CSV" | tr ',' '\n' | wc -l)
  echo
  echo "=== [2/4] Pretraining on $N_TASKS tasks (laptop streaming) ==="
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
    --max_steps_per_epoch "$MAX_STEPS" \
    --num_workers 0 \
    --lr 1e-4 --warmup_steps 200 \
    --loss_balance "${LOSS_BALANCE:-none}" \
    --seed "$SEED" \
    --out_dir "$PRETRAIN_DIR" \
    --run_name "${RUN_NAME:-phase4_multi_holdout_pretrain}" \
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
