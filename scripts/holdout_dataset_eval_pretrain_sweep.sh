#!/usr/bin/env bash
# Backbone-variance sweep on top of holdout_dataset_eval.sh.
#
# Pretrain N independent backbones (one per seed in PRETRAIN_SEEDS),
# adopt to TARGET from each, and aggregate test metrics across the
# pretrain seeds. Captures BACKBONE variance (random init + sampler
# RNG inside the multi-task pretrain) -- the regular launcher's
# SEEDS knob only varies the adoption-side RNG with a single shared
# backbone.
#
# This is significantly more expensive than the in-launcher SEEDS
# sweep: every PRETRAIN_SEEDS entry triggers a full multi-task DDP
# pretrain (hours on p4d). Use sparingly and prefer 3 seeds unless
# you really need a tighter SD.
#
# Per-trial seed isolation -- every RNG that touches the trial is
# derived from the pretrain seed:
#   SEED=$PSEED                forwarded to main_node_ddp.py --seed
#                              (model init, DataLoader shuffle,
#                              multi-task sampler).
#   PYTHONHASHSEED=$PSEED      pins CPython's hash() for strings
#                              and tuples-with-strings, which is
#                              what gfm_data/sampler.py uses to
#                              derive the per-row seed_val and to
#                              order set->list conversions.
#                              Without this, sampling RNG is
#                              uncontrolled noise across trials.
#   SHARDS_SUBDIR=pretrain_seedN
#                              forces each trial to build its own
#                              precomputed shards under
#                              $CACHE/shards[_full]/pretrain_seedN/
#                              instead of reusing the first trial's
#                              cache. tf_store + materialization
#                              (deterministic) stay shared.
#
# Net effect: each trial's pretrain sees a different per-row
# neighbor list AND different model init; the resulting backbone-
# variance number captures both jointly.
#
# Each pretrain seed gets its own RUN_DIR / artifacts. The aggregator
# at the bottom reads each run's summary.json and rolls per-task
# test_metrics into mean +/- SD across the N pretrain seeds, written
# to <OUT_DIR>/aggregate.json.
#
# Defaults are p4d-ready and match the LOO defaults of the inner
# launcher. Override SOURCE / TARGET / SEEDS / RUN_TABPFN at will.
#
# Usage:
#   bash scripts/holdout_dataset_eval_pretrain_sweep.sh
#
# Knobs (env vars; defaults shown):
#   PRETRAIN_SEEDS  "0 1 2"    space-separated list of pretrain seeds
#   SOURCE          "rel-f1 rel-event"   space-separated source list
#   TARGET          rel-arxiv  held-out adoption dataset
#   ADOPT_SEEDS     "0"        adoption-side seeds per pretrain
#                              (default 1; total = #pretrain x #adopt
#                              experiments; bump if you also want
#                              adoption-side variance per backbone)
#   EPOCHS          10
#   STEPS_PER_TASK  500
#   NPROC           8
#   SHARD_WORKERS   10
#   RUN_TABPFN      0          skip TabPFN by default (it's slow and
#                              we usually want backbone-variance, not
#                              TabPFN-subsample variance)
#   FT_HEAD         mlp2       2-layer MLP head (override =linear for
#                              the legacy linear-probe ablation)
#   OUT_DIR         results/holdout_dataset_eval_pretrain_sweep
#   INNER           scripts/holdout_dataset_eval_clean.sh
#                              The inner launcher to wrap. Defaults
#                              to the clean variant (35 tasks across
#                              9 datasets, drops the 5 RelBench v2
#                              tasks whose supervised GNN baseline
#                              is at-or-below random per the paper).
#                              Set INNER=scripts/holdout_dataset_eval.sh
#                              for the full 40-task setup.
#
# Recommended invocation (rel-f1 + rel-event -> rel-arxiv, no
# TabPFN, single adoption seed per pretrain seed):
#
#   NPROC=8 SHARD_WORKERS=10 \
#     PRETRAIN_SEEDS="0 1 2" \
#     SOURCE="rel-f1 rel-event" TARGET=rel-arxiv \
#     ADOPT_SEEDS="0" RUN_TABPFN=0 \
#     EPOCHS=10 STEPS_PER_TASK=500 \
#     bash scripts/holdout_dataset_eval_pretrain_sweep.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-0 1 2}"
SOURCE="${SOURCE:-rel-f1 rel-event}"
TARGET="${TARGET:-rel-arxiv}"
ADOPT_SEEDS="${ADOPT_SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
STEPS_PER_TASK="${STEPS_PER_TASK:-500}"
NPROC="${NPROC:-8}"
SHARD_WORKERS="${SHARD_WORKERS:-10}"
RUN_TABPFN="${RUN_TABPFN:-0}"
FT_HEAD="${FT_HEAD:-mlp2}"
INNER="${INNER:-scripts/holdout_dataset_eval_clean.sh}"
OUT_DIR_BASE="${OUT_DIR:-results/holdout_dataset_eval_pretrain_sweep}"
# Force-rebuild the per-trial precomputed shard tree under
# $CACHE/shards[_full]/pretrain_seed<N>/ before each trial. Default
# 0 (fast iteration): pretrain_p4d.sh's .done sentinel reuses
# shards from a prior invocation, which is what you want when only
# the training code changed. Set FORCE_REBUILD_SHARDS=1 when the
# sampler / seed_time / truncation policy moved between runs and
# you need neighbor lists rebuilt to match the new code.
FORCE_REBUILD_SHARDS="${FORCE_REBUILD_SHARDS:-0}"
CACHE_LOCAL="${CACHE_DIR:-$HOME/.cache/relbench_examples}"

if [ ! -x "$REPO_ROOT/$INNER" ] && [ ! -f "$REPO_ROOT/$INNER" ]; then
  echo "ERR: inner launcher '$INNER' not found at $REPO_ROOT/$INNER" >&2
  exit 2
fi

# Slug for the run directory (spaces don't survive a path component).
SOURCE_SLUG=$(printf '%s' "$SOURCE" | tr ' ' '+')
SWEEP_DIR="$OUT_DIR_BASE/${SOURCE_SLUG}_to_${TARGET}"
mkdir -p "$SWEEP_DIR"

echo "=================================================================="
echo "Backbone-variance sweep"
echo "  inner launcher : $INNER"
echo "  PRETRAIN_SEEDS : [$PRETRAIN_SEEDS]"
echo "  SOURCE         : $SOURCE"
echo "  TARGET         : $TARGET"
echo "  ADOPT_SEEDS    : [$ADOPT_SEEDS]"
echo "  EPOCHS         : $EPOCHS  STEPS_PER_TASK: $STEPS_PER_TASK"
echo "  NPROC          : $NPROC   SHARD_WORKERS : $SHARD_WORKERS"
echo "  RUN_TABPFN     : $RUN_TABPFN  FT_HEAD: $FT_HEAD"
echo "  out            : $SWEEP_DIR"
echo "=================================================================="

# ---- Sweep loop ----
PSEED_DIRS=()
for PSEED in $PRETRAIN_SEEDS; do
  PSEED_DIR="$SWEEP_DIR/pretrain_seed${PSEED}"
  mkdir -p "$PSEED_DIR"
  PSEED_DIRS+=("$PSEED_DIR")

  echo
  echo "------------------------------------------------------------------"
  echo "[$(date)] PRETRAIN_SEED=$PSEED  ->  $PSEED_DIR"
  echo "------------------------------------------------------------------"

  # Defensive shard wipe so a prior invocation in this container
  # cannot silently feed stale neighbor lists into this trial.
  if [ "$FORCE_REBUILD_SHARDS" = "1" ]; then
    for _shroot in "$CACHE_LOCAL/shards" "$CACHE_LOCAL/shards_full"; do
      _shpath="$_shroot/pretrain_seed${PSEED}"
      if [ -d "$_shpath" ]; then
        echo "[$(date)] [force-rebuild] wiping $_shpath"
        rm -rf "$_shpath"
      fi
    done
  fi

  # Per-trial seed plumbing -- want every RNG that touches this run
  # to be deterministically derived from PSEED, AND want each trial
  # to actually rebuild its neighbor list from scratch:
  #
  #   SEED=$PSEED            pretrain --seed (model init, train
  #                          DataLoader shuffle, multi-task sampler).
  #   SEEDS=$ADOPT_SEEDS     adoption-side seeds (passed straight to
  #                          extract_embeddings / finetune_head /
  #                          tabpfn_eval --seed inside the inner
  #                          launcher's adoption Phase A loop).
  #   PYTHONHASHSEED=$PSEED  controls hash() of strings + tuples in
  #                          CPython, which feeds into:
  #                            (a) the per-row seed_val =
  #                                hash((seed_node_type, idx, t, K))
  #                                inside gfm_data/sampler.py and
  #                                utils.py, and
  #                            (b) the iteration order of any
  #                                set->list conversion the sampler
  #                                does on string-keyed neighbor
  #                                sets.
  #                          Without this set, every Python
  #                          subprocess gets a random PYTHONHASHSEED
  #                          and the "different seed for sampling"
  #                          control is uncontrolled noise instead.
  #   SHARDS_SUBDIR=...      pretrain_p4d.sh namespaces its
  #                          precomputed shards under
  #                          $CACHE/shards[_full]/<SHARDS_SUBDIR>/
  #                          so each trial truly rebuilds the
  #                          per-row neighbor lists rather than
  #                          reusing the first-trial cache.
  #                          tf_store + materialization (which are
  #                          deterministic from the raw data) stay
  #                          shared across trials.
  PYTHONHASHSEED="$PSEED" \
  SEED="$PSEED" \
  SEEDS="$ADOPT_SEEDS" \
  SHARDS_SUBDIR="pretrain_seed${PSEED}" \
  SOURCE="$SOURCE" TARGET="$TARGET" \
  EPOCHS="$EPOCHS" STEPS_PER_TASK="$STEPS_PER_TASK" \
  NPROC="$NPROC" SHARD_WORKERS="$SHARD_WORKERS" \
  RUN_TABPFN="$RUN_TABPFN" FT_HEAD="$FT_HEAD" \
  OUT_DIR="$PSEED_DIR" \
  bash "$REPO_ROOT/$INNER"
done

# ---- Aggregate across pretrain seeds ----
AGG="$SWEEP_DIR/aggregate.json"
PSEED_DIRS_STR=$(printf '%s\n' "${PSEED_DIRS[@]}")
PRETRAIN_SEEDS_STR="$PRETRAIN_SEEDS"

python3 - <<PY
import json, os, statistics

inner_dir_pattern = None  # let the script discover it dynamically
pseed_dirs = """$PSEED_DIRS_STR""".strip().split("\n")
pretrain_seeds = "$PRETRAIN_SEEDS_STR".split()

def _mean_sd(values):
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

# Each per-seed dir contains <src>_to_<target>/summary.json (the
# inner launcher's aggregate). Discover that subdir.
def _find_summary(pseed_dir):
    for child in sorted(os.listdir(pseed_dir)):
        cand = os.path.join(pseed_dir, child, "summary.json")
        if os.path.exists(cand):
            return cand
    return None

# Collect: per_task -> per_metric -> list of (pretrain_seed, mean of
# adopter's seeds within that pretrain).
per_task = {}
adopter_keys = ("finetune_head_summary", "tabpfn_summary")

for pseed, pdir in zip(pretrain_seeds, pseed_dirs):
    summary = _find_summary(pdir)
    if summary is None:
        print(f"# WARN: no summary.json under {pdir}; skipping pretrain_seed={pseed}")
        continue
    with open(summary) as f:
        data = json.load(f)
    for task_name, entry in data.get("per_task", {}).items():
        per_task.setdefault(task_name, {})
        for adopter in adopter_keys:
            metrics = entry.get(adopter) or {}
            tgt = per_task[task_name].setdefault(adopter, {})
            for metric_name, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue
                tgt.setdefault(metric_name, []).append({
                    "pretrain_seed": pseed,
                    "value": stats.get("mean"),
                    "adopt_n": stats.get("n"),
                })

# Roll per-task metrics into mean +/- SD across pretrain seeds.
out = {
    "source": "$SOURCE",
    "target": "$TARGET",
    "pretrain_seeds": pretrain_seeds,
    "adopt_seeds": "$ADOPT_SEEDS".split(),
    "per_task": {},
}
for task_name, adopter_data in per_task.items():
    out["per_task"][task_name] = {}
    for adopter, metrics in adopter_data.items():
        rolled = {}
        for metric_name, points in metrics.items():
            stats = _mean_sd([p["value"] for p in points])
            if stats is not None:
                stats["per_pretrain_seed"] = points
                rolled[metric_name] = stats
        out["per_task"][task_name][adopter] = rolled

with open("$AGG", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY

echo
echo "=================================================================="
echo "Backbone-variance sweep done."
echo "  per-pretrain artifacts: $SWEEP_DIR/pretrain_seed*/"
echo "  aggregate              : $AGG"
echo "=================================================================="
