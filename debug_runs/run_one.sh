#!/usr/bin/env bash
# Usage: run_one.sh <worktree_dir> <run_name> <seed> <epochs> [extra_args...]
set -e

WORKTREE="$1"
RUN_NAME="$2"
SEED="$3"
EPOCHS="$4"
shift 4

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="rel-f1"
TASK="driver-top3"
echo "[run_one.sh] DATASET=$DATASET TASK=$TASK"

# Clear precomputed K-subgraph sampling cache (random per seed) but preserve
# the deterministic scope cache (scope_*/) used by PASS sampler.
PRECOMPUTE_DIR="$HOME/.cache/relbench_examples/precomputed/${DATASET}/${TASK}"
if [ -d "$PRECOMPUTE_DIR" ]; then
    echo "[run_one.sh] clearing K-subgraph caches (preserving scope_*)..."
    for d in "$PRECOMPUTE_DIR"/*/; do
        case "$(basename "$d")" in
            scope_*) echo "[run_one.sh]   keeping $d" ;;
            *)       echo "[run_one.sh]   removing $d"; rm -rf "$d" ;;
        esac
    done
fi

OUT_DIR="/home/jedi/research_repos/GFM/debug_runs/results/${RUN_NAME}"
LOG="/home/jedi/research_repos/GFM/debug_runs/logs/${RUN_NAME}.log"
mkdir -p "$OUT_DIR" /home/jedi/research_repos/GFM/debug_runs/logs

cd "$WORKTREE"

MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    main_node_ddp.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --precompute \
    --seed "$SEED" \
    --batch_size 32 \
    --num_neighbors 50 \
    --num_layers 4 \
    --channels 512 \
    --max_steps_per_epoch 1000 \
    --num_workers 2 \
    --epochs "$EPOCHS" \
    --lr 0.0001 \
    --warmup_steps 100 \
    --ff_dropout 0.3 \
    --attn_dropout 0.3 \
    --run_name "$RUN_NAME" \
    --out_dir "$OUT_DIR" \
    "$@" \
    2>&1 | tee "$LOG"
