#!/usr/bin/env bash
# Distilled-sampler experiment on rel-event/user-repeat.
# Hyperparameters match expts/run-large-base-experiments.sh except batch_size=32.
# Runs the three phases sequentially on a single GPU.
#
# Usage:
#   bash expts/run-distilled-sampler-user-repeat.sh [GPU_ID]
set -euo pipefail

GPU_ID="${1:-0}"
MASTER_PORT="${MASTER_PORT:-29300}"

DATASET="rel-event"
TASK="user-repeat"
SEED=0
BATCH_SIZE=32
NUM_NEIGHBORS=300
NUM_LAYERS=4
GT_CONV_TYPE="full"
ABLATE="none"
CHANNELS=512
MAX_STEPS_PER_EPOCH=500
NUM_WORKERS=8
EPOCHS=10
LR=0.0001
WARMUP=10
DROPOUT=0.3

SAMPLE_SCOPE=3000
SAMPLE_TEMP=1.0

RUN_NAME="distilled-sampler-${DATASET}-${TASK}-l${NUM_LAYERS}-${CHANNELS}-BS${BATCH_SIZE}"
OUT_DIR="results/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
    --dataset "${DATASET}"
    --task "${TASK}"
    --precompute
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --num_neighbors "${NUM_NEIGHBORS}"
    --num_layers "${NUM_LAYERS}"
    --gt_conv_type "${GT_CONV_TYPE}"
    --ablate "${ABLATE}"
    --channels "${CHANNELS}"
    --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --warmup_steps "${WARMUP}"
    --ff_dropout "${DROPOUT}"
    --attn_dropout "${DROPOUT}"
    --run_name "${RUN_NAME}"
    --out_dir "${OUT_DIR}"
)

run_phase() {
    local phase_name="$1"
    shift
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase: ${phase_name}"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    torchrun \
        --nproc_per_node=1 \
        --master_port="${MASTER_PORT}" \
        main_node_ddp.py \
        "${COMMON_ARGS[@]}" \
        "$@"
}

# Phase 1: teacher (train RelGT with random sampling).
run_phase "teacher" --run_mode teacher

# Phase 2: distill (freeze RelGT, train DistillSampler via MSE on pre-softmax logits).
run_phase "distill" --run_mode distill

# Phase 3: joint (curate K from scope=3000 using sampler, fine-tune transformer + head).
run_phase "joint" \
    --run_mode joint \
    --sample_scope "${SAMPLE_SCOPE}" \
    --sample_temp "${SAMPLE_TEMP}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All three phases complete. Artifacts in ${OUT_DIR}"
