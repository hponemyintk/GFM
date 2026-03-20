#!/usr/bin/env bash
# Simulate multi-GPU DDP training on a single GPU.
# All ranks run on cuda:0 — useful for testing DDP code paths (rank guards,
# barriers, DistributedSampler, all_gather) without multiple physical GPUs.
#
# Usage: ./simulate_multigpu.sh [dataset] [task] [batch_size] [epochs] [num_gpus]
# Examples:
#   ./simulate_multigpu.sh rel-f1 driver-top3 8 1 2    # 2 simulated GPUs
#   ./simulate_multigpu.sh rel-f1 driver-top3 4 1 4    # 4 simulated GPUs
#
# NOTE: Reduce batch_size to avoid OOM — all ranks share one physical GPU.

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="${1:-rel-f1}"
TASK="${2:-driver-top3}"
BATCH_SIZE="${3:-8}"
EPOCHS="${4:-1}"
NUM_GPUS="${5:-2}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sim_${NUM_GPUS}gpu_${DATASET}_${TASK}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "Simulating ${NUM_GPUS}-GPU DDP on a single GPU"
echo "Running: dataset=$DATASET task=$TASK batch_size=$BATCH_SIZE epochs=$EPOCHS"
echo "Logging to: $LOG_FILE"

WANDB_MODE="${WANDB_MODE:-disabled}" \
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    simulate_multigpu.py \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --precompute \
    --seed 42 \
    --batch_size "${BATCH_SIZE}" \
    --num_neighbors 300 \
    --num_layers 4 \
    --gt_conv_type full \
    --channels 512 \
    --num_centroids 4096 \
    --max_steps_per_epoch 3000 \
    --num_workers 0 \
    --epochs "${EPOCHS}" \
    --amp --amp_dtype bfloat16 \
    --lr 0.0001 \
    --warmup_steps 100 \
    --ff_dropout 0.3 \
    --attn_dropout 0.3 \
    --run_name "sim-${NUM_GPUS}gpu-relgt-l4-512-bs${BATCH_SIZE}" \
    --out_dir "results/sim-${NUM_GPUS}gpu-relgt-l4-512-bs${BATCH_SIZE}" \
    2>&1 | tee "${LOG_FILE}"
