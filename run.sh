#!/usr/bin/env bash
# Run script for single-GPU training (tested on RTX 5070 12GB)
# Usage: ./run.sh [dataset] [task] [batch_size] [epochs] [num_gpus]
# Examples:
#   ./run.sh rel-f1 driver-position 32 10       # single GPU
#   ./run.sh rel-amazon user-churn 32 10 4      # 4 GPUs

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="${1:-rel-f1}"
TASK="${2:-driver-top3}"
# DATASET="${1:-rel-hm}"
# TASK="${2:-user-churn}"
BATCH_SIZE="${3:-32}"
EPOCHS="${4:-10}"
NUM_GPUS="${5:-8}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${DATASET}_${TASK}_bs${BATCH_SIZE}_gpu${NUM_GPUS}_$(date +%Y%m%d_%H%M%S).log"

echo "Running: dataset=$DATASET task=$TASK batch_size=$BATCH_SIZE epochs=$EPOCHS gpus=$NUM_GPUS"
echo "Logging to: $LOG_FILE"

WANDB_MODE="${WANDB_MODE:-disabled}" \
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    main_node_ddp.py \
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
    --num_workers 2 \
    --epochs "${EPOCHS}" \
    --amp --amp_dtype bfloat16 \
    --lr 0.0001 \
    --warmup_steps 100 \
    --ff_dropout 0.3 \
    --attn_dropout 0.3 \
    --run_name "relgt-l4-512-bs${BATCH_SIZE}" \
    --out_dir "results/relgt-l4-512-bs${BATCH_SIZE}" \
    2>&1 | tee "${LOG_FILE}"
