#!/usr/bin/env bash
# Run script for single RTX 5070 (12GB VRAM)
# Model size kept at 512 channels, batch size reduced to fit in memory

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="${1:-rel-f1}"
TASK="${2:-driver-top3}"
BATCH_SIZE="${3:-32}"
EPOCHS="${4:-10}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${DATASET}_${TASK}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "Running: dataset=$DATASET task=$TASK batch_size=$BATCH_SIZE epochs=$EPOCHS"
echo "Logging to: $LOG_FILE"

WANDB_MODE="${WANDB_MODE:-disabled}" \
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=1 \
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
    --num_workers 8 \
    --epochs "${EPOCHS}" \
    --lr 0.0001 \
    --warmup_steps 100 \
    --ff_dropout 0.3 \
    --attn_dropout 0.3 \
    --run_name "relgt-l4-512-bs${BATCH_SIZE}" \
    --out_dir "results/relgt-l4-512-bs${BATCH_SIZE}" \
    2>&1 | tee "${LOG_FILE}"
