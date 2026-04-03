#!/usr/bin/env bash
# Two-stage training pipeline:
#   Stage 1: Train with uniform sampling until convergence
#   Stage 2: Re-sample with learned similarity, train again
set -euo pipefail

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="${1:-rel-f1}"
TASK="${2:-driver-top3}"
BATCH_SIZE="${3:-32}"
STAGE1_EPOCHS="${4:-10}"
STAGE2_EPOCHS="${5:-10}"

# Shared config
CHANNELS=512
NUM_NEIGHBORS=300
NUM_LAYERS=4
NUM_WORKERS=8
SEED=42
LR=0.0001

# Scoring weights for Stage 2 (override via env vars)
W_FK="${W_FK:-5.0}"
W_SIM="${W_SIM:-2.0}"
W_RECENCY="${W_RECENCY:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

STAGE1_OUT="results/${DATASET}_${TASK}_stage1"
STAGE2_OUT="results/${DATASET}_${TASK}_stage2"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
    --dataset "${DATASET}"
    --task "${TASK}"
    --precompute
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --num_neighbors "${NUM_NEIGHBORS}"
    --num_layers "${NUM_LAYERS}"
    --gt_conv_type full
    --channels "${CHANNELS}"
    --num_centroids 4096
    --max_steps_per_epoch 3000
    --num_workers "${NUM_WORKERS}"
    --lr "${LR}"
    --warmup_steps 100
    --ff_dropout 0.3
    --attn_dropout 0.3
)

########################################
# Stage 1: Uniform sampling (finetune)
########################################
echo "=========================================="
echo "Stage 1: Training with uniform sampling"
echo "  Dataset: ${DATASET}, Task: ${TASK}"
echo "  Epochs: ${STAGE1_EPOCHS}"
echo "=========================================="

STAGE1_LOG="${LOG_DIR}/${DATASET}_${TASK}_stage1_$(date +%Y%m%d_%H%M%S).log"

WANDB_MODE="${WANDB_MODE:-disabled}" \
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    main_node_ddp.py \
    "${COMMON_ARGS[@]}" \
    --train_stage finetune \
    --epochs "${STAGE1_EPOCHS}" \
    --out_dir "${STAGE1_OUT}" \
    --run_name "${DATASET}-${TASK}-stage1" \
    2>&1 | tee "${STAGE1_LOG}"

# Verify Stage 1 produced checkpoint
# main_node_ddp.py saves to {out_dir}/{dataset}/{task}/finetuned.pt
ENCODER_WEIGHTS="${STAGE1_OUT}/${DATASET}/${TASK}/finetuned.pt"
if [ ! -f "${ENCODER_WEIGHTS}" ]; then
    echo "ERROR: Stage 1 did not produce ${ENCODER_WEIGHTS}"
    exit 1
fi
echo "Stage 1 complete. Encoder weights saved to: ${ENCODER_WEIGHTS}"

########################################
# Stage 2: Similarity-based resampling
########################################
echo ""
echo "=========================================="
echo "Stage 2: Training with similarity-based sampling"
echo "  Encoder weights: ${ENCODER_WEIGHTS}"
echo "  Scoring: w_fk=${W_FK}, w_sim=${W_SIM}, w_recency=${W_RECENCY}, temp=${TEMPERATURE}"
echo "  Epochs: ${STAGE2_EPOCHS}"
echo "=========================================="

STAGE2_LOG="${LOG_DIR}/${DATASET}_${TASK}_stage2_$(date +%Y%m%d_%H%M%S).log"

WANDB_MODE="${WANDB_MODE:-disabled}" \
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    main_node_ddp.py \
    "${COMMON_ARGS[@]}" \
    --train_stage similarity_resample \
    --epochs "${STAGE2_EPOCHS}" \
    --encoder_weights_path "${ENCODER_WEIGHTS}" \
    --w_fk "${W_FK}" \
    --w_sim "${W_SIM}" \
    --w_recency "${W_RECENCY}" \
    --temperature "${TEMPERATURE}" \
    --out_dir "${STAGE2_OUT}" \
    --run_name "${DATASET}-${TASK}-stage2" \
    2>&1 | tee "${STAGE2_LOG}"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Stage 1 results: ${STAGE1_OUT}"
echo "  Stage 2 results: ${STAGE2_OUT}"
echo "  Logs: ${STAGE1_LOG}, ${STAGE2_LOG}"
echo "=========================================="
