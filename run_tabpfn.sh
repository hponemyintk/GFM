#!/usr/bin/env bash
# Run TabPFN ICL inference using features extracted from a trained RelGT model

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

DATASET="${1:-rel-f1}"
TASK="${2:-driver-top3}"
CHECKPOINT="${3:-}"
NUM_GPUS="${4:-1}"
BATCH_SIZE="${5:-512}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tabpfn_${DATASET}_${TASK}_$(date +%Y%m%d_%H%M%S).log"

echo "Running TabPFN ICL: dataset=$DATASET task=$TASK num_gpus=$NUM_GPUS"
echo "Logging to: $LOG_FILE"

CHECKPOINT_ARG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_ARG="--checkpoint ${CHECKPOINT}"
fi

CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29501 \
    tabpfn_icl_inference.py \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --precompute \
    --seed 42 \
    --batch_size "${BATCH_SIZE}" \
    --num_neighbors 300 \
    --num_layers 4 \
    --gt_conv_type full \
    --channels 168 \
    --num_centroids 4096 \
    --num_workers 8 \
    --tabpfn_context_size 10000 \
    --run_name "tabpfn-icl-${DATASET}-${TASK}" \
    --out_dir "results/tabpfn-icl" \
    --amp \
    ${CHECKPOINT_ARG} \
    2>&1 | tee "${LOG_FILE}"
