#!/usr/bin/env bash
# Reduced-config smoke test for the distilled-sampler three-phase pipeline.
# Designed to fit on a single 12GB GPU in roughly 30-60 minutes total
# (dominated by one-time HDF5 precompute). This script does NOT modify any
# training code — it wraps main_node_ddp.py, tees per-phase stdout for later
# parsing, runs a diagnostic dump after phase 2, and produces plots at the end.
#
# Usage:
#   bash expts/smoke-test-distilled-sampler.sh [GPU_ID]
set -euo pipefail

# --- Per-process memory guard (cgroup-based) -------------------------------
# Each torchrun phase runs inside a systemd-user transient scope with a 20 GB
# resident-memory cap. When usage exceeds the cap, the kernel cgroup OOM killer
# terminates only the offender (not the whole WSL2 VM). This is a real-RAM
# cap, unlike ulimit -v which counts virtual address space (CUDA reserves
# tens of GB of VA at import, tripping -v immediately).
#
# Requires user systemd (XDG_RUNTIME_DIR set). Fallback: run commands
# bare without the cgroup if systemd-run --user is not available.
MEM_MAX="${MEM_MAX:-27G}"
if systemd-run --user --scope -p MemoryMax=256M -q true 2>/dev/null; then
    WRAP_CMD=(systemd-run --user --scope --quiet -p MemoryMax="${MEM_MAX}" -p MemorySwapMax=4G)
    echo "[$(date '+%F %T')] memory guard: systemd-user scope with MemoryMax=${MEM_MAX}"
else
    WRAP_CMD=()
    echo "[$(date '+%F %T')] memory guard: DISABLED (systemd-run --user unavailable)"
fi

GPU_ID="${1:-0}"
MASTER_PORT="${MASTER_PORT:-29310}"

DATASET="rel-f1"
TASK="driver-top3"
SEED=0

# Reduced from run-large-base-experiments.sh so the test finishes quickly:
BATCH_SIZE=16
NUM_NEIGHBORS=128
NUM_LAYERS=2
CHANNELS=128
MAX_STEPS_PER_EPOCH=30
NUM_WORKERS=1          # keep low: precompute pickles HeteroData per task; >1 worker OOMs on 30 GB RAM
EPOCHS_TEACHER=2
EPOCHS_DISTILL=3
EPOCHS_JOINT=2
LR=0.0001
WARMUP=5
DROPOUT=0.3

SAMPLE_SCOPE=512
SAMPLE_TEMP=1.0

RUN_NAME="smoke-distilled-sampler-${DATASET}-${TASK}"
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
    --gt_conv_type "full"
    --ablate "none"
    --channels "${CHANNELS}"
    --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}"
    --num_workers "${NUM_WORKERS}"
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
    local log_path="${OUT_DIR}/${phase_name}.log"
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase: ${phase_name}  (log: ${log_path})"
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    WANDB_MODE=offline \
    WANDB_SILENT=true \
    "${WRAP_CMD[@]}" torchrun \
        --nproc_per_node=1 \
        --master_port="${MASTER_PORT}" \
        main_node_ddp.py \
        "${COMMON_ARGS[@]}" \
        "$@" 2>&1 | tee "${log_path}"
}

run_phase "teacher" --run_mode teacher --epochs "${EPOCHS_TEACHER}"
run_phase "distill" --run_mode distill --epochs "${EPOCHS_DISTILL}"

# Standalone diagnostic: dump teacher logits vs sampler predictions on a val batch.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dumping sampler diagnostic NPZ..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${WRAP_CMD[@]}" python dump_sampler_diagnostic.py \
    --dataset "${DATASET}" --task "${TASK}" --out_dir "${OUT_DIR}" \
    --num_neighbors "${NUM_NEIGHBORS}" --num_layers "${NUM_LAYERS}" \
    --channels "${CHANNELS}" --ff_dropout "${DROPOUT}" --attn_dropout "${DROPOUT}" \
    --gt_conv_type "full" --ablate "none" \
    --batch_size "${BATCH_SIZE}" --seed "${SEED}" --split val

run_phase "joint" --run_mode joint --epochs "${EPOCHS_JOINT}" \
    --sample_scope "${SAMPLE_SCOPE}" --sample_temp "${SAMPLE_TEMP}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering plots..."
python plot_distilled_sampler.py --out_dir "${OUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Artifacts in ${OUT_DIR}"
