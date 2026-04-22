#!/usr/bin/env bash
# 5-seed × 4-temperature sweep of the distilled-sampler pipeline.
# Phases 1 and 2 depend only on seed (teacher + sampler). Phase 3's curate
# and Gumbel-Top-K depend on seed AND sample_temp, so phase 3 is rerun for
# each (seed, temp_mode).
#
# Temp modes:
#   det : deterministic top-K (--curate_stochastic unset)
#   t1  : Gumbel-Top-K with temperature 1.0
#   t2  : Gumbel-Top-K with temperature 2.0
#   t5  : Gumbel-Top-K with temperature 5.0
#
# Usage:
#   bash expts/full-sweep-distilled-sampler.sh [GPU_ID]
set -uo pipefail

GPU_ID="${1:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29320}"

# --- Per-process memory guard (cgroup) -----------------------------------
# Cap each torchrun/python process group at 27 GB real RAM. The kernel
# cgroup OOM killer fires on this cgroup only; WSL2 stays alive.
MEM_MAX="${MEM_MAX:-27G}"
if systemd-run --user --scope -p MemoryMax=256M -q true 2>/dev/null; then
    WRAP_CMD=(systemd-run --user --scope --quiet -p MemoryMax="${MEM_MAX}" -p MemorySwapMax=4G)
    echo "[$(date '+%F %T')] memory guard: systemd-user scope with MemoryMax=${MEM_MAX}"
else
    WRAP_CMD=()
    echo "[$(date '+%F %T')] memory guard: DISABLED"
fi

DATASET="${DATASET:-rel-f1}"
TASK="${TASK:-driver-top3}"

BATCH_SIZE=16
NUM_NEIGHBORS=10
NUM_LAYERS=3
CHANNELS=256
MAX_STEPS_PER_EPOCH=500
NUM_WORKERS=1
EPOCHS=10
LR=0.0001
WARMUP=10
DROPOUT=0.3

SAMPLE_SCOPE=1024

SEEDS=(0 1 2 3 4)
# Each entry is "label|extra_args"; label becomes the phase-3 subdir name.
TEMP_MODES=(
    "det|"
    "t1|--curate_stochastic --sample_temp 1.0"
    "t2|--curate_stochastic --sample_temp 2.0"
    "t5|--curate_stochastic --sample_temp 5.0"
)

SWEEP_ROOT="results/full-sweep-distilled-sampler-${DATASET}-${TASK}"
mkdir -p "${SWEEP_ROOT}"

# Curated-HDF5 path is shared across seeds AND temps (cache_dir-scoped, not
# out_dir-scoped). We delete before each phase-3 run so every (seed, temp)
# gets its own curated HDF5 written from that seed's sampler + that temp.
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/relbench_examples}"
CURATED_DIR="${CACHE_DIR}/precomputed/${DATASET}/${TASK}/curated_${NUM_NEIGHBORS}"
wipe_curated() {
    if [[ -d "${CURATED_DIR}" ]]; then
        rm -rf "${CURATED_DIR}"/*
    fi
}

run_phase() {
    local out_dir="$1"; local log_name="$2"; local port="$3"; local seed="$4"
    local done_marker="$5"
    shift 5
    local log_path="${out_dir}/${log_name}.log"

    # Skip if already completed successfully.
    if grep -q "${done_marker}" "${log_path}" 2>/dev/null; then
        echo "=== [$(date '+%F %T')] SKIP seed=${seed} phase=${log_name} (already done) ==="
        return 0
    fi

    echo "=== [$(date '+%F %T')] seed=${seed} phase=${log_name} out=${out_dir} ==="
    local attempt max_attempts=3
    for attempt in 1 2 3; do
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        WANDB_MODE=offline \
        WANDB_SILENT=true \
        "${WRAP_CMD[@]}" torchrun \
            --nproc_per_node=1 \
            --master_port="${port}" \
            main_node_ddp.py \
            --dataset "${DATASET}" --task "${TASK}" --precompute \
            --seed "${seed}" \
            --batch_size "${BATCH_SIZE}" --num_neighbors "${NUM_NEIGHBORS}" \
            --num_layers "${NUM_LAYERS}" --channels "${CHANNELS}" \
            --gt_conv_type full --ablate none \
            --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
            --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
            --lr "${LR}" --warmup_steps "${WARMUP}" \
            --ff_dropout "${DROPOUT}" --attn_dropout "${DROPOUT}" \
            --run_name "seed${seed}-${log_name}" --out_dir "${out_dir}" \
            "$@" 2>&1 | tee "${log_path}" && return 0
        echo "=== [$(date '+%F %T')] RETRY ${attempt}/${max_attempts} seed=${seed} phase=${log_name} ==="
    done
    echo "=== [$(date '+%F %T')] FAILED after ${max_attempts} attempts: seed=${seed} phase=${log_name} ==="
    return 1
}

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    PORT=$((MASTER_PORT_BASE + i))
    SEED_DIR="${SWEEP_ROOT}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"

    # --- Phase 1: teacher (once per seed) ---
    run_phase "${SEED_DIR}" "teacher" "${PORT}" "${SEED}" "Best Test metrics" --run_mode teacher

    # --- Phase 2: distill (once per seed) ---
    run_phase "${SEED_DIR}" "distill" "${PORT}" "${SEED}" "ridge head_weights" --run_mode distill

    # Dump (teacher_logits, q_imp) on one val batch (skip if NPZ already exists).
    NPZ_PATH="${SEED_DIR}/${DATASET}/${TASK}/distill_diagnostic.npz"
    if [[ -f "${NPZ_PATH}" ]]; then
        echo "--- seed=${SEED} diagnostic NPZ already exists, skipping ---"
    else
        echo "--- seed=${SEED} dumping sampler diagnostic NPZ ---"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${WRAP_CMD[@]}" python dump_sampler_diagnostic.py \
            --dataset "${DATASET}" --task "${TASK}" --out_dir "${SEED_DIR}" \
            --num_neighbors "${NUM_NEIGHBORS}" --num_layers "${NUM_LAYERS}" \
            --channels "${CHANNELS}" --ff_dropout "${DROPOUT}" --attn_dropout "${DROPOUT}" \
            --gt_conv_type full --ablate none \
            --batch_size "${BATCH_SIZE}" --seed "${SEED}" --split val
    fi

    # Checkpoints are nested under {out_dir}/{dataset}/{task}/
    SEED_CKPT_DIR="${SEED_DIR}/${DATASET}/${TASK}"

    # --- Phase 3: joint (per temperature mode) ---
    for tm in "${TEMP_MODES[@]}"; do
        TM_LABEL="${tm%%|*}"
        TM_EXTRA="${tm#*|}"
        TM_DIR="${SEED_DIR}/${TM_LABEL}"
        mkdir -p "${TM_DIR}"

        # Force curated HDF5 rewrite (path is shared across seeds AND temps).
        wipe_curated

        # shellcheck disable=SC2086
        run_phase "${TM_DIR}" "joint" "${PORT}" "${SEED}" "Best Test metrics" \
            --run_mode joint --sample_scope "${SAMPLE_SCOPE}" \
            --teacher_ckpt "${SEED_CKPT_DIR}/phase1.pt" \
            --sampler_ckpt "${SEED_CKPT_DIR}/sampler.pt" \
            ${TM_EXTRA}
    done
done

echo "=== [$(date '+%F %T')] All ${#SEEDS[@]} seeds × ${#TEMP_MODES[@]} temps done. Aggregating... ==="
python plot_distilled_sampler_sweep.py --sweep_root "${SWEEP_ROOT}"
echo "=== [$(date '+%F %T')] Done. Artifacts under ${SWEEP_ROOT}/aggregate/ ==="
