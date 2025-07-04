#!/usr/bin/env bash

check_array_lengths() {
    local arr1=("$1")
    local arr2=("$2")

    if [ "${#arr1[@]}" -ne "${#arr2[@]}" ]; then
        echo "Error: Arrays have different lengths."
        exit 1
    fi
}

BASE_PORT=29100

datasets=("rel-amazon" "rel-hm" "rel-trial" "rel-trial" "rel-event" "rel-avito" "rel-avito" "rel-avito")
tasks=("user-churn" "item-sales" "site-success" "study-outcome" "user-ignore" "ad-ctr" "user-clicks" "user-visits")
dropouts_per=(0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3)
bs_per=(512 512 256 256 256 256 256 256)
max_steps_per=(1000 1000 3000 3000 3000 3000 3000 3000)
epochs_per=(10 20 50 100 100 100 100 100)
#est_time=(28 8 27 3 4 2 11 15)

check_array_lengths "${datasets[@]}" "${tasks[@]}"
check_array_lengths "${datasets[@]}" "${dropouts_per[@]}"
check_array_lengths "${datasets[@]}" "${bs_per[@]}"
check_array_lengths "${datasets[@]}" "${max_steps_per[@]}"
check_array_lengths "${datasets[@]}" "${epochs_per[@]}"

# GPU IDs
gpu_nodes=(0 1 2 3 4 5 6 7)

num_layers=(4)
use_ks=(500 300 100)

# Build the list of all experiment configurations
expt_configs=()
for l in "${num_layers[@]}"; do
    for k in "${use_ks[@]}"; do
        for i in "${!datasets[@]}"; do
            expt_configs+=("${datasets[$i]}|${tasks[$i]}|${dropouts_per[$i]}|$l|$k|${epochs_per[$i]}|${bs_per[$i]}|${max_steps_per[$i]}")
        done
    done
done


total_experiments=${#expt_configs[@]}
num_gpus=${#gpu_nodes[@]}
echo "We have $total_experiments total experiments and $num_gpus available GPUs."

# Function to check if GPU is in use directly with nvidia-smi
is_gpu_in_use() {
    local gpu_id=$1
    # Check if GPU is in use by querying nvidia-smi
    if nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader | grep -q "[0-9]"; then
        return 0  # true - GPU is in use
    else
        return 1  # false - GPU is free
    fi
}

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to launch an experiment
launch_experiment() {
    local config="$1"
    local gpu_id=$2
    local port=$3

    # Parse config
    IFS='|' read -r dataset task dropout layers num_k epochs batch_size max_steps_per_epoch <<< "$config"
    local dro="${dropout//./}"  # e.g. 0.3 -> 03

    local run_name="relgt-l${layers}-k${num_k}-512-dropout${dro}-BS${batch_size}-MAX${max_steps_per_epoch}"
    local out_dir="results/${run_name}"
    mkdir -p "$out_dir"

    log_message "[GPU ${gpu_id}] Launching $dataset($task) with dropout=$dropout, layers=$layers, epochs=$epochs"

    # Create a lockfile to indicate this GPU is in use
    local lockfile="/tmp/gpu_${gpu_id}.lock"
    echo "experiment: $dataset-$task" > "$lockfile"

    # Launch the job
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    torchrun \
        --nproc_per_node=1 \
        --master_port="${port}" \
        main_node_ddp.py \
        --dataset "${dataset}" \
        --task "${task}" \
        --precompute \
        --seed 0 \
        --batch_size "${batch_size}" \
        --num_neighbors "${num_k}" \
        --num_layers "${layers}" \
        --channels 512 \
        --max_steps_per_epoch "${max_steps_per_epoch}" \
        --num_workers 8 \
        --epochs "${epochs}" \
        --lr 0.0001 \
        --warmup_steps 10 \
        --ff_dropout "${dropout}" \
        --attn_dropout "${dropout}" \
        --run_name "${run_name}" \
        --out_dir "${out_dir}" &

    # Wait briefly to ensure the job starts
    sleep 10

    # Check if the GPU is actually being used
    if ! is_gpu_in_use "$gpu_id"; then
        log_message "[WARNING] GPU ${gpu_id} doesn't appear to be running a job after launch attempt!"
    else
        log_message "[GPU ${gpu_id}] Job successfully started"
    fi
}

# Function to log status of all GPUs
log_gpu_status() {
    log_message "--- GPU Status Report ---"
    for gpu_id in "${gpu_nodes[@]}"; do
        if is_gpu_in_use "$gpu_id"; then
            # Get the PID of process using this GPU
            local pid
            pid=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader | head -n1)
            local mem
            mem=$(nvidia-smi -i "$gpu_id" --query-compute-apps=used_memory --format=csv,noheader | head -n1)
            log_message "GPU $gpu_id: BUSY (PID: $pid, Memory: $mem)"
        else
            log_message "GPU $gpu_id: FREE"
        fi
    done
    log_message "------------------------"
}

current_expt=0

# Initialize the first wave of experiments
log_message "Starting first wave of experiments"
for gpu_id in "${gpu_nodes[@]}"; do
    if (( current_expt < total_experiments )); then
        # Remove any stale lockfiles
        rm -f "/tmp/gpu_${gpu_id}.lock"

        # Check if GPU is truly free before launching
        if ! is_gpu_in_use "$gpu_id"; then
            # Use a unique master_port for each experiment
            port=$((BASE_PORT + current_expt))
            launch_experiment "${expt_configs[$current_expt]}" "$gpu_id" "$port"
            ((current_expt++))
        else
            log_message "[WARNING] GPU $gpu_id appears to be in use before we tried to use it!"
        fi
    fi
done

log_gpu_status

# Monitor and launch remaining experiments as GPUs become available
while (( current_expt < total_experiments )); do
    # Wait before checking again
    sleep 60

    # Check all GPUs and launch jobs on free ones
    for gpu_id in "${gpu_nodes[@]}"; do
        if ! is_gpu_in_use "$gpu_id" && (( current_expt < total_experiments )); then
            log_message "[GPU $gpu_id] is free. Preparing to launch next experiment..."
            # Sleep briefly to ensure GPU resources are fully released
            sleep 5
            port=$((BASE_PORT + current_expt))
            launch_experiment "${expt_configs[$current_expt]}" "$gpu_id" "$port"
            ((current_expt++))

            # Stop if no more experiments remain
            if (( current_expt >= total_experiments )); then
                break
            fi
        fi
    done

    # Log status every iteration
    log_gpu_status

    # Report progress
    log_message "Progress: $current_expt / $total_experiments experiments launched"
done

log_message "All experiments have been launched. Waiting for remaining jobs to complete..."

# Wait for all GPUs to become free
while true; do
    all_free=true
    for gpu_id in "${gpu_nodes[@]}"; do
        if is_gpu_in_use "$gpu_id"; then
            all_free=false
            break
        fi
    done

    if $all_free; then
        break
    fi

    log_message "Waiting for remaining jobs to complete..."
    log_gpu_status
    sleep 300  # Check every 5 minutes
done

# Clean up any lockfiles
for gpu_id in "${gpu_nodes[@]}"; do
    rm -f "/tmp/gpu_${gpu_id}.lock"
done

log_message "All experiments finished."
