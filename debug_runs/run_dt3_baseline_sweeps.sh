#!/usr/bin/env bash
# Driver: runs all 3 driver-top3 baseline-fix sweeps sequentially,
# summarizing after each so partial results land in
# experiment_results_baseline_fix.md even if a later sweep is interrupted.
set -e

SCRIPT_DIR="/home/jedi/research_repos/GFM/debug_runs"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

run_sweep() {
    local name="$1" k="$2" ep="$3"
    local out="$LOG_DIR/sweep_dt3_bl_${name}.out"
    echo "=========================================================="
    echo "[run_dt3_baseline_sweeps] starting sweep $name (K=$k, ep=$ep)"
    echo "[run_dt3_baseline_sweeps] driver log: $out"
    echo "=========================================================="
    "$SCRIPT_DIR/sweep_dt3_baseline_${name}.sh" 2>&1 | tee "$out"
    echo "[run_dt3_baseline_sweeps] sweep $name complete — summarizing"
    python "$SCRIPT_DIR/summarize_dt3.py" --k "$k" --epochs "$ep"
}

run_sweep n1  1  20
run_sweep n10 10 20
run_sweep n50 50 20

echo "=========================================================="
echo "[run_dt3_baseline_sweeps] ALL SWEEPS COMPLETE"
echo "=========================================================="
