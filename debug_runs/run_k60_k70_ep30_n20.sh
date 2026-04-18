#!/usr/bin/env bash
# Run K=60 and K=70 sweeps (20 seeds × 2 branches × 30 epochs each).
# Waits for K=300 (run_remaining_ep30_n20.sh) to finish before starting.
# Summarizes, plots training curves, and commits+pushes after each K.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS=30
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

# Wait for K=300 sweep to complete
echo "[wait] Waiting for run_remaining_ep30_n20.sh (K=300) to finish..."
while pgrep -f "run_remaining_ep30_n20.sh" > /dev/null 2>&1; do
    sleep 60
done
echo "[wait] K=300 sweep finished. Starting K=60, K=70.  [$(date)]"

run_sweep() {
    local k="$1"
    local out="$LOG_DIR/sweep_n${k}_ep${EPOCHS}.log"

    echo ""
    echo "============================================================"
    echo " Starting sweep K=${k}, epochs=${EPOCHS}  [$(date)]"
    echo "============================================================"

    if bash "$SCRIPT_DIR/sweep_dt3_baseline_n${k}_ep${EPOCHS}_n20.sh" 2>&1 | tee "$out"; then
        echo "[OK] K=${k} sweep finished [$(date)]"
    else
        echo "[WARN] K=${k} sweep exited non-zero — partial results may exist [$(date)]"
    fi

    echo "[summarize] K=${k} epochs=${EPOCHS}"
    python "$SCRIPT_DIR/summarize_dt3.py" --k "$k" --epochs "$EPOCHS"

    echo "[plot] K=${k} epochs=${EPOCHS}"
    python "$SCRIPT_DIR/plot_training_curves.py" --k "$k" --epochs "$EPOCHS"

    echo "[git] committing K=${k} results [$(date)]"
    cd /home/jedi/research_repos/GFM
    git add \
        debug_runs/training_curves_n${k}_ep${EPOCHS}.png \
        experiment_results_baseline_fix.md
    git commit -m "Add K=${k} ep${EPOCHS} results: training curves and summary (n=20 seeds)"
    git push
    echo "[git] pushed K=${k} [$(date)]"
}

run_sweep 60
run_sweep 70

echo ""
echo "============================================================"
echo " K=60 and K=70 SWEEPS DONE  [$(date)]"
echo "============================================================"
