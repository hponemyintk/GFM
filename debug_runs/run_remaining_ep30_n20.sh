#!/usr/bin/env bash
# Resumes the 20-seed × 30-epoch sweep from K=20 (seed 6) through K=40, K=300.
# K=50, K=10, K=30 are already complete and committed.
# After each K: summarize → plot training curves → git commit + push.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS=30
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

run_sweep() {
    local k="$1"
    local sweep_script="$2"
    local out="$LOG_DIR/sweep_n${k}_ep${EPOCHS}.log"

    echo ""
    echo "============================================================"
    echo " Starting sweep K=${k}, epochs=${EPOCHS}  [$(date)]"
    echo "============================================================"

    if bash "$sweep_script" 2>&1 | tee -a "$out"; then
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

# K=20 resume (seeds 6-19; s0-s5 already done for both branches)
run_sweep 20 "$SCRIPT_DIR/sweep_dt3_baseline_n20_ep30_n20_resume.sh"

# K=40 full sweep
run_sweep 40 "$SCRIPT_DIR/sweep_dt3_baseline_n40_ep30_n20.sh"

# K=300 full sweep
run_sweep 300 "$SCRIPT_DIR/sweep_dt3_baseline_n300_ep30_n20.sh"

echo ""
echo "============================================================"
echo " ALL REMAINING SWEEPS DONE  [$(date)]"
echo "============================================================"
