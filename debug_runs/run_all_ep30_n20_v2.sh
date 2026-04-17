#!/usr/bin/env bash
# Master orchestrator: 20 seeds × 30 epochs × K in [50,10,30,20,40,300], both branches.
# Runs each K-sweep then immediately summarizes into experiment_results_baseline_fix.md.
# Does NOT set -e at the top level so a single-run failure won't abort later sweeps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS=30
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH=~/miniforge3/bin:$PATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gt

run_sweep() {
    local k="$1"
    local sweep_script="$SCRIPT_DIR/sweep_dt3_baseline_n${k}_ep${EPOCHS}_n20.sh"
    local out="$LOG_DIR/sweep_n${k}_ep${EPOCHS}.log"

    echo ""
    echo "============================================================"
    echo " Starting sweep K=${k}, epochs=${EPOCHS}  [$(date)]"
    echo "============================================================"

    if bash "$sweep_script" 2>&1 | tee "$out"; then
        echo "[OK] K=${k} sweep finished [$(date)]"
    else
        echo "[WARN] K=${k} sweep exited non-zero — partial results may exist [$(date)]"
    fi

    echo "[summarize] K=${k} epochs=${EPOCHS}"
    python "$SCRIPT_DIR/summarize_dt3.py" --k "$k" --epochs "$EPOCHS"
}

for K in 50 10 30 20 40 300; do
    run_sweep "$K"
done

echo ""
echo "============================================================"
echo " ALL SWEEPS DONE  [$(date)]"
echo "============================================================"
echo ""
echo "=== Final cross-K summary (AP mean ± std) ==="
python - <<'PYEOF'
import json, math, pathlib

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
EPOCHS = 30
KS = [50, 10, 30, 20, 40, 300]
BRANCHES = [("devkyaw", "dev-kyaw"), ("ogpass", "ogPASS")]
DATASET, TASK = "rel-f1", "driver-top3"

def mean_std(xs):
    if not xs: return float("nan"), float("nan")
    m = sum(xs)/len(xs)
    v = sum((x-m)**2 for x in xs)/(len(xs)-1) if len(xs)>1 else 0
    return m, math.sqrt(v)

header = f"{'K':>4}  {'branch':<10}  {'n':>3}  {'AP mean':>8}  {'AP std':>8}  {'delta AP':>9}"
print(header)
print("-" * len(header))
for k in KS:
    deltas = {}
    for slug, label in BRANCHES:
        vals = []
        for seed in range(20):
            p = RESULTS / f"{slug}_dt3_bl_n{k}_ep{EPOCHS}_s{seed}" / DATASET / TASK / f"{seed}.json"
            if p.exists():
                d = json.loads(p.read_text())
                v = d.get("test_metrics", {}).get("average_precision")
                if v is not None:
                    vals.append(v)
        m, s = mean_std(vals)
        deltas[slug] = m
        ap_str = f"{m:.4f}" if not math.isnan(m) else "    —"
        std_str = f"{s:.4f}" if not math.isnan(s) else "    —"
        print(f"{k:>4}  {label:<10}  {len(vals):>3}  {ap_str:>8}  {std_str:>8}")
    if "devkyaw" in deltas and "ogpass" in deltas:
        if not math.isnan(deltas["devkyaw"]) and not math.isnan(deltas["ogpass"]):
            delta = deltas["ogpass"] - deltas["devkyaw"]
            print(f"{'':>4}  {'Δ(og-dev)':<10}  {'':>3}  {'':>8}  {'':>8}  {delta:>+9.4f}")
    print()
PYEOF
