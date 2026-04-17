#!/usr/bin/env bash
# rel-f1 driver-top3, K=20, 30 epochs, 20 seeds, both branches.
# ogPASS 3-phase: warmup=6, sampler_only=6, joint=18 (20/20/60% of 30).
set -e
RUN_ONE=/home/jedi/research_repos/GFM/debug_runs/run_one.sh
EPOCHS=30
K=20
COMMON_ARGS=(--num_neighbors "$K")
PASS_ARGS=(--sampler_warmup_epochs 6 --sampler_only_epochs 6 --use_reinforce_baseline)

for SEED in $(seq 0 19); do
    echo "=== devkyaw_dt3_bl_n${K}_ep${EPOCHS}_s${SEED} ==="
    "$RUN_ONE" /tmp/gfm-dev-kyaw "devkyaw_dt3_bl_n${K}_ep${EPOCHS}_s${SEED}" "$SEED" "$EPOCHS" "${COMMON_ARGS[@]}"
    echo "=== ogpass_dt3_bl_n${K}_ep${EPOCHS}_s${SEED} ==="
    "$RUN_ONE" /home/jedi/research_repos/GFM "ogpass_dt3_bl_n${K}_ep${EPOCHS}_s${SEED}" "$SEED" "$EPOCHS" "${COMMON_ARGS[@]}" "${PASS_ARGS[@]}"
done
