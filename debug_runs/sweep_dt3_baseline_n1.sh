#!/usr/bin/env bash
# Exp A1: rel-f1 driver-top3, K=1, 20 epochs, 5 seeds, both branches.
# K=1 seed-only — PASS sampler_only is a no-op; schedule shape kept.
set -e
RUN_ONE=/home/jedi/research_repos/GFM/debug_runs/run_one.sh
EPOCHS=20
K=1
COMMON_ARGS=(--num_neighbors "$K")
PASS_ARGS=(--sampler_warmup_epochs 4 --sampler_only_epochs 4 --use_reinforce_baseline)

for SEED in 0 1 2 3 4; do
    echo "=== devkyaw_dt3_bl_n${K}_ep${EPOCHS}_s${SEED} ==="
    "$RUN_ONE" /tmp/gfm-dev-kyaw "devkyaw_dt3_bl_n${K}_ep${EPOCHS}_s${SEED}" "$SEED" "$EPOCHS" "${COMMON_ARGS[@]}"
    echo "=== ogpass_dt3_bl_n${K}_ep${EPOCHS}_s${SEED} ==="
    "$RUN_ONE" /home/jedi/research_repos/GFM "ogpass_dt3_bl_n${K}_ep${EPOCHS}_s${SEED}" "$SEED" "$EPOCHS" "${COMMON_ARGS[@]}" "${PASS_ARGS[@]}"
done
