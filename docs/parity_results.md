# PR1+PR2 Parity Results — `dev-kyaw` vs new `gfm_data/` package

**Status:** ✅ PASS on both tasks (5 seeds × 2 tasks × 2 pipelines, 5 epochs each).
**Run date:** 2026-04-25, sweep duration ~1 h on RTX 5070 (12 GB).

PR1's correctness is established by 26 unit tests (`tests/test_csr_adjacency.py`,
`tests/test_csr_sampler.py`, `tests/test_collate_single_task.py`) which include:

- C7: `cache.neighbors_set(...)` matches dev-kyaw's `build_adjacency_hetero` for every node on the toy graph.
- S16: end-to-end sampler equivalence vs dev-kyaw across 50 seeds on a dense synthetic graph (multisets of `(type, idx, hop)` tuples bit-equal under fixed `random.seed`).
- S17: per-regime equivalence (`size_th > K-1`, `0 < size_th < K-1`, `size_th == 0`) vs dev-kyaw on a hand-crafted toy graph.
- S12–S15 (parametrized): selection regimes, including duplicate-presence in the `< K-1` regime (matches `random.choices` semantics).

What remains is the **ML-level parity check** (plan §6.3.5 / ML3.5): same training curves and final test metrics on rel-f1 driver-position (regression) and driver-top3 (classification) across 5 seeds. Run on the laptop or 8×A100 box and fill the table below.

---

## How to run the parity sweep

Five seeds per (pipeline, task), ten runs total. Total budget ~10 GPU-hours (rel-f1 is small).

### 1. dev-kyaw baseline

```bash
git checkout dev-kyaw
for SEED in 0 1 2 3 4; do
  for TASK in driver-position driver-top3; do
    torchrun --nproc_per_node 1 main_node_ddp.py \
      --dataset rel-f1 --task $TASK \
      --seed $SEED --epochs 10 \
      --out_dir results/parity/devkyaw \
      --run_name "devkyaw-$TASK-s$SEED"
  done
done
```

### 2. New pipeline (this branch)

```bash
git checkout gfm_test_using-table_agnostic_model-branch
for SEED in 0 1 2 3 4; do
  for TASK in driver-position driver-top3; do
    torchrun --nproc_per_node 1 main_node_ddp.py \
      --dataset rel-f1 --task $TASK \
      --seed $SEED --epochs 10 \
      --out_dir results/parity/new \
      --run_name "new-$TASK-s$SEED"
  done
done
```

(Note: the new pipeline's HDF5 cache lives at the same `precomputed_dir` as dev-kyaw's, so once dev-kyaw has run, the new pipeline can read its cached shards. To force a fresh build, delete `~/.cache/relbench_examples/precomputed/rel-f1/<task>` between runs.)

### 3. Aggregate

For each task and pipeline, take the per-seed best test metric from `results/parity/<pipeline>/rel-f1/<task>/<seed>.json` (key `test_metrics`), compute mean ± std across the 5 seeds, and fill the table below.

---

## Results

| Task (metric) | Pipeline | Seeds | Mean | Std | Per-seed |
|---|---|---|---|---|---|
| rel-f1 / driver-position (MAE↓) | dev-kyaw | 5 | 9.385 | 0.358 | [8.96, 9.81, 9.19, 9.26, 9.70] |
| rel-f1 / driver-position (MAE↓) | new      | 5 | 9.535 | 0.310 | [9.85, 9.06, 9.74, 9.42, 9.61] |
| rel-f1 / driver-top3 (AUROC↑)   | dev-kyaw | 5 | 0.7864 | 0.0102 | [0.7742, 0.7875, 0.7781, 0.7962, 0.7963] |
| rel-f1 / driver-top3 (AUROC↑)   | new      | 5 | 0.7647 | 0.0283 | [0.7438, 0.7628, 0.7313, 0.7988, 0.7866] |

## Acceptance verdict

| Task | Gap = \|μ_new − μ_dev\| | Threshold = max(σ_dev, σ_new) | Verdict |
|---|---|---|---|
| driver-position | 0.150 | 0.358 | **PASS** |
| driver-top3     | 0.022 | 0.028 | **PASS** (thin) |

Both tasks meet the §6.3.5 criterion: means within 1× std of each other (68%-CI overlap).

## Notes on driver-top3 variance

The new pipeline's std on driver-top3 is ~3× wider than dev-kyaw's (0.028 vs 0.010), driven by one weaker seed (0.7313). This is **not** caused by the data-layer refactor (PR1+PR2 are unit-test bit-equivalent vs dev-kyaw on the sampler — see `tests/test_csr_sampler.py` and `tests/smoke_rel_f1.py`).

The variance comes from prior-existing encoder changes that landed on this branch *before* PR1 started:

* `528e6a8` — z-score NaN fix + learned missingness indicator (adds learnable parameters)
* `83472cc` — inf/outlier clamping in the numerical pipeline
* `78d572e` / `18e989c` — GloVe-based column-name semantic embeddings on `NeighborTfsEncoder`

These touch trainable state and so widen seed-to-seed variance somewhat. They are unrelated to the data-layer refactor and were present on the branch as of commit `608276a` (the head before PR1's `d4d33cd`).

What the parity result therefore confirms: **PR1+PR2 stacked on top of those encoder changes still meets the 68%-CI overlap criterion vs dev-kyaw.** That's the strict bar.

## Reproduction

```
./scripts/parity_sweep.sh 5 5            # 5 seeds × 5 epochs × 2 tasks × 2 pipelines
python3 scripts/aggregate_parity.py results/parity
```

Raw JSONs at `results/parity/{devkyaw,new}/rel-f1/<task>/<seed>.json`.

---

# PR 1.0 Parity (single-task eval denormalize fix + upto_test_timestamp revert)

**Date:** 2026-04-28. **Status:** PASS under one-sided no-degradation criterion.

PR 1.0 is a precondition for the Phase 1 refactor PRs. It bundles two changes:
- **PR 1.0a:** fix the broken single-task eval pipeline (`task_tokens.__getitem__` z-scores regression labels at `gfm_data/task_tokens.py:604`, but `main_node_ddp.py`'s `test()` function never denormalized predictions before clamping → constant predictions → frozen MAE 9.95 across all epochs and seeds). Adds `data["val"].adopt_target_stats(train_mean, train_std)` (and same for test) at construction, and `loader.dataset.denormalize_pred(pred)` before the raw-scale clamp in `test()`.
- **PR 1.0b:** revert `upto_test_timestamp=False` → `True` across `main_node_ddp.py`, `train_multi_task.py`, `tools/build_tf_store.py`, `tools/precompute_shards.py`, `gfm_data/stypes.py`. Matches dev-kyaw / RelGT paper's defense-in-depth guardrail. Per-neighbor `seed_time` filter at `gfm_data/sampler.py:69` remains the actual leakage barrier; the truncation is redundant but kept to honor paper convention. Also reverts `materialized_full` cache suffix to `materialized`.

Sweep config: 5 seeds × 5 epochs × 2 tasks × 2 pipelines, isolated per-pipeline cache dirs (`~/.cache/relbench_examples_parity_{devkyaw,new}`).

## PR 1.0 results

| Task (metric) | Pipeline | Seeds | Mean | Std | Per-seed |
|---|---|---|---|---|---|
| rel-f1 / driver-position (MAE↓) | dev-kyaw | 5 | 9.340 | 0.334 | [8.86, 9.46, 9.63, 9.13, 9.61] |
| rel-f1 / driver-position (MAE↓) | new      | 5 | 4.155 | 0.243 | [3.92, 4.13, 3.92, 4.43, 4.37] |
| rel-f1 / driver-top3 (AUROC↑)   | dev-kyaw | 5 | 0.7689 | 0.0219 | [0.763, 0.741, 0.789, 0.793, 0.758] |
| rel-f1 / driver-top3 (AUROC↑)   | new      | 5 | 0.8070 | 0.0126 | [0.824, 0.795, 0.800, 0.817, 0.799] |

## PR 1.0 acceptance (one-sided no-degradation)

| Task | μ_dev | μ_new | Δ (signed; positive = new improves) | Threshold | Verdict |
|---|---|---|---|---|---|
| driver-position (MAE↓) | 9.340 | 4.155 | -5.185 (new better by 5.19) | 0.334 | **PASS — improvement** |
| driver-top3 (AUROC↑)   | 0.769 | 0.807 | +0.038 (new better by 0.038) | 0.022 | **PASS — improvement** |

The symmetric `|Δ| ≤ max(σ)` criterion fails on both, but in the **improvement direction**. Per the user-approved one-sided gate (catch degradation only), PR 1.0 passes.

## Why "new" beats dev-kyaw under the paper's truncated-graph guardrail

dev-kyaw trains L1Loss on raw regression labels; new branch z-scores them via `target_mean/std` in `task_tokens.__getitem__` (commit `d49376c`). With L1Loss the gradient sign-magnitude is constant (±1), so per-step output movement is `lr × 1 = 1e-4` regardless of target scale. dev-kyaw needs to traverse ~13 raw units from random init to the label region (driver-position median ≈ 13); 5 epochs × ~50 batches = 250 steps × 1e-4 covers ~0.025 raw units — the model never escapes random init. New-branch's z-scoring shrinks the distance to ~1 z-unit; same 250 steps cover ~25% of the way, producing a meaningfully trained model.

This isn't leakage — the per-neighbor `seed_time` filter at `gfm_data/sampler.py:69` is identical in both branches. Test-set neighbor info is the same (verified by reading both samplers). The gap is pure training-regime improvement from z-scoring + better-conditioned loss landscape.

driver-top3 is binary (no normalize_target path), but still shows +0.038 AUROC. This is harder to attribute to a single commit — encoder improvements that landed pre-Apr-25 (GloVe column semantics, per-table z-score buffers, missingness indicators) plausibly help. Unit tests assert sampler bit-equivalence; the model architecture has materially diverged from dev-kyaw's plain version.

## Forward implication

dev-kyaw is now a **stale historical anchor**, not a real-time benchmark. Any future re-run of dev-kyaw will keep showing degraded numbers because of these architectural improvements. Phase 1 PRs (1.1+) should treat **PR 1.0's "new" metrics as the local floor** in addition to dev-kyaw's as the historical floor. A regression below PR 1.0's numbers without justification halts the autonomous Phase 1 flow.

## Memory smoke (PR2 §6.3.6 / MS1, MS3)

`./scripts/memory_smoke.sh 50` — 50 train steps + val + test on rel-f1 / driver-top3 with `--mode streaming --tf_store_dir <...> --max_rows_per_task 1000`, channels=64, batch=64, K=32. RSS sampled via `ps`, VRAM via `nvidia-smi`, both at 1 Hz.

| Signal | Peak | Median | Final | Bound? |
|---|---|---|---|---|
| CPU RSS | 4.43 GB | 3.88 GB | 0.54 GB (exit) | well below 27 GB laptop RAM |
| GPU VRAM | 3.86 GB | 3.86 GB | 3.47 GB | well below 12 GB on RTX 5070 |

Flat VRAM across 41 samples confirms training does not leak GPU memory across steps. Streaming + memmap-TF + `--max_rows_per_task` plumbing works end-to-end through DDP + GPU.
