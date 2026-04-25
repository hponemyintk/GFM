# PR1 Parity Results — `dev-kyaw` vs new `data/` package

**Status:** awaiting runs.

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

| Task | Pipeline | Seeds | Mean | Std | Notes |
|---|---|---|---|---|---|
| rel-f1 / driver-position (MAE↓) | dev-kyaw | 5 | _TBD_ | _TBD_ | |
| rel-f1 / driver-position (MAE↓) | new | 5 | _TBD_ | _TBD_ | |
| rel-f1 / driver-top3 (AUROC↑) | dev-kyaw | 5 | _TBD_ | _TBD_ | |
| rel-f1 / driver-top3 (AUROC↑) | new | 5 | _TBD_ | _TBD_ | |

## Acceptance criterion

For each task, the new pipeline's `(mean − std)` must be ≥ dev-kyaw's `(mean − std)` (regression: flip signs since lower is better), and the means must be within `1 × std` of each other (68%-CI overlap).

If the criterion fails for either task, do NOT merge PR1 — investigate. Likely culprits: (a) the new sampler's set-iteration order diverging from dev-kyaw under some PYTHONHASHSEED edge case (rerun with `PYTHONHASHSEED=0` set explicitly); (b) collate ordering differences; (c) a missed edge case in `_precompute_sampling`.
