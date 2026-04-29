# Truncated-graph caveat — autocomplete tasks crash on test-split seeds

**TL;DR**: We keep `upto_test_timestamp=True` (RelBench's default; matches the RelGT paper) so entity tables are truncated at `train_cutoff`. **Forecasting tasks** with stable seed entities (drivers, users, items) are unaffected — their val/test seed ids are within the truncated table. **Autocomplete tasks** (new in RelBench v2: `users-birthyear`, `results-position`, `qualifying-position`, `transactions-price`) ask "predict an attribute of a row that just appeared" — by definition their test seeds reference rows added after `train_cutoff`, which IndexError the truncated CSR adjacency at `graph_cache.py:288`.

The fix is the `--full_graph` flag (added in this PR): pass it to the build tools to use `upto_test_timestamp=False`. The per-neighbor `seed_time` filter at `gfm_data/sampler.py:69` is the actual leakage barrier; entity-table truncation is only defense-in-depth and is safe to disable when the task requires it.

## What's actually affected (empirical OOB check)

`(val OOB, test OOB)` = number of seed entity ids in val/test seed tables that exceed the truncated entity table size.

| dataset | task | seed entity | val OOB | test OOB | verdict |
|---|---|---|---|---|---|
| rel-f1 | driver-position | drivers | 0/499 | 0/760 | **safe** |
| rel-f1 | driver-dnf | drivers | 0/566 | 0/702 | **safe** |
| rel-f1 | driver-top3 | drivers | 0/588 | 0/726 | **safe** |
| rel-f1 | results-position | results | 0/1400 | **4798/4798 (100%)** | needs `--full_graph` |
| rel-f1 | qualifying-position | qualifying | 0/1854 | **5733/5733 (100%)** | needs `--full_graph` |
| rel-event | user-attendance | users | 0/2013 | 0/1958 | **safe (paper Table 1a, MAE 0.2502)** |
| rel-event | user-repeat | users | 0/268 | 0/246 | **safe (paper Table 1b, AUC 0.7609)** |
| rel-event | user-ignore | users | 0/2013 | 0/1958 | **safe (paper Table 1b, AUC 0.8157)** |
| rel-event | users-birthyear | users | 0/1731 | (test materialization 49 GiB pd.date_range OOM — separate bug) | needs `--full_graph` *and* RelBench fix |
| rel-event | event_interest-interested | event_interest | 0/536 | (same 49 GiB OOM) | same |
| rel-event | event_interest-not_interested | event_interest | 0/536 | (same 49 GiB OOM) | same |
| rel-hm | user-churn | users | 0 | 0 | **safe** |
| rel-hm | item-sales | items | 0 | 0 | **safe** |
| rel-hm | transactions-price | transactions | growing | growing | needs `--full_graph` |

Key correction from earlier drafts of this doc: **the RelGT paper does benchmark `user-attendance`, `user-repeat`, and `user-ignore` on rel-event** (Table 1a/1b in 2505.10960v2; dev-kyaw's `expts/run-hyperparam-sweep-small-experiments.sh` runs all three). Those tasks happen to be safe under truncation because their val/test seeds reference users that all existed before `train_cutoff` — even though `users` is nominally a growing table. Empirical OOB is the right check, not the seed-entity-name heuristic.

## The bug pattern (autocomplete tasks)

**Setup.** RelBench has:
- **Entity tables** (`users`, `events`, `results`, ...) — each row is one entity, often with its own creation timestamp.
- **Task seed tables** (per split) — labeled rows like `(entity_id, seed_time, label)` referencing an entity table.
- A **graph** built by linking entity tables via foreign keys.

**`get_db(upto_test_timestamp=True)`** (RelBench's default) drops every entity row whose own timestamp exceeds `train_cutoff`. The truncated entity tables drive the CSR adjacency build:

```python
# gfm_data/graph_cache.py
block.indptr.shape[0] == num_truncated_entities + 1
```

**The mismatch — autocomplete only.** Forecasting tasks ask "what will this *existing* entity do next?" — their seeds reference entities that already existed by `train_cutoff` and therefore live within the truncated table. Autocomplete tasks ask "an entity *just appeared* — predict its attribute"; by construction their val/test seeds index rows created after the cutoff. Concrete: `rel-f1.results-position` test seeds reference `resultId` 20323..26078, but the truncated `results` table has 20323 rows (valid ids 0..20322). Every test seed is OOB.

**The crash.** When the sampler hits `block.indptr[seed_idx + 1]` (`graph_cache.py:288`), `26079 >= 20324` → `IndexError`.

**Why the per-neighbor `seed_time` filter doesn't save us here.** That filter (`gfm_data/sampler.py:69`: `data[t].time[i] <= seed_time`) runs on *neighbors* AFTER the seed has been located. The seed itself can't be located in the first place — the indptr lookup crashes before any filtering.

## How the RelGT paper handles it

It doesn't, because it doesn't run autocomplete tasks. The paper benchmarks forecasting tasks on stable seed types: drivers, users, items, sites, studies, ads. Autocomplete tasks (`users-birthyear`, `results-position`, etc.) are new in **RelBench v2** and predate the RelGT paper's experiments. Dev-kyaw's adjacency (`utils.py:40-56`) has the same structural shape as our CSR — it would IndexError identically — but its experiment scripts only invoke forecasting tasks, so the bug never fires.

## What we do today

Two layers, both in this PR:

**Layer 1 — `--full_graph` flag (the actual fix).** Added to:
- `tools/precompute_shards.py`
- `tools/build_tf_store.py`
- `main_node_ddp.py` (and threaded into `train_multi_task.py::_load_dataset` via `args.full_graph`)

When set, the build flips `upto_test_timestamp=False` and writes its `make_pkey_fkey_graph` cache to `<cache_dir>/<dataset>/materialized_full/` (so the truncated build is not overwritten). The per-neighbor `seed_time` filter at `gfm_data/sampler.py:69` remains the leakage barrier — mathematically sufficient.

**Layer 2 — defensive bounds check in `graph_cache.neighbors_set`.** Out-of-bounds `src_idx` returns an empty set instead of IndexError. Belt-and-suspenders: if a user forgets `--full_graph`, the seed gets a seed-only sample (no neighbors) and the model predicts from static features only — degraded eval but not crashed.

The two holdout launchers (`scripts/holdout_task_dev.sh`, `scripts/holdout_dataset_eval.sh`) restrict their default task lists to the paper-benchmarked, empirically-safe subset:

| dataset | default tasks | excluded (require `--full_graph`) |
|---|---|---|
| rel-f1 | driver-position, driver-dnf, driver-top3 | results-position, qualifying-position, driver-circuit-compete |
| rel-event | user-attendance, user-repeat, user-ignore | users-birthyear, event_interest-* (also blocked by RelBench's pd.date_range OOM on test split) |
| rel-hm | user-churn, item-sales | transactions-price |

`driver-circuit-compete` is a link-prediction (recommendation) task with a different seed shape (driver × circuit pair); the paper doesn't benchmark it and we keep it out of defaults pending separate adoption work.

## Usage

Build shards + TF store for an autocomplete task:

```bash
python tools/build_tf_store.py \
    --dataset rel-event \
    --out_dir ~/.cache/relbench_examples/tf_store_full/rel-event \
    --full_graph

python tools/precompute_shards.py \
    --dataset rel-event --task users-birthyear \
    --K 300 --shard_size 50000 \
    --out_dir ~/.cache/relbench_examples/shards/rel-event/users-birthyear \
    --splits train val test \
    --name_prefix rel-event \
    --full_graph
```

Train (single-task) with `--full_graph`:

```bash
torchrun --nproc_per_node 1 main_node_ddp.py \
    --dataset rel-event --task users-birthyear \
    --mode precomputed_shards \
    --shards_dir ~/.cache/relbench_examples/shards/rel-event/users-birthyear \
    --tf_store_dir ~/.cache/relbench_examples/tf_store_full \
    --full_graph \
    ...
```

Multi-task with `--full_graph` mixed in:

```bash
torchrun --nproc_per_node 8 main_node_ddp.py \
    --tasks "rel-event.user-attendance:1.0,rel-event.users-birthyear:1.0" \
    --mode precomputed_shards \
    --tf_store_dir ~/.cache/relbench_examples/tf_store_full \
    --full_graph \
    ...
```

## What does NOT change

- `gfm_data/sampler.py` — the `time` filter at line 69 is unchanged; it remains the temporal leakage barrier.
- `gfm_data/task_tokens.py`, model code, losses — unchanged.
- Default behavior — without `--full_graph`, everything works as before.
- Adoption tooling (`tools/extract_embeddings.py`, `tools/finetune_head.py`, `tools/tabpfn_eval.py`) does not need a flag because it consumes pre-built shards/TF stores.

## Verification

For each affected task, after building with `--full_graph`:
1. All three splits complete shard-build without IndexError.
2. Shard counts match RelBench's reported split sizes.
3. For a sampled test seed with `seed_time = T`, no neighbor in the subgraph has `time > T` (the leakage check).

## Layered fix history (for reference)

The earlier drafts of this doc considered three layered fixes:
- **Layer 1 — defensive bounds check.** Implemented in this PR.
- **Layer 2 — per-task `--full_graph` opt-in.** Implemented in this PR.
- **Layer 3 — re-key seeds.** Map seed ids from full-DB scope to truncated-table scope via a lookup so out-of-bounds seeds get a sentinel index. More invasive than 1+2 and unnecessary now that 2 is in.

## References

- RelBench v2 task taxonomy (`relbench.tasks.get_task_names`).
- RelGT paper (arXiv 2505.10960v2), Tables 1a/1b — confirms the paper benchmarks user-attendance, user-repeat, user-ignore.
- dev-kyaw `expts/run-hyperparam-sweep-small-experiments.sh` — runs all three rel-event tasks.
- `gfm_data/sampler.py:69` — per-neighbor `seed_time` filter (the actual leakage barrier).
- `gfm_data/graph_cache.py:274` — `neighbors_set` (Layer 1 bounds check + indptr lookup).
