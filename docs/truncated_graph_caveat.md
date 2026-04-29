# Truncated-graph caveat — RelGT paper guardrail vs growing-seed tasks

**TL;DR**: We keep `upto_test_timestamp=True` (the RelGT paper's defense-in-depth guardrail; PR 1.0b). That guardrail crashes the sampler on tasks whose seed entity grows over time, because val/test seed ids reference rows that the truncated entity table doesn't contain. The RelGT paper itself sidesteps this by avoiding such tasks in its benchmark. We do the same for now: defaults are pinned to the paper-benchmarked subset. To opt into a growing-seed task, override `PRETRAIN_TASKS_CSV` / `SOURCE_TASKS_CSV` / `TARGET_TASKS_CSV` and accept that the build will crash unless the data layer is also flipped to `upto_test_timestamp=False`.

## The bug pattern

**Setup.** A RelBench v2 dataset has:
- **Entity tables** — `users`, `events`, `races`, `results`, `transactions`, ... — each row is one entity, often with its own creation timestamp.
- **Task seed tables** — labeled rows like `(user_id, seed_time, label)` where `user_id` references an entity table.
- A **graph** built by linking entity tables via foreign keys.

**`get_db(upto_test_timestamp=True)`** (RelGT paper guardrail; our PR 1.0b default) drops every entity row whose own timestamp exceeds `train_cutoff`. The truncated entity tables drive the CSR adjacency build:

```python
# graph_cache.py
block.indptr.shape[0] == num_truncated_entities + 1
```

**The mismatch.** Val/test seed tables are NOT truncated — RelBench keeps them whole because they're the actual evaluation set. Their `seed_idx` values are entity ids assigned at full-DB scope. So a val seed can have `seed_idx = 50000` even when the truncated entity table only has `30000` rows.

When the sampler hits `block.indptr[seed_idx + 1]` (`graph_cache.py:288`), `50000+1 >= 30001` → `IndexError`.

**Why the per-neighbor `seed_time` filter doesn't save us.** The filter (`gfm_data/sampler.py:69`: `data[t].time[i] <= seed_time`) runs on *neighbors* AFTER the seed has been successfully located. The seed itself can't be located in the first place — the lookup crashes before any filtering.

**Affected task types.** Any task whose seed sits on a *growing* entity type. Confirmed cases on our laptop runs:
- `rel-event.users-birthyear` — seed=user, users grow with sign-ups.
- `rel-f1.results-position` — seed=result row, new race results every weekend.
- Likely (not yet hit but same shape):
  - `rel-f1.qualifying-position` — seed=qualifying row.
  - `rel-hm.transactions-price` — seed=transaction.

## How the RelGT paper handles it

It doesn't. **The paper sidesteps the bug by not benchmarking on growing-seed tasks.**

Confirmed by reading dev-kyaw's `utils.py:40-56`:

```python
adjacency = {
    node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
    for node_type in hetero_data.node_types
}
```

dev-kyaw's adjacency is a Python `dict[type] -> list[set]`, indexed by `adjacency[node_type][node_idx]` (line 82). No bounds check anywhere. Identical structural shape to our CSR `indptr` — would crash with the same `IndexError` on out-of-bounds.

The paper's published benchmark task list (`expts/run-large-base-experiments.sh`, `expts/run-hyperparam-sweep-small-experiments.sh`) sticks to tasks where the seed entity is **stable** over time:

| dataset | paper-benchmarked tasks | seed entity | growing? |
|---|---|---|---|
| rel-amazon | user-churn, item-churn, user-ltv, item-ltv | user / item | stable |
| rel-stack | user-engagement, user-badge, post-votes | user / post | stable |
| rel-hm | user-churn, item-sales | user / item | stable |
| rel-event | user-ignore | user×event composite | stable |
| rel-trial | site-success, study-outcome, study-adverse | site / study | stable |
| rel-avito | ad-ctr, user-clicks, user-visits | ad / user | stable |
| rel-f1 | driver-dnf, driver-position, driver-top3 | driver | stable |

The tasks the paper avoids: `users-birthyear`, `results-position`, `qualifying-position`, `transactions-price`, `user-attendance`, `user-repeat`. All on growing seed entities.

## What we do today

The two holdout launchers (`scripts/holdout_task_dev.sh`, `scripts/holdout_dataset_eval.sh`) restrict their default task lists to the paper-benchmarked subset:

| dataset | default tasks (this repo's launchers) | omitted (growing seed) |
|---|---|---|
| rel-f1 | driver-position, driver-dnf, driver-top3 | results-position, qualifying-position, driver-circuit-compete |
| rel-event | user-ignore | user-attendance, user-repeat, event_interest-interested, event_interest-not_interested, users-birthyear |
| rel-hm | user-churn, item-sales | transactions-price |

Notes:
- **rel-event has only one paper-benchmarked task (`user-ignore`)**. "All-but-one" pretrain doesn't apply since the holdout would consume the only available task. We omit rel-event from the **default** `DATASETS` of `holdout_task_dev.sh` for now; users wanting it can override `DATASETS="rel-f1 rel-event rel-hm"` and accept that pretrain-on-rel-event reduces to "include 0 rel-event tasks alongside rel-f1 + rel-hm" — equivalent to dropping rel-event entirely.
- **`driver-circuit-compete` is dropped despite stable seed (driver × circuit)** out of caution: the paper doesn't benchmark it, so we don't have an external sanity check that it works under the truncated graph.

## How to opt into an affected task

If you want to evaluate a paper-omitted growing-seed task, two routes:

**1. Override the CSV explicitly** — *crashes unless the data layer is also flipped*:
```bash
PRETRAIN_TASKS_CSV="rel-event.users-birthyear:1.0,rel-event.user-ignore:1.0" \
  bash scripts/holdout_task_dev.sh
# IndexError at graph_cache.py:288 -- expected.
```

**2. Override the CSV AND flip `upto_test_timestamp=False`** — works, but loses the paper guardrail. Affects: `gfm_data/stypes.py` (default), `main_node_ddp.py`, `train_multi_task.py`, `tools/build_tf_store.py`, `tools/precompute_shards.py`. Per-neighbor `seed_time` filter at `gfm_data/sampler.py:69` remains the leakage barrier (mathematically sufficient; see PR 1.0b discussion).

## Code-fix options (future work)

We could move past the paper-task subset with these changes (not yet implemented):

**Layer 1 — defensive bounds-check in `graph_cache.neighbors_set`.** Return empty set + warn-once on out-of-bounds. Build completes for every task. Affected seeds get a seed-only sample (no neighbors); model predicts from the seed's static features only. ~5 lines. Strictly more capable than dev-kyaw, no leakage risk. Eval metric on those seeds is degraded but not crashed.

**Layer 2 — per-task `--upto_test_timestamp false` opt-in.** Add a CLI flag to `tools/precompute_shards.py` and `tools/build_tf_store.py`. Default stays True. For tasks where Layer 1's "no-context" approximation is too lossy, the user opts into the full-graph build per-task; the per-neighbor `seed_time` filter remains the only barrier (correct).

**Layer 3 (optional) — re-key seeds.** Map seed ids from full-DB scope to truncated-table scope via a lookup. Out-of-bounds seeds get a sentinel index that the encoder treats as "novel". More invasive than Layer 1+2.

Combined Layer 1+2 would let us run `users-birthyear` / `results-position` / `transactions-price` end-to-end without crash, with a clear knob (`--upto_test_timestamp`) for users to choose between "stricter guardrail, degraded eval" and "looser guardrail, real eval".

## References

- PR 1.0b commit message (`353e4d9`) — original caveat documentation.
- dev-kyaw `utils.py:40-56` — the adjacency-build that has the same bug shape.
- RelGT paper expts:
  - `expts/run-large-base-experiments.sh`
  - `expts/run-hyperparam-sweep-small-experiments.sh`
- `gfm_data/sampler.py:69` — per-neighbor `seed_time` filter (the actual leakage barrier).
- `graph_cache.py:288` — `block.indptr[src_idx + 1]` (the IndexError site).
