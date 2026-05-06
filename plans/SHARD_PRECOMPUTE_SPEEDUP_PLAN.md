# Speed up Phase-2 shard precompute on p4d (FINAL)

## Context

`scripts/holdout_dataset_eval.sh` currently takes 5+ days during Phase-2 shard
precompute (delegated to `scripts/pretrain_p4d.sh`). For a Phase-5 LOO sweep
across 9 RelBench v2 datasets we build ~33 (ds, task) × 3 splits = ~99 split
shards, with rel-event train (~13M seeds) and rel-amazon train (~10M seeds)
forming the long pole.

The bottleneck is per-seed Python overhead in `gfm_data/sampler.py` and
`gfm_data/graph_cache.py`, not I/O or orchestration. At K=300 with 1-hop cap
5000, every seed performs:

- ~5001 calls to `cache.neighbors_set()` (1 for the seed, ≤5000 for each kept
  1-hop) plus ~K more during edge construction.
- Per-neighbor PyTorch tensor indexing + `.item()` for the time filter.
- A Python `for k in range(end-start): out.add((str_lookup[int(t[k])], int(i[k])))`
  loop inside `neighbors_set`, which boxes ~10k Python objects per high-degree
  node.

The 1-hop cap of 5000 and 2-hop cap of 1000 are inherited verbatim from
dev-kyaw's `gather_1_and_2_hop_with_seed_time` (the original RelGT
implementation). Our `gfm_data/sampler.py` is a line-for-line port preserving
them; tests S2/S3 in `tests/test_csr_sampler.py` verify they fire correctly.
This plan keeps them as-is — changing them is a separate ablation, not a
correctness or perf fix.

**Hard constraint:** dev-kyaw is the spec. Every optimization in this plan is
bit-exact: it produces the same Python `set` / `list` objects at each
`random.sample` / `random.choices` call site, with the same elements as
today, so CPython's hash-determined iteration order is preserved and RNG
outputs are unchanged. The full S1-S19 test suite (existing S1-S17 + new
S18-S19 below) must continue to pass.

The plan is split into three phases:

- **Phase 0 (baseline verification)** — confirm dev-kyaw == current
  pre-optimization sampler on real relbench data. ~30 min one-time.
- **Phase 1 (per-process speedup)** — rewrites the Python hot path. ~5-8x per
  builder. Bit-exact. Self-contained changes in `gfm_data/`.
- **Phase 2 (long-pole orchestration)** — range-partitions the slowest task
  across multiple builder processes. Additional ~3-6x on the long pole.
  Bit-exact. Touches `tools/precompute_shards.py`, `gfm_data/shard_io.py`,
  `scripts/pretrain_p4d.sh`. **Gated on Phase 1 not hitting the sub-day
  target.**

Combined target: 5+ days → sub-day Phase-2 wall on p4d.24xlarge with no
change in pretrain/holdout test metrics.

---

## Phase 0 — baseline verification (do first)

Before touching any code, confirm that our current port is already
bit-equivalent to dev-kyaw on **real** relbench data, not just on the
synthetic graphs S12-S17 cover.

### Step 0.1 — add S19 (rel-f1 real-data dev-kyaw equivalence)

Add `test_S19_dev_kyaw_equivalent_on_rel_f1` to
`tests/test_csr_sampler.py`.

What it does:
1. Materialize rel-f1 via the same `make_pkey_fkey_graph` call site as
   `tools/precompute_shards.py:121`.
2. Build adjacency BOTH ways: dev-kyaw's `build_adjacency_hetero(data)` and
   the new `DatasetGraphCache(data)`.
3. Take 200 train seeds from `rel-f1.driver-top3`.
4. For K ∈ {16, 64, 300}, for each seed:
   - Compute `seed_val = hash((seed_type, idx, time, K)) & 0xFFFFFFFF`
   - Run dev-kyaw `_process_one_seed(...)`
   - Run new `sample_local_subgraph(...)`
   - Assert `_multiset(final_tokens_dev) == _multiset(final_tokens_new)`
   - Assert `np.array_equal(edge_index_dev, edge_index_new)`

Marked `@pytest.mark.slow` (rel-f1 materialization is ~30s). Run via
`pytest -m slow tests/test_csr_sampler.py::test_S19_dev_kyaw_equivalent_on_rel_f1`.

Coverage: 200 seeds × 3 K values = 600 real-data head-to-heads.

### Step 0.2 — add S20 (rel-event extended check, optional/manual)

Same pattern as S19 but on rel-event with 100 seeds at K=300. Rationale:
rel-event is the production long pole and stresses the 5000-cap path that
rel-f1 rarely hits. Marked `@pytest.mark.slow_xl` (excluded from default
runs because rel-event materialization is ~25 GB peak); run manually on
p4d after Phase 1 lands.

### Phase 0 acceptance gate

Run `pytest -m slow tests/test_csr_sampler.py -v` and confirm S1-S19 all
pass on the current pre-optimization code. If S19 fails, the existing port
already differs from dev-kyaw on real data — that's a separate bug and
must be fixed before Phase 1.

---

## Phase 1 — bit-exact per-process optimizations

### Step 1.2 (FIRST) — vectorize `neighbors_set` set construction

`gfm_data/graph_cache.py:274-306`. Build `self._type_str_by_id` once in
`__init__` as an object array indexed by int16 type id:

```python
self._type_str_by_id = np.empty(len(self.node_types), dtype=object)
for i, t in enumerate(self.node_types):
    self._type_str_by_id[i] = t  # prefixed string
```

Rewrite `neighbors_set` body (lines 294-306):

```python
block = self.csr[src_type]
if src_idx < 0 or src_idx + 1 >= block.indptr.shape[0]:
    return set()
start = int(block.indptr[src_idx])
end = int(block.indptr[src_idx + 1])
if end == start:
    return set()
nt_strs = self._type_str_by_id[block.nbr_type_id[start:end]].tolist()
nidx_list = block.nbr_idx[start:end].tolist()
return set(zip(nt_strs, nidx_list))
```

`numpy.ndarray.tolist()` returns Python `int`s (matching the existing
`int(nidx[k])` behavior). `set(zip(...))` is a C-level constructor producing
the identical `(str, int)` tuple set.

**Bit-exact:** identical-element sets in CPython hash to the same buckets
regardless of insertion order, so `list(set)` traversal in the sampler is
unchanged.

**Estimated speedup:** 3-5x on `neighbors_set`. This single change is the
biggest fraction of gather-phase Python overhead.

**Verify:** S1-S19 + Tier 1 byte-equality on rel-f1 driver-top3.

### Step 1.1 — pre-extract per-type time arrays + has_time flags

`gfm_data/graph_cache.py:115-148`. In `DatasetGraphCache.__init__`, build
once at construction:

```python
self._time_by_prefixed: Dict[str, Optional[np.ndarray]] = {}
self._has_time_by_prefixed: Dict[str, bool] = {}
for raw_t in raw_types:
    pt = self._with_prefix(raw_t)
    if hasattr(data[raw_t], "time"):
        ta = data[raw_t].time
        if isinstance(ta, torch.Tensor):
            ta = ta.cpu().numpy()
        self._time_by_prefixed[pt] = ta
        self._has_time_by_prefixed[pt] = True
    else:
        self._time_by_prefixed[pt] = None
        self._has_time_by_prefixed[pt] = False
```

In `gfm_data/sampler.py` rewrite the time-filter sites at lines 67-72,
85-90, 96-103, 105-112, 162-169:
- Replace `data[raw_nbr_t].time[nbr_i].item()` with
  `cache._time_by_prefixed[nbr_t][nbr_i]` (numpy float scalar; same value as
  `.item()` for finite RelBench timestamps).
- Replace `hasattr(data[raw_nbr_t], "time")` with
  `cache._has_time_by_prefixed[nbr_t]`.
- Drop `cache.prefixed_to_raw[...]` lookups in the hot path.

**Bit-exact:** `np.float32 <= float` and `torch.float32 <= float` produce
identical booleans for finite RelBench timestamps; `(seed_time - nbr_time)
/ 86400` yields the same Python float regardless of tensor vs numpy origin.

**Estimated speedup:** 2-3x on the gather phase.

**Verify:** S1-S19 + Tier 1 byte-equality on rel-f1 driver-top3.

### Step 1.3 — per-seed `neighbors_set` cache in `sample_local_subgraph`

`gfm_data/sampler.py:182-198`. Today line 189 calls
`cache.neighbors_set(t_str, i)` for each of the K final tokens. The seed
and every kept 1-hop were already fetched during `gather_1_and_2_hop`.

Build a per-seed `dict[(t_str, i) -> set]` cache in `gather_1_and_2_hop`,
return it alongside the token list, reuse during edge construction. Fall
back to `cache.neighbors_set` for 2-hop / fallback tokens that weren't
fetched.

**Bit-exact:** the cached set is the same Python object that was returned
on the first call; its hash buckets and iteration order are identical, so
the edge-building loop sees the same neighbors in the same order.

**Estimated speedup:** 1.3-1.5x end-to-end.

**Verify:** S1-S19 + Tier 1 byte-equality on rel-f1 driver-top3.

### Step 1.4 — vectorize the time filter via per-type bucketing

`gfm_data/sampler.py:65-72, 75-90`. Replace per-element loops with per-type
bucketed numpy comparisons:

```python
# n1_full is a list of (prefixed_type, idx) tuples (post-cap, post-list)
by_type: Dict[str, List[int]] = defaultdict(list)
positions: Dict[str, List[int]] = defaultdict(list)  # original list pos
for pos, (nbr_t, nbr_i) in enumerate(n1_full):
    by_type[nbr_t].append(nbr_i)
    positions[nbr_t].append(pos)
keep_mask = [True] * len(n1_full)
for nbr_t, idxs in by_type.items():
    if cache._has_time_by_prefixed[nbr_t]:
        ta = cache._time_by_prefixed[nbr_t]
        sub_mask = ta[np.asarray(idxs)] <= seed_time
        for j, ok in enumerate(sub_mask.tolist()):
            if not ok:
                keep_mask[positions[nbr_t][j]] = False
n1: Set[Tuple[str, int]] = set()
for ok, pair in zip(keep_mask, n1_full):
    if ok:
        n1.add(pair)
```

Same pattern for the 2-hop filter (75-90): bucket the flat list, vectorize
the `<=` comparison, then walk survivors in original iteration order to
build the dedup `defaultdict`.

**Bit-exact:** `n1` and `n2` end up with the same elements as today; sets
are insertion-order-irrelevant; the `n2` `defaultdict` is built from the
same survivor iteration, producing identical `connecting_1hops` sets.

**Estimated speedup:** 1.5-2x on the gather phase (compounds with 1.1+1.2).

**Verify:** S1-S19 + Tier 1 byte-equality on rel-f1 driver-top3.

### Step 1.5 — add S18 (post-optimization regression snapshot)

After 1.1-1.4 land, add `test_S18_optimized_path_bit_equivalent` to
`tests/test_csr_sampler.py`:
- Pickle a small `DatasetGraphCache` (synthetic dense graph, e.g. the
  S16 builder).
- Run the optimized sampler over 1000 seeds.
- Assert `np.array_equal` against a reference snapshot saved on disk
  (committed alongside the test as a small pickle / npz).

This is a defense-in-depth check: S19 catches dev-kyaw drift on real data;
S18 catches drift between optimized vs the snapshot of "known good"
post-optimization output. If a future refactor breaks the optimized path
in a way that compensates between gather and select (so S1-S17 stays
green), S18's frozen snapshot catches it.

### Phase 1 cumulative estimate

Per-process: ~5-8x. Phase 2 wall: 5 days → ~15-24 hours.

### Phase 1 acceptance gate

1. `pytest -m slow tests/test_csr_sampler.py -v` — S1-S19 green.
2. Tier 1 ML sanity: byte-equality of shards on rel-f1 driver-top3 (see
   Verification below).
3. Tier 2 ML sanity: 1-hour multi-task pretrain shake-out (see
   Verification below).

If 2 fails, optimization leaks something the unit tests miss — STOP and
investigate. If 3 metrics drift outside seed noise, optimization changes
the data distribution despite byte-equality (memory layout, alignment,
edge cases) — STOP and investigate.

---

## Phase 2 — long-pole orchestration (only if Phase 1 is insufficient)

### Step 2.1 — range-partition a single big task across builders

Today `phase2_build_one` in `scripts/pretrain_p4d.sh:411-453` runs ONE
Python process per (ds, task), with `SHARD_WORKERS=10` fork-workers
inside. With `PARALLEL_SHARD_BUILDS=8`, the long-pole task gates
everything: even after Phase 1 makes each task ~5-8x faster, rel-event
train alone may still be the slowest single task by a wide margin.

**Implementation pieces:**

- `tools/precompute_shards.py`: add `--seed_range_lo` (default 0),
  `--seed_range_hi` (default = total_samples), `--write_meta` (default
  true; partition workers pass false). The `precompute_split` loop owns
  only `for s_idx in shards_in_range(lo, hi)` and skips
  `writer.finalize()` when `--write_meta=false`.
- `gfm_data/shard_io.py`: relax `ShardWriter`'s "all shards written"
  assertion to support partial writers (e.g.
  `expected_shards: Optional[List[int]]`); add a small standalone
  `write_meta(root, K, total_samples, shard_size, num_shards)` helper for
  the parent's final no-op invocation.
- `scripts/pretrain_p4d.sh`: in `phase2_build_one`, peek the task's split
  sizes; if any split exceeds `BIG_TASK_THRESHOLD` (e.g., 5M seeds),
  dispatch `BIG_TASK_PARTITIONS` (default 8) parallel sub-builders each
  with disjoint `--seed_range_lo/--seed_range_hi`. After all sub-builders
  return, write `meta.json` once.
- New env var: `BIG_TASK_PARTITIONS` (default 8 on p4d).

**Bit-exact:** every seed's RNG depends only on
`(seed_type, node_idx, seed_t, K)` — worker assignment is irrelevant.
Each shard's bytes are written by exactly one process to a path
determined by `shard_idx`; no seam, no ordering hazard.

**Estimated speedup:** 3-6x on the long pole. Total proc count needs to
stay near 96 vCPUs, so `SHARD_WORKERS` may need to drop while a
partitioned task is running.

### Step 2.2 — confirm parallelism dial settings

After Phase 1, validate on a 1-hour smoke test that
`PARALLEL_SHARD_BUILDS=8`, `SHARD_WORKERS=10` are still optimal. The
per-process speedup may shift the optimum. Sweep ±2 around each and pick
the fastest combination that fits in 1 TB RAM. No code change unless
empirics demand it.

### Phase 2 acceptance gate

1. **Byte-equality with vs without partitioning:** build a smoke task's
   shards (a) unpartitioned (b) with `BIG_TASK_PARTITIONS=4`. `diff -r`
   must be empty.
2. **Long-pole wall test:** time rel-event train shard build before
   Phase 2 vs after. Target ≥3x speedup on this single task.
3. **End-to-end Tier 3 ML sanity:** full LOO holdout smoke (see
   Verification below).

---

## Verification suite

### Unit tests (run after each Phase-1 step)

| Test file | Purpose |
|---|---|
| `tests/test_csr_sampler.py::S1`-`S17` | Existing synthetic dev-kyaw equivalence |
| `tests/test_csr_sampler.py::S18` (new in 1.5) | Post-optimization regression snapshot |
| `tests/test_csr_sampler.py::S19` (new in 0.1) | dev-kyaw equivalence on real rel-f1, 200 seeds × 3 K values |
| `tests/test_csr_sampler.py::S20` (new in 0.2, optional) | dev-kyaw equivalence on rel-event |
| `tests/test_csr_adjacency.py` | CSR layout regression |
| `tests/test_precompute_shards_prefix.py` | `--name_prefix` flow |
| `tests/test_shard_io.py` | ShardWriter/Reader roundtrip |
| `tests/test_shard_build_drops_tf.py` | TF columns dropped post-load |
| `tests/test_shard_type_remap.py` | Type id remapping at training time |

Command per step:
```bash
pytest tests/test_csr_sampler.py tests/test_csr_adjacency.py \
       tests/test_precompute_shards_prefix.py tests/test_shard_io.py \
       tests/test_shard_build_drops_tf.py tests/test_shard_type_remap.py -v
# Once at start of Phase 0 + once at Phase 1 acceptance gate, also include:
pytest -m slow tests/test_csr_sampler.py::test_S19_dev_kyaw_equivalent_on_rel_f1 -v
```

### ML sanity (Tier 1, 2, 3)

**Tier 1 — shard byte-equality on real data (~5 min).** Strongest possible
check. Run after each Phase-1 step (cheap, bit-exact safety net):

```bash
# Reference build (run once before Phase 1)
python tools/precompute_shards.py --dataset rel-f1 --task driver-top3 \
  --K 300 --shard_size 50000 --splits train val test \
  --out_dir /tmp/shards_pre_phase1 --name_prefix rel-f1

# After applying the step
python tools/precompute_shards.py --dataset rel-f1 --task driver-top3 \
  --K 300 --shard_size 50000 --splits train val test \
  --out_dir /tmp/shards_after_step --name_prefix rel-f1

diff -r /tmp/shards_pre_phase1 /tmp/shards_after_step
# Must be empty
```

**Tier 2 — multi-task pretrain parity (~1 hour, end of Phase 1):** confirms
training metrics don't drift.

```bash
NPROC=8 EPOCHS=3 STEPS_PER_TASK=200 PRETRAIN_SEEDS="0" \
  bash scripts/pretrain_only_seed_sweep.sh
```

Compare per-task test metrics in
`results/pretrain_only_seed_sweep/<slug>/aggregate.json` against the
baseline in `results/20260506_pretrain-3trial-results.md`. Per-metric
within seed-noise (1× std). If Tier 1 was clean, this is a tautology, but
keep it as a paranoid backstop.

**Tier 3 — full LOO holdout smoke (~1 day, only if Phase 2 lands):**

```bash
NPROC=8 SOURCE="rel-f1 rel-event" TARGET=rel-arxiv \
  EPOCHS=3 STEPS_PER_TASK=200 RUN_TABPFN=0 \
  bash scripts/holdout_dataset_eval.sh
```

Compare holdout metrics against
`results/20260506_holdout-dataset-eval-rel-f1-event-to-arxiv.md` within
seed noise.

### Intuition checks (run end of Phase 1)

1. **Per-call microbench** — pick a high-degree rel-event seed (e.g. an
   event with thousands of attendees):
   ```python
   import timeit
   # Before / after step 1.2:
   t = timeit.timeit(
       lambda: cache.neighbors_set("rel-event::events", popular_id),
       number=10000,
   )
   ```
   Expect ≥3x speedup on `neighbors_set`. If <2x, we've optimized the
   wrong thing — pause and re-profile.

2. **Per-seed gather wall** — same idea, calling `gather_1_and_2_hop(...)`
   directly. Expect 3-5x after step 1.2; 5-8x after all four steps.

3. **End-to-end split wall on long pole** — `time
   tools/precompute_shards.py --dataset rel-event --task user-attendance
   --splits train`. Expect ≥5x. This is the production wall-clock
   metric — must hit ≥5x for the 5-days-to-sub-day claim.

4. **Data structure spot-checks:**
   - `cache._time_by_prefixed[t].shape == (cache._num_nodes_of(data, t.split('::')[-1]),)`
     for every type with time.
   - `cache._time_by_prefixed[t][i] == data[raw_t].time[i].item()` on
     5 spot-checked indices.
   - `cache._type_str_by_id[i] == cache.index_to_node_type[i]` for every
     type id.

5. **Memory regression** — `psutil.Process().memory_info().rss` increase
   from `DatasetGraphCache.__init__` should stay <500 MB above the
   pre-optimization baseline. (Time arrays add
   `~sum(num_nodes_with_time) × 4` bytes; rel-event ~160 MB.)

6. **Pickle/fork sanity** — pickle a `DatasetGraphCache`, unpickle, run
   the sampler over a few seeds; output identical to non-pickled. Confirms
   `--workers >1` (fork+COW) still works.

7. **Seed determinism** — for a fixed `random.seed(42)` and K=300, run
   `sample_local_subgraph` on the same seed before vs after. Output must
   be byte-identical (this is what S18's snapshot codifies).

---

## Critical files to modify

| File | Phase / Step | Lines |
|---|---|---|
| `gfm_data/graph_cache.py` | 1.1, 1.2 | 115-148 (`__init__`), 274-306 (`neighbors_set`) |
| `gfm_data/sampler.py` | 1.1, 1.3, 1.4 | 34-114 (`gather_1_and_2_hop`), 118-198 (`sample_local_subgraph`) |
| `tests/test_csr_sampler.py` | 0.1, 0.2, 1.5 | append S18, S19, S20 |
| `tools/precompute_shards.py` | 2.1 | 44-89 (parse_args), 210-316 (`precompute_split`) |
| `gfm_data/shard_io.py` | 2.1 | 82-154 (`ShardWriter`); add `write_meta` helper |
| `scripts/pretrain_p4d.sh` | 2.1 | 411-453 (`phase2_build_one`), 522-540 (Phase-2 dispatch) |

---

## Out of scope

- **Numba/Cython JIT.** Numba's RNG differs from CPython Mersenne Twister;
  a pure-Python rewrite cannot share the per-seed `random.seed(seed_val)`
  semantics. Revisit only if Phase 1+2 are insufficient.
- **Streaming sampler at training time** as a precompute alternative.
  Streaming pays the sampler cost every epoch; precompute pays once.
  EPOCHS=10 would multiply the cost ~10x.
- **Lowering `max_1hop_threshold` / `max_2hop_threshold`.** dev-kyaw
  spec'd 5000/1000; lowering them is a separate ablation.
- **Removing the caps to match upstream RelGT paper.** The user confirmed
  dev-kyaw is the spec, so the caps stay.
- **Distributing the build across multiple p4d nodes.** Single-node only.

---

## Suggested execution order

1. **Phase 0.1** — write S19 + run as a baseline check on current
   pre-optimization code. Confirm dev-kyaw == current on real rel-f1.
2. **Phase 0.2** (optional) — write S20 and run on rel-event manually.
3. **Phase 1.2** (`neighbors_set` rewrite) — biggest single win, smallest
   blast radius. Run S1-S19 + Tier 1 byte-diff on rel-f1.
4. **Phase 1.1** (time array pre-extraction) — pairs with the sampler-side
   rewrites in 1.4. Run S1-S19 + Tier 1.
5. **Phase 1.3** (per-seed cache) — small refactor, easy correctness check.
   Run S1-S19 + Tier 1.
6. **Phase 1.4** (vectorized time filter) — last so it builds on data
   structures from 1.1+1.2. Run S1-S19 + Tier 1.
7. **Phase 1.5** — add S18 snapshot regression test.
8. **Phase 1 acceptance gate:** run the full unit suite + Tier 2 ML
   sanity. Time the long-pole split wall on rel-event. If ≥5x and Tier 2
   metrics within seed noise, ship Phase 1 as a single PR.
9. **Stop if Phase 1 hits sub-day target.** Otherwise, proceed to
   Phase 2 — gated on real measurement, not speculation.
10. **Phase 2.1** — partitioned builders. Run S1-S19 + byte-equality with
    vs without partitioning + Tier 3 ML sanity on rel-arxiv smoke.
11. **Phase 2.2** — empirical sweep of `PARALLEL_SHARD_BUILDS` /
    `SHARD_WORKERS`. Pick the empirical winner.
