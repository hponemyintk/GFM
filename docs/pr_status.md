# Multi-task refactor — PR status & results

Single source of truth for what has landed on this branch and what's next. The architectural plan lives in [`multi_task_refactor_plan.md`](multi_task_refactor_plan.md); the empirical parity numbers live in [`parity_results.md`](parity_results.md).

---

## Status board

| PR | Title | State | Commits |
|---|---|---|---|
| **PR1** | Refactor preprocessing into `gfm_data/` with CSR adjacency | ✅ merged | `d4d33cd` |
| **PR2** | Memmap shards + memmap TFs + streaming/precomputed modes | ✅ merged | `372c82a`, `6a557e5`, `93f506b`, `31f4474` |
| **PR3** | Multi-task plumbing (rel-f1 first) | ✅ merged | `d49376c`, `4570fdb` |
| **PR4** | Cross-dataset namespacing + 6-task launcher | ✅ scaffolding merged | this commit |

197 unit tests green, 3 real-data smokes green, 1 ML parity sweep green, 1 multi-task ML4-ML7 smoke green.

---

## PR1 — preprocessing refactor (`d4d33cd`)

**Goal.** Replace `dict[node_type] -> list[set]` adjacency in `utils.py` with a memory-efficient CSR layout, while preserving dev-kyaw sampler semantics bit-for-bit under fixed `random.seed`.

**New code (926 LOC across 5 files):**

```
gfm_data/__init__.py
gfm_data/graph_cache.py    # CSR per source-type, optional dataset-name prefix
gfm_data/sampler.py        # gather_1_and_2_hop + sample_local_subgraph
gfm_data/task_tokens.py    # per-(task, split) Dataset wrapping the cache
gfm_data/collate.py        # free-function collate; adds task_id / task_type_id
```

`main_node_ddp.py` rewired to use the new package; single-task behavior unchanged.

**Memory savings.** Per-edge cost goes from ~80 B (Python set of tuples) to ~12 B (CSR int16 type id + int32 idx + int64 indptr). For rel-event (~165M undirected edges) that's a ~12 GB reduction.

**Equivalence vs dev-kyaw — three layers of evidence:**

1. **Unit (toy graphs):** 26 tests in `tests/test_csr_adjacency.py` (C1–C7), `tests/test_csr_sampler.py` (S1–S17), `tests/test_collate_single_task.py` (Co5). S12–S17 cover the four selection regimes (`size_th > K-1`, `= K-1`, `0 < size_th < K-1`, `= 0`) including duplicate-presence in the `random.choices` regime.
2. **Real rel-f1 (`tests/smoke_rel_f1.py`):** 1677 (type, idx) pairs across 9 node types and 200 driver-top3 seeds — **zero mismatches** vs dev-kyaw.
3. **End-to-end ML parity:** see PR2 close-out below.

**Key design call.** The new sampler still builds Python `set` objects for gather-phase dedup. CPython's set iteration is hash-determined and identical for identical-element sets, so under fixed `random.seed` `random.sample` returns the same draws as dev-kyaw. The CSR is purely a memory-efficient *storage* layer feeding the same sampling algorithm.

---

## PR2 — memmap layer + automation (`372c82a` + `6a557e5` + `93f506b` + `31f4474`)

**Goal.** Disk-back the two largest in-RAM data structures (sample tokens + TensorFrame columns) so resident memory stops scaling with dataset size. Unlocks rel-event and beyond.

**New code:**

```
gfm_data/shard_io.py       # ShardWriter / ShardReader (memmap layout)
gfm_data/tf_store.py       # build_tf_store / TFStoreReader (per-stype memmaps)
tools/precompute_shards.py # offline shard builder
tools/build_tf_store.py    # offline TF memmap builder
scripts/memory_smoke.sh    # RSS + VRAM smoke (MS1 / MS3)
scripts/parity_sweep.sh    # 5×2×2 sweep driver
scripts/aggregate_parity.py # markdown-table + acceptance check
```

`DatasetGraphCache` gained an optional `tf_store_root` and a `tf_view(...)` method that dispatches to either the in-RAM `data[type].tf` (default) or the `TFStoreReader`, so collate / TaskTokens are mode-agnostic.

`TaskTokens` now supports three modes:

| mode | precompute | RAM-bound? | use |
|---|---|---|---|
| `hdf5` (default) | yes, in-process | no (TFs in RAM) | dev-kyaw parity / single-dataset |
| `streaming` | none | yes (TFs can be memmap) | laptop / iteration |
| `precomputed_shards` | yes, offline tool | yes (TFs can be memmap) | 8×A100 / production |

**Memmap TF schema** mirrors `torch_frame.TensorFrame`:

```
<root>/<table>/
  meta.json
  numerical.float32                # [N, C_num]
  categorical.int64                # [N, C_cat]
  timestamp.int64                  # [N, C_ts, 7]
  embedding.values.f32             # [N, total_emb_dim]
  embedding.offset.i64             # [C_emb + 1]
  multicategorical.values.i64      # 1D
  multicategorical.offset.i64      # [N*C_mc + 1] layout-major
```

Multi-categorical was added in `6a557e5` after a review caught it as a real gap (RelGT's `SharedMultiCategoricalEncoder` consumes `MultiNestedTensor` directly).

**Flag wiring on `main_node_ddp.py`:**

```
--mode {hdf5, streaming, precomputed_shards}   default: hdf5
--shards_dir <path>                            for precomputed_shards
--tf_store_dir <path>                          memmap-back TFs
--max_rows_per_task N                          cap seed rows / split
```

Default unchanged so dev-kyaw parity remains apples-to-apples.

**Validation:**

- 11 new unit tests (Sh1–Sh4 shard_io; T1–T7 tf_store including the multi-cat round-trip).
- `tests/smoke_pr2_modes.py` end-to-end on real rel-f1 val split:
  - **E1**: streaming batch == precomputed_shards batch (subgraph tensors bit-equal across a 16-row batch on K=32).
  - **E3**: memmap-TF batch == in-RAM-TF batch (per-stype features equal, NaN-tolerant since numerical columns carry NaN missing-value markers).
- PR1's rel-f1 dev-kyaw smoke still passes (no regression to the sampler).

### PR2 close-out — empirical results

**ML3.5 parity sweep (5 seeds × 2 tasks × 2 pipelines × 5 epochs each, total 20 runs):**

| Task (metric) | dev-kyaw | new | gap | threshold | verdict |
|---|---|---|---|---|---|
| rel-f1 / driver-position (MAE↓) | 9.385 ± 0.358 | 9.535 ± 0.310 | 0.150 | 0.358 | **PASS** |
| rel-f1 / driver-top3 (AUROC↑)   | 0.786 ± 0.010 | 0.765 ± 0.028 | 0.022 | 0.028 | **PASS** (thin) |

Both meet the §6.3.5 acceptance criterion (means within 1× std of each other). driver-top3's wider variance on the new pipeline traces to encoder changes that pre-date PR1 (`528e6a8`/`83472cc`/`78d572e` — z-score NaN fix + learned missingness + GloVe column-name embeddings). The sampler/collate refactor is bit-equivalent vs dev-kyaw at the unit level.

**Memory smoke (`scripts/memory_smoke.sh 50` on RTX 5070, 12 GB):**

| Signal | Peak | Median | Final | Bound |
|---|---|---|---|---|
| CPU RSS | 4.43 GB | 3.88 GB | 0.54 GB (exit) | 27 GB laptop RAM |
| GPU VRAM | 3.86 GB | 3.86 GB | 3.47 GB | 12 GB on RTX 5070 |

Flat VRAM across 41 samples confirms training does not leak GPU memory across steps. Streaming + memmap-TF + `--max_rows_per_task` plumbing works end-to-end through DDP + GPU.

---

## What's next (PR3 + PR4)

### PR3 — Multi-task plumbing

**Scope** (per [`multi_task_refactor_plan.md` §6.1.5–§6.1.8](multi_task_refactor_plan.md)):

- `gfm_data/multi_task_dataset.py` — `MultiTaskConcat` + `DistributedMultiTaskSampler` (DDP-aware, all ranks pick same task per step).
- `gfm_data/collate.py` — extend to handle mixed-task batches (`task_id` / `task_type_id` already plumbed in PR1).
- `heads/multi_task_head.py` — shared **numeric head** (regression) + **boolean head** (binary) per RT §3.3.
- `losses/multi_task_loss.py` — HuberLoss(reg rows) + BCE(bin rows), plain batch mean (RT formula). `--loss_balance` flag for ablations.
- `gfm_data/task_tokens.py` — z-score regression target normalization (stats fit on train split, applied in collate, denormalized for metrics).
- `main_node_ddp.py` — `--tasks "rel-f1.driver-position,rel-f1.driver-top3"` arg, multi-task wrapper around RelGT (no edits to `model.py`).

**Tests:** Co1–Co6 (mixed-batch collate), D1–D5 (DDP sampler), H1–H7 (heads + loss), M1–M4 (per-task metrics), ML4–ML7 (multi-task overfit + balance + interference). Re-run ML3.5 in single-task degenerate mode as PR3 acceptance.

### PR4 — cross-dataset namespacing + 6-task launcher

**What landed (laptop side):**
- `unified_type_map` plumbed through `TaskTokens.__init__` so HDF5 / shard precompute writes a single global type vocabulary across datasets. Required because each `DatasetGraphCache` re-numbers types from 0 with its own prefix; without unification, the model's `NeighborNodeTypeEncoder` would mis-embed cross-dataset rows.
- Per-(dataset, task) shards path resolution in `train_multi_task.py` so `--shards_dir <root>` automatically expands to `<root>/<ds>/<task>/` per task.
- `tests/test_cross_dataset_types.py` (3 tests): two caches with different `name_prefix` produce disjoint namespaces; the unified-map branch in `TaskTokens` overrides the per-cache map; correct global ids assigned.
- `scripts/pretrain_6task.sh` — three-phase reference launcher (build TF memmaps → build shards → DDP train).

**Deferred to 8×A100 box (per plan §8):**
- Actual rel-event TF memmap + shards build. On this laptop, loading rel-event TFs takes ~10–15 min just for `events.pt` (13 GB on disk → ~14 GB RAM peak). On 8×A100/1TB the build is trivial.
- The 6-task pretraining run itself.
- Tests ML11–ML13 (cross-dataset sanity on real run), MS2 (8×A100 throughput).
- ML4–ML7 acceptance on the rel-event side (we have the rel-f1 side: train_loss 0.470 → 0.331 → 0.303 over 3 epochs, multi-task driver-top3 AUROC 0.835 vs single-task 0.786).

**Reproduction:**
```
# 8xA100 / 1TB box, full pretraining (11 tasks, paper-style hyperparams):
./scripts/pretrain_alltasks.sh
# Knobs via env vars: K, EPOCHS, MAX_STEPS, NPROC, LR, LOSS_BALANCE, OUT_DIR, RUN_NAME

# This laptop (27 GB RAM, RTX 5070 12 GB VRAM), reduced config:
./scripts/pretrain_laptop.sh
# Knobs: K, BATCH, CHANNELS, HEADS, EPOCHS, MAX_STEPS, MAX_ROWS_TRAIN, LOSS_BALANCE
```

**ML6 grad-norm logging** (added in PR4 close-out): `train_multi_task.py` now logs `head_grad_norm_numeric` and `head_grad_norm_boolean` to wandb each step. After warmup the two should stay within an order of magnitude of each other; one going to zero would indicate head starvation (a real risk if a task type has very few rows in a batch).

### Future (post-PR4, out of current "don't modify model.py" scope)

- ICL / context table prompting (touches `model.py`).
- Multi-class / multi-label / link-prediction heads.
- Forward-looking samplers / PQL.
