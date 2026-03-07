# Foundation Model Architecture: Changes & Tradeoffs

## Goal

Transform the current single-dataset, single-task RelGT into a foundation model that:
- Trains on ALL RelBench datasets simultaneously (leave-one-out)
- Uses self-supervised masked token prediction (a la RT paper, arXiv 2510.06377)
- Evaluates via zero-shot prediction on the held-out dataset
- Learns universal relational representations across diverse schemas
- Runs correctly under DDP multi-GPU training

## Current Architecture (Single-Dataset)

```
RelGTTokens (per dataset/task/split)
    -> neighbor sampling, precompute to HDF5
    -> collate groups TensorFrames by node type

RelGT
    -> 5 encoders: type (GloVe), hop (embed), time (pos enc), features (table-agnostic TF), GNN PE
    -> concat 5 tokens -> MLP mixture -> [B, K, channels]
    -> RelGTLayer(s): local attention (subgraph) + global attention (VQ centroids)
    -> seed node representation (position 0) -> task-specific MLP head -> prediction
```

### Known DDP Bugs in Current Code

Even the existing single-dataset code has subtle DDP issues:
1. VQ codebook EMA buffers (`_embedding`, `_ema_w`, `_ema_cluster_size`) are NOT synced across GPUs
2. `c_idx` (per-node centroid assignments) diverges across GPUs
3. `BatchNorm` inside VQ may or may not be converted by `SyncBatchNorm`

These are masked in single-dataset training because all GPUs see data from the same distribution, so drift is slow. In multi-dataset training they WILL break.

## Proposed Architecture (Foundation Model)

```
Multi-Dataset Loader
    -> loads ALL RelBench datasets
    -> builds union node_type_map, col_names_dict, col_stats_dict
    -> namespaces table names: "drivers" -> "rel-f1__drivers"

RelGTTokens (per dataset/task/split, with shared union mappings)
    -> same neighbor sampling, uses union node_type_map
    -> samples carry dataset_id metadata

RelGT Backbone (shared, DDP-safe)
    -> GloVe embeddings precomputed at init (no CPU bottleneck)
    -> same 5 encoders (already table-agnostic)
    -> RelGTLayer(s) with DDP-safe global attention:
        - c_idx eliminated, replaced by _ema_cluster_size for log-count bias
        - VQ EMA buffers explicitly all-reduced across GPUs after each update
    -> encode() returns FULL sequence [B, K, channels] for masking head
    -> forward() returns seed-only [B, channels] for backward compat

Pretraining Head (masked token prediction)
    -> mask random neighbor tokens' features
    -> reconstruct masked features from context
    -> single reconstruction loss (no reg/cls balancing needed)

Zero-Shot Evaluation
    -> frame target column as masked feature to predict
    -> backbone produces representation, predict target without fine-tuning
```

---

## Key Design Decisions & Tradeoffs

### 1. Pretraining Objective: Masked Token Prediction vs Multi-Task Supervised

**Chosen: Masked Token Prediction (self-supervised)**

| Aspect | Masked Token Prediction | Multi-Task Supervised |
|--------|------------------------|----------------------|
| Loss balancing | Single loss type — no balancing needed | Must balance L1 (regression) vs BCE (classification) across tasks |
| Zero-shot capability | Natural — frame target as masked token | Requires trained head per task type; true zero-shot is hard |
| Data efficiency | Uses ALL data, not just labeled samples | Only uses labeled train splits |
| Foundation model purity | True self-supervised pretraining | More like multi-task fine-tuning |
| Implementation complexity | Need to design masking strategy for 5-token decomposition | Need per-task heads, loss weighting, interleaved sampling |
| Risk | Reconstruction objective may not align with downstream tasks | Supervised signal is more directly useful |

**Rationale**: Masked token prediction sidesteps the mixed-loss problem entirely, aligns with the zero-shot evaluation goal, and is the approach used by the RT paper (arXiv 2510.06377) which achieves 93% of fully supervised AUROC zero-shot.

### 2. Table Name Namespacing

**Chosen: Dataset prefix with double underscore separator**

Format: `"{dataset_name}__{table_name}"` (e.g., `"rel-f1__drivers"`)

- Applied everywhere: `node_type_map`, `col_names_dict`, `col_stats_dict`, Z-score buffer names
- `NeighborNodeTypeEncoder` precomputes GloVe embeddings at init from cleaned text: `"rel-f1__drivers"` -> `"rel f1 drivers"`
- Prevents collisions when different datasets have tables with the same name but different schemas

**Tradeoff**: Adds a prefix to GloVe embeddings which changes the embedding (average of more words). This is actually desirable — it gives the model dataset-level context alongside table semantics.

### 3. Batch Interleaving Strategy

**Chosen: Temperature-scaled sampling (alpha = 0.5)**

Sample dataset with probability proportional to `dataset_size^alpha`.

- `alpha = 1.0` = proportional (large datasets dominate)
- `alpha = 0.0` = uniform (all datasets equal)
- `alpha = 0.5` = square root scaling (compromise)

**Tradeoff**: Square-root scaling gives smaller datasets more representation than proportional but avoids the over-repetition of uniform. This is the standard approach from multilingual NLP (mBERT, XLM-R). May need tuning.

### 4. Global Attention: DDP-Safe VQ Centroids

**Design**: Single shared VQ codebook. `c_idx` eliminated. VQ EMA buffers synced via all-reduce.

Previous design used `c_idx` (per-node centroid assignment, shape `[num_nodes]`) for log-count attention bias. This has two problems:
- Memory: scales linearly with total nodes across all datasets (could be millions)
- DDP: each GPU updates its local copy independently → divergence

**New design**:
- Replace `c_idx`-based counting with `_ema_cluster_size` (already tracked by VQ, already a smooth estimate)
- After each VQ `update()`, explicitly `all_reduce` the EMA buffers across GPUs
- At eval on unseen dataset: centroids are a learned dictionary; unseen nodes query via attention without per-node state

**Tradeoff**: Loses exact per-node centroid tracking. `_ema_cluster_size` is a smoothed estimate, not exact counts. In practice this is better — EMA is more stable than raw counts, especially early in training.

### 5. GloVe Precomputation (Performance) — IMPLEMENTED (commit 58cdc8d)

**Design**: Precompute all GloVe embeddings at `__init__` time, store as buffer.

Previous code ran SentenceTransformer on CPU every forward pass AND registered it as a submodule (wasting memory, causing DDP unused-parameter issues). With ~50-100 table names across all datasets, this is a fixed, small set.

**Implementation**: SentenceTransformer is a local variable in `__init__`, never assigned to `self`. Embeddings stored as `register_buffer("glove_embeddings", ...)`. Forward is `buffer[indices]` → `Linear(300, dim)`. Buffer is ~120KB for all RelBench.

**Tradeoff**: Loses the ability to dynamically embed new table names at inference without re-initializing. This is acceptable — the set of tables is known at model creation time.

### 5b. SharedEmbeddingEncoder Dimension Discovery — IMPLEMENTED (commit 58cdc8d)

**Design**: Discover embedding dimensions from `col_stats_dict` via `StatType.EMB_DIM` (already computed by torch_frame). Create one `nn.Linear(dim, out_channels)` per unique dim in `nn.ModuleDict`.

**Previous code**: Hardcoded `nn.Linear(300, out_channels)`, would crash on non-300d embeddings.

**Implementation**: `NeighborTfsEncoder` scans `col_stats_dict` at init → extracts all `EMB_DIM` values → passes `emb_dims` set through `TableAgnosticStypeEncoder` → `SharedEmbeddingEncoder`.

**Tradeoff**: Requires user to use the same text embedder at inference as training. This is enforced by the data pipeline (`make_pkey_fkey_graph` + `text_embedder_cfg`).

### 6. Zero-Shot Evaluation Protocol

**Design**: No fine-tuning on held-out dataset. Backbone produces representation, target is predicted by treating it as a masked token.

**Tradeoff**: Hardest evaluation protocol. Performance will be lower than fine-tuning or linear probing, but most scientifically meaningful for a "foundation model" claim. If results are poor, can fall back to linear probing (freeze backbone, train small head) as a secondary evaluation.

### 7. What Stays the Same

The following components require NO architectural changes (verified via full audit 2026-03-07):
- `local_module.py` — generic local attention, fully universal, DDP-safe (standard parameters + SyncBatchNorm)
- `NeighborHopEncoder` — generic hop embedding
- `NeighborTimeEncoder` — generic time positional encoding
- `GNNPEEncoder` — generic GIN-based PE
- `TableAgnosticStypeEncoder` and sub-encoders — already table-agnostic via hashing/shared MLPs
- `VectorQuantizerEMA` — shared codebook, no table-specific params (DDP sync issue is separate)

### 8. torch_frame Internals (discovered 2026-03-07)

Key findings about the data pipeline:
- `StatType.EMB_DIM` exists in torch_frame stats — stores embedding column dimension per column
- torch_frame's `LinearEncoder` already does Z-score normalization using `MEAN`/`STD` from `col_stats`
- `col_stats_dict` from `make_pkey_fkey_graph()` is keyed by raw table name — will collide across datasets without namespacing
- The same collision applies to `col_names_dict`, `node_type_map`, and HeteroData node types
- **Namespacing is a data pipeline concern**, not an encoder concern — encoders are already agnostic

---

## DDP Multi-GPU Summary

| Component | DDP Status | Issue | Fix |
|-----------|-----------|-------|-----|
| Local attention (LocalModule) | Safe | None | SyncBatchNorm handles BN |
| All 5 encoders | Safe | None | Standard nn.Parameter, gradients synced |
| Feed-forward / MLP layers | Safe | None | Standard nn.Parameter |
| VQ codebook EMA buffers | BROKEN | Buffers not synced across GPUs | Explicit all-reduce after update (Task 7) |
| `c_idx` per-node tracking | BROKEN | Each GPU updates locally, diverges | Eliminate entirely, use `_ema_cluster_size` (Task 7) |
| VQ BatchNorm | CHECK | May or may not be caught by SyncBatchNorm | Verify conversion, add explicit check (Task 7) |
| GloVe in forward() | FIXED (58cdc8d) | CPU SentenceTransformer every step | Precomputed at init, stored as buffer |
| Interleaved sampling | Safe | Different GPUs may get different datasets | This is fine — DDP syncs gradients regardless |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Masked token prediction doesn't transfer well to task prediction | Medium | Can add supervised multi-task fine-tuning as second stage |
| Memory blow-up from combined graphs | Medium | Process datasets sequentially per epoch, only load one graph at a time |
| GloVe embeddings don't capture namespaced table semantics well | Low | GloVe averages word embeddings; "rel f1 drivers" gives meaningful tokens |
| VQ codebook doesn't converge with diverse data | Low | EMA update is robust; can increase num_centroids; now synced across GPUs |
| Zero-shot performance too low to be useful | Medium-High | Fall back to linear probe evaluation as secondary metric |
| VQ EMA all-reduce adds communication overhead | Low | Small tensors (num_centroids x dim); negligible vs gradient sync |
| TensorFrame collation slow with many namespaced types | Medium | Temperature-scaled sampling keeps per-batch type count manageable |

---

## Task Dependency Graph & Execution Order

```
Phase 1 (parallel, no dependencies):
    Task 1: Table name namespacing + GloVe precomputation
    Task 2: Backbone/head separation + encode() method

Phase 2 (parallel, depends on Phase 1):
    Task 3: Multi-dataset data loading (depends on Task 1)
    Task 5: Masked token pretraining objective (depends on Task 2)
    Task 7: DDP fixes for global attention (depends on Task 2)

Phase 3 (sequential, depends on Phase 2):
    Task 4: Foundation training script (depends on Tasks 1,2,3,5,7)

Phase 4 (depends on Phase 3):
    Task 6: Zero-shot evaluation pipeline (depends on Tasks 2,3,4,5)

Testing: Runs after each phase (see testing_plan.md)
```
