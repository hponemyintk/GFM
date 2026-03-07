# Encoder Table-Agnosticism Audit (2026-03-07)

## Result: All 5 Encoders Are Table-Agnostic

Full audit of every component that touches table/dataset identity. Verified that no encoder has per-table learnable parameters.

### Per-Encoder Status

| Encoder | Table-specific params? | Mechanism | Status |
|---|---|---|---|
| `NeighborNodeTypeEncoder` | None | GloVe buffer + shared `proj` Linear | Table-agnostic |
| `NeighborHopEncoder` | None | Shared `nn.Embedding` (hop values universal) | Table-agnostic |
| `NeighborTimeEncoder` | None | Shared positional encoding + linear | Table-agnostic |
| `NeighborTfsEncoder` | Z-score buffers are per-table but only DATA (not params). All trainable weights are shared. | CLS + shared transformer | Table-agnostic |
| `GNNPEEncoder` | None | Shared GIN on subgraph structure | Table-agnostic |
| `LocalModule` | None | Pure shared attention + FFN | Table-agnostic |
| `VectorQuantizerEMA` | None | Shared codebook indexed by centroid | Table-agnostic |

### Sub-Encoder Details (inside NeighborTfsEncoder)

| Sub-Encoder | How it handles any table |
|---|---|
| `SharedNumericalEncoder` | Shared MLP(1→channels) per scalar, Z-score normalized by precomputed per-table stats |
| `SharedCategoricalEncoder` | Hash % 9311 → shared embedding. No per-table vocabulary. |
| `SharedMultiCategoricalEncoder` | Same hash trick, mean pool over elements |
| `SharedTimestampEncoder` | Sin/cos positional encoding → linear. Universal. |
| `SharedEmbeddingEncoder` | ModuleDict keyed by embedding dim (discovered from StatType.EMB_DIM). 2D input reshaped via divisibility check; ambiguous matches raise KeyError. |

### Model-Level Blockers (NOT in encoders, in model.py)

3 issues remain in `model.py` / `RelGTLayer`:

1. **c_idx buffer** (`model.py:65`): Shape `[num_nodes]`, per-node centroid assignment. Blocks multi-dataset if using global/full conv_type. Works with `conv_type="local"`.
2. **Task-specific head** (`model.py:258`): Fixed `out_channels`. Can't handle multiple task types simultaneously.
3. **num_nodes parameter** (`model.py:166`): Needed for c_idx sizing. Must be total across all datasets.

**Workaround**: Use `conv_type="local"` to bypass all global attention issues, then only the task head problem remains.

## torch_frame Discoveries

### StatType.EMB_DIM
- `torch_frame.data.stats.StatType.EMB_DIM` stores the embedding dimension per column
- Computed as `len(ser[0])` — dimension of the first embedding in that column
- Available in `col_stats_dict[table_name][col_name][StatType.EMB_DIM]`
- Used by `SharedEmbeddingEncoder` to create per-dim projectors instead of hardcoding 300d

### Z-Score Already Handled by torch_frame
- torch_frame's `LinearEncoder` (used in the original ResNet-based pipeline) reads `StatType.MEAN` and `StatType.STD` from `col_stats` and does `feat = (feat - mean) / std` internally
- Our manual Z-score in `NeighborTfsEncoder._normalize_numerical` correctly replicates this behavior at a different level
- Stats come from the same `col_stats_dict` source in both cases

### col_stats_dict Won't Work Across Datasets Without Namespacing
- `make_pkey_fkey_graph()` keys `col_stats_dict` by raw table name
- Multiple datasets can have tables with the same name but different schemas (e.g., "user")
- Merging without namespacing silently overwrites one dataset's stats
- Same collision problem for `col_names_dict`, `node_type_map`, HeteroData node types
- Fix: namespace with `"{dataset_name}__{table_name}"` before merging (data pipeline concern, not encoder concern)

## Validation Guards Added (2026-03-07)

| Guard | Location | What it catches |
|---|---|---|
| Contiguous index check | `NeighborNodeTypeEncoder.__init__` | Non-contiguous or non-0-based node_type_map values |
| Ambiguous 2D reshape | `SharedEmbeddingEncoder.forward` | Multiple projector dims dividing the flattened embedding dim |
| Safe name collision | `NeighborTfsEncoder.__init__` | Two table names sanitizing to same buffer name (e.g. "a-b" and "a_b") |
| Missing col_dicts at inference | `NeighborTfsEncoder.forward` | `_num_zscore_tables > 0` but `_node_type_to_safe` empty after checkpoint load |
| NaN mean imputation | `NeighborTfsEncoder._normalize_numerical` | NaN replaced with column mean before Z-score (prevents outlier injection) |

## GloVe Buffer Memory
- Buffer shape: `[num_table_names+1, 300]` floats
- ~100 table names across all RelBench = 100 × 300 × 4 bytes = 120KB
- Negligible memory. No need for CPU/GPU switching — just lives on model device.
