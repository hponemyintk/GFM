# Task 1: Table Name Namespacing + GloVe Precomputation

## Objective

1. Prefix all table/node type names with the dataset name to prevent collisions. Format: `"{dataset_name}__{table_name}"`.
2. Precompute GloVe embeddings at init time to eliminate the CPU bottleneck in every forward pass.

## Context

Different RelBench datasets may have tables with the same name (e.g., "users") but different schemas. Namespacing prevents collisions in the union `node_type_map`. Additionally, the current `NeighborNodeTypeEncoder` runs SentenceTransformer on CPU every forward pass — a major DDP bottleneck since it forces CPU-GPU sync every step.

## Files to Modify

### 1. `encoders.py` — `NeighborNodeTypeEncoder`

**A) Add GloVe text cleaning:**
- Add a static method `_name_to_glove_text(name: str) -> str`
- Implementation: `re.sub(r'[^a-zA-Z0-9\s]', ' ', name).strip()`
  - `"rel-f1__drivers"` -> `"rel f1 drivers"`
  - `"rel-hm__articles"` -> `"rel hm articles"`
  - `"drivers"` -> `"drivers"` (backward compatible)
  - `"mask"` -> `"mask"`

**B) Precompute GloVe embeddings at init (DDP performance fix):**

Replace the current design (GloVe called every forward pass) with precomputed embeddings:

```python
def __init__(self, node_type_map, embedding_dim):
    super().__init__()
    self.inv_node_type_map = {v: k for k, v in node_type_map.items()}
    self.mask_idx = len(node_type_map)
    self.inv_node_type_map[self.mask_idx] = "mask"

    # Precompute GloVe embeddings for ALL known table names
    embedder = GloveTextEmbedding(device="cpu")
    all_indices = sorted(self.inv_node_type_map.keys())
    all_names = [self._name_to_glove_text(self.inv_node_type_map[idx]) for idx in all_indices]

    with torch.no_grad():
        all_embeddings = embedder(all_names)  # [num_types+1, 300]

    # Store as buffer — moves with model across devices, saved in state_dict
    self.register_buffer("glove_embeddings", all_embeddings)

    # Projection layer (GloVe 300d -> embedding_dim)
    self.proj = nn.Linear(300, embedding_dim)

    # GloVe model no longer needed at runtime — don't store it
```

**C) Simplify forward() to pure tensor lookup:**

```python
def forward(self, type_indices):
    # type_indices: [B, K] int tensor
    # Pure GPU tensor operation — no CPU, no SentenceTransformer
    x = self.glove_embeddings[type_indices]  # [B, K, 300]
    return self.proj(x)
```

This eliminates: the `self.embedder` attribute, the `self.st_model` attribute, the `torch.unique` deduplication (unnecessary with buffer lookup), and all CPU-GPU synchronization.

**Backward compatibility**: The new code works identically for un-namespaced names. `"drivers"` is cleaned to `"drivers"` and embedded normally.

### 2. `encoders.py` — `NeighborTfsEncoder`

**No changes needed.** The Z-score normalization buffers already use `re.sub(r'[^a-zA-Z0-9]', '_', node_type)` to create safe buffer names (line 495). This handles namespaced names correctly. The `inv_node_type_map` lookup in `forward()` (line 578) will naturally contain namespaced names.

### 3. `utils.py` — `RelGTTokens`

**New parameters in `__init__`:**
- `union_node_type_map: Optional[Dict[str, int]] = None` — if provided, use instead of building from `self.data.node_types`
- `dataset_name: Optional[str] = None` — stored as metadata

**Changes to `__init__` (lines 313-315):**
```python
if union_node_type_map is not None:
    self.node_type_to_index = union_node_type_map
    self.index_to_node_type = {idx: nt for nt, idx in union_node_type_map.items()}
else:
    # existing behavior (backward compat)
    self.node_type_to_index = {nt: idx for idx, nt in enumerate(self.node_types)}
    self.index_to_node_type = {idx: nt for idx, nt in enumerate(self.node_types)}
```

**Changes to `__getitem__` and `collate`:**
- Include `dataset_name` in the sample dict (pass-through metadata)

**Recommended approach:** Namespace the HeteroData at the data loading level (Task 3), BEFORE creating RelGTTokens. This way RelGTTokens receives already-namespaced data.

### 4. New utility function (in `utils.py` or `data_loading.py`)

```python
def namespace_hetero_data(data: HeteroData, dataset_name: str) -> HeteroData:
    """Prefix all node types and edge types with dataset name.
    "drivers" -> "rel-f1__drivers"
    ("drivers", "to", "races") -> ("rel-f1__drivers", "to", "rel-f1__races")
    """
```

Called once per dataset during loading, before any downstream code sees the data.

## Testing

See `agent_tasks/testing_plan.md` — Phase 1, Task 1 tests.

## Dependencies

- None (this task can be done independently)
- Other tasks depend on this one being complete
- DDP benefit: GloVe precomputation eliminates CPU-GPU sync bottleneck on all GPUs
