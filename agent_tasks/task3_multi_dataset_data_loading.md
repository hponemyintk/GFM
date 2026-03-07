# Task 3: Multi-Dataset Data Loading

## Objective

Load ALL RelBench datasets, build union mappings (node types, column names, column stats), namespace table names, compute global node index offsets, and create `RelGTTokens` instances for every dataset/task/split combination.

## Context

Currently `main_node_ddp.py` loads a single dataset and task. For foundation model training, we need to load all RelBench datasets simultaneously, with a unified namespace and shared mappings so the model sees a consistent view of all tables.

## RelBench Datasets and Tasks

The following are the known RelBench datasets and their tasks (verify against current RelBench version):

```python
RELBENCH_CONFIGS = {
    "rel-f1": ["driver-top3", "driver-dnf", "driver-position"],
    "rel-hm": ["user-churn", "item-sales"],
    "rel-stack": ["user-engagement", "post-votes", "user-badge"],
    "rel-amazon": ["user-churn", "item-churn", "user-ltv", "item-ltv"],
    "rel-event": ["user-repeat", "user-ignore"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success"],
    "rel-avito": ["user-clicks", "user-visits", "ad-ctr"],
}
```

**NOTE:** This list must be verified against the actual RelBench package at runtime. Some tasks may not exist or may have been renamed.

## Files to Create/Modify

### 1. New file: `data_loading.py`

Central module for multi-dataset loading. Contains:

**`namespace_hetero_data(data: HeteroData, dataset_name: str) -> HeteroData`**
- Rename all node types: `"drivers"` -> `"rel-f1__drivers"`
- Rename all edge types: `("drivers", "rel", "races")` -> `("rel-f1__drivers", "rel", "rel-f1__races")`
- Must update: `data.node_types`, `data.edge_types`, all stored attributes (`.tf`, `.time`, `.num_nodes`, `.edge_index`, etc.)
- Use PyG's rename utilities if available, otherwise manually rebuild the HeteroData

**`namespace_col_dicts(col_names_dict, col_stats_dict, dataset_name) -> (dict, dict)`**
- Prefix keys: `"drivers"` -> `"rel-f1__drivers"` in both dicts
- Values (column names, stats) stay the same — only the table-level keys change

**`build_union_mappings(dataset_configs: List[str]) -> dict`**
- Load each dataset via `get_dataset(name)`
- Build `make_pkey_fkey_graph()` for each
- Namespace all table names
- Return a dict containing:
  ```python
  {
      "union_node_type_map": {nt: idx for idx, nt in enumerate(sorted(all_node_types))},
      "union_col_names_dict": {...},  # merged across all datasets
      "union_col_stats_dict": {...},  # merged across all datasets
      "total_num_nodes": int,         # sum across all datasets
      "dataset_offsets": {"rel-f1": 0, "rel-hm": 10000, ...},  # global node index offsets
      "per_dataset": {
          "rel-f1": {"data": HeteroData, "col_stats": dict, "num_nodes": int},
          ...
      }
  }
  ```

**`create_all_relgt_tokens(union_info, held_out_dataset, K, ...) -> dict`**
- For each dataset (excluding held_out):
  - For each task in that dataset:
    - Create `RelGTTokens` for train/val splits
    - Pass `union_node_type_map` and `global_node_offset`
- For the held_out dataset:
  - Create `RelGTTokens` for val/test splits only (no training data)
- Return organized dict:
  ```python
  {
      "train": {("rel-f1", "driver-top3"): RelGTTokens, ...},
      "val": {("rel-f1", "driver-top3"): RelGTTokens, ...},
      "held_out_val": {("rel-hm", "user-churn"): RelGTTokens, ...},
      "held_out_test": {("rel-hm", "user-churn"): RelGTTokens, ...},
  }
  ```

### 2. `utils.py` — `RelGTTokens` modifications

**New parameters in `__init__`:**
- `union_node_type_map: Optional[Dict[str, int]] = None` — if provided, use instead of building from `self.data.node_types`
- `global_node_offset: int = 0` — added to all `node_indices` for global attention
- `dataset_id: Optional[str] = None` — stored as metadata, passed through in samples

**Changes to `__init__`:**
```python
# Replace lines 313-315:
if union_node_type_map is not None:
    self.node_type_to_index = union_node_type_map
    self.index_to_node_type = {idx: nt for nt, idx in union_node_type_map.items()}
else:
    # existing behavior (backward compat)
    self.node_type_to_index = {nt: idx for idx, nt in enumerate(self.node_types)}
    self.index_to_node_type = {idx: nt for idx, nt in enumerate(self.node_types)}
```

**Changes to `_create_global_mappings()`:**
- Add `self.global_node_offset` to all global indices

**Changes to `__getitem__` and `collate`:**
- Include `dataset_id` in the sample dict
- Include `task_id` in the sample dict (the task name)

### 3. Caching strategy

Each dataset's precomputed HDF5 files should be stored under:
```
{cache_dir}/precomputed/{dataset_name}/{task_name}/{K}/{split}.h5
```
This is already the pattern used (line 150 in `main_node_ddp.py`). No change needed.

**Important:** The `node_type_to_index` mapping stored in the HDF5 types column must match the union mapping. If precomputed files exist from single-dataset runs, they will have WRONG type indices and must be regenerated.

## Memory Considerations

Loading all datasets simultaneously may exceed memory. Recommended approach:
- Load and namespace one dataset at a time
- Build the union mappings in a first pass (just metadata, not full graphs)
- In a second pass, create `RelGTTokens` and precompute to HDF5
- At training time, `RelGTTokens.__getitem__` reads from HDF5 (lazy), so only the graph metadata needs to stay in memory for collation (for TensorFrame lookups)

## Testing

- Verify that union `node_type_map` contains all table names from all datasets, properly namespaced
- Verify that `dataset_offsets` are non-overlapping
- Verify that `col_stats_dict` has no key collisions after namespacing
- Verify that `RelGTTokens` with `union_node_type_map` produces correct type indices in HDF5
- Verify backward compatibility: `RelGTTokens` without `union_node_type_map` works as before

## Dependencies

- Task 1 (namespacing) must define the `_name_to_glove_text` method and `namespace_hetero_data` interface
- This task is a prerequisite for Task 4 (training script) and Task 6 (evaluation)
