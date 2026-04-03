# Similarity-Based Sampler: Detailed Implementation Guide

This document describes all code changes needed to add two-stage training with
three-tier priority weighted sampling to the RelGT codebase. Use this to
recreate the changes on any branch.

## Overview

**Problem**: The local subgraph sampler uses uniform random sampling — all
neighbors at the same hop distance are treated equally.

**Solution**: Two-stage training pipeline:
- **Stage 1** (`finetune`): Train normally with uniform sampling. Save model
  checkpoint (includes `NeighborTfsEncoder` weights).
- **Stage 2** (`similarity_resample`): Load trained encoder, pre-compute
  embeddings for all nodes on GPU, then re-sample neighbors with weighted
  priorities. Overwrite the existing HDF5 precomputed files. Train again with
  the improved samples.

**Three-tier priority scoring** (per candidate neighbor):
```
score = w_fk * is_fk_parent + w_sim * cosine_similarity + w_rec * recency
```
1. **FK parent bonus** (`w_fk=5.0`): 1.0 if neighbor is a direct FK-to-PK
   parent of the seed, else 0.0. Detected via `rel_name.startswith("f2p_")`.
2. **Similarity** (`w_sim=2.0`): Cosine similarity of L2-normalized embeddings.
   Same-type only; cross-type defaults to 0.
3. **Recency** (`w_rec=1.0`): `exp(-0.01 * relative_time_days)`. More recent
   nodes score higher.

Scores are converted to probabilities via softmax with temperature, then sampled
using `np.random.choice`.

## Background: FK/PK Edge Convention

relbench's `make_pkey_fkey_graph()` creates two edge types per FK relationship:
- `(fk_table, "f2p_{fkey_col}", pk_table)` — FK-to-PK (child → parent)
- `(pk_table, "rev_f2p_{fkey_col}", fk_table)` — reverse (parent → child)

Both directions are always present in `HeteroData.edge_types`.

---

## Files Modified

Only two files need changes: `utils.py` and `main_node_ddp.py`. No changes to
`encoders.py`, `model.py`, `local_module.py`, or the HDF5 schema.

One new file: `run_pipeline.sh` (optional convenience script).

---

## Changes to `utils.py`

### 1. Add new global variables (after existing `GLOBAL_ALL_NODES`)

```python
GLOBAL_ADJ = None
GLOBAL_ALL_NODES = None
GLOBAL_NODE_EMBEDDINGS = None  # NEW
GLOBAL_W_FK = 5.0              # NEW
GLOBAL_W_SIM = 2.0             # NEW
GLOBAL_W_REC = 1.0             # NEW
GLOBAL_TEMPERATURE = 1.0       # NEW
```

### 2. Add `precompute_node_embeddings()` function

Add this new function **before** `class GloveTextEmbedding`. It loads trained
`NeighborTfsEncoder` weights from a Stage 1 checkpoint, runs batched GPU forward
pass over all nodes per type, L2-normalizes, returns numpy arrays.

```python
def precompute_node_embeddings(
    data: HeteroData,
    encoder_weights_path: str,
    node_type_map: Dict[str, int],
    col_names_dict,
    col_stats_dict,
    channels: int,
    device: str = "cuda",
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """
    Load trained NeighborTfsEncoder weights from a Stage 1 checkpoint,
    encode all nodes per type on GPU, L2-normalize, return as numpy arrays.

    Returns:
        {node_type_str: np.ndarray of shape [num_nodes, channels]} with L2-normalized rows.
    """
    import torch_frame
    from encoders import NeighborTfsEncoder

    encoder = NeighborTfsEncoder(
        channels=channels,
        node_type_map=node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
    )

    checkpoint = torch.load(encoder_weights_path, map_location="cpu")
    tfs_keys = {k.replace("tfs_encoder.", ""): v
                for k, v in checkpoint.items() if k.startswith("tfs_encoder.")}
    encoder.load_state_dict(tfs_keys)
    encoder.to(device).eval()

    node_embeddings = {}
    with torch.no_grad():
        for node_type in data.node_types:
            if node_type not in encoder.encoders:
                continue
            tf = data[node_type].tf
            num_nodes = data[node_type].num_nodes
            all_embs = []

            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch_tf = tf[start:end].to(device=device)
                # Clean NaN/Inf
                for stype_key, tensor in batch_tf.feat_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        batch_tf.feat_dict[stype_key] = torch.nan_to_num(
                            tensor, nan=0.0, posinf=1e6, neginf=-1e6
                        )

                out = encoder.encoders[node_type](batch_tf)
                if out.dim() == 3:
                    out = out.squeeze(1)
                all_embs.append(out.cpu())

            embs = torch.cat(all_embs, dim=0).numpy()  # [num_nodes, channels]
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            node_embeddings[node_type] = (embs / norms).astype(np.float32)

    total_nodes = sum(v.shape[0] for v in node_embeddings.values())
    print(f"[precompute_node_embeddings] Encoded {total_nodes} nodes "
          f"across {len(node_embeddings)} types, dim={channels}")
    return node_embeddings
```

### 3. Modify `build_adjacency_hetero` — store relation names as 3-tuples

**Before**: Adjacency entries are `(dst_type, dst_id)` 2-tuples. Relation name
is discarded (`src_type, _, dst_type = edge_type`).

**After**: Store `(dst_type, dst_id, rel_name)` 3-tuples.

```python
def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool = True):
    adjacency = {
        node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
        for node_type in hetero_data.node_types
    }
    for edge_type in hetero_data.edge_types:
        src_type, rel_name, dst_type = edge_type          # CHANGED: capture rel_name
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d, rel_name))  # CHANGED: 3-tuple
            if undirected:
                rev_rel = ("rev_" + rel_name) if not rel_name.startswith("rev_") else rel_name[4:]
                adjacency[dst_type][d].add((src_type, s, rev_rel))  # CHANGED: 3-tuple
    return adjacency
```

### 4. Modify `init_worker_globals` — accept new params

```python
def init_worker_globals(adj, all_nodes, node_embeddings=None, w_fk=5.0, w_sim=2.0, w_rec=1.0, temperature=1.0):
    global GLOBAL_ADJ, GLOBAL_ALL_NODES, GLOBAL_NODE_EMBEDDINGS
    global GLOBAL_W_FK, GLOBAL_W_SIM, GLOBAL_W_REC, GLOBAL_TEMPERATURE
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes
    GLOBAL_NODE_EMBEDDINGS = node_embeddings
    GLOBAL_W_FK = w_fk
    GLOBAL_W_SIM = w_sim
    GLOBAL_W_REC = w_rec
    GLOBAL_TEMPERATURE = temperature
```

### 5. Modify `gather_1_and_2_hop_with_seed_time` — 3-tuple adjacency + `is_fk_parent`

Key changes:
- Return type changes to `List[Tuple[str, int, int, float, Optional[set], bool]]`
  (6-tuples instead of 5-tuples)
- `n1` changes from a `set` to a `dict` mapping `(nbr_t, nbr_i) -> is_fk_parent`
- Unpack 3-tuple adjacency entries `(nbr_t, nbr_i, rel_name)`
- Determine `is_fk_parent` via `rel_name.startswith("f2p_")`
- If a neighbor appears via multiple edges, preserve FK parent status (OR logic)
- 2-hop neighbors always have `is_fk_parent = False`

```python
def gather_1_and_2_hop_with_seed_time(
    adjacency, data, node_type, node_idx, seed_time,
    max_1hop_threshold=5000, max_2hop_threshold=1000
) -> List[Tuple[str, int, int, float, Optional[set], bool]]:
    """Returns 6-tuples: (nbr_t, nbr_i, hop, rel_time_days, connecting_1hops, is_fk_parent)"""

    n1_full = adjacency[node_type][node_idx]
    if len(n1_full) > max_1hop_threshold:
        n1_full = random.sample(list(n1_full), max_1hop_threshold)
    else:
        n1_full = list(n1_full)

    # n1: maps (nbr_t, nbr_i) -> is_fk_parent
    n1 = {}
    for (nbr_t, nbr_i, rel_name) in n1_full:           # CHANGED: 3-tuple unpack
        if hasattr(data[nbr_t], "time"):
            if data[nbr_t].time[nbr_i] <= seed_time:
                is_fk = rel_name.startswith("f2p_")
                if (nbr_t, nbr_i) in n1:
                    n1[(nbr_t, nbr_i)] = n1[(nbr_t, nbr_i)] or is_fk
                else:
                    n1[(nbr_t, nbr_i)] = is_fk
        else:
            is_fk = rel_name.startswith("f2p_")
            if (nbr_t, nbr_i) in n1:
                n1[(nbr_t, nbr_i)] = n1[(nbr_t, nbr_i)] or is_fk
            else:
                n1[(nbr_t, nbr_i)] = is_fk

    # 2-hop gathering
    n2 = defaultdict(set)
    for (nbr_t, nbr_i) in n1:
        nbr2_full = adjacency[nbr_t][nbr_i]
        if len(nbr2_full) > max_2hop_threshold:
            nbr2_full = random.sample(list(nbr2_full), max_2hop_threshold)
        else:
            nbr2_full = list(nbr2_full)

        for (nbr2_t, nbr2_i, _rel) in nbr2_full:       # CHANGED: 3-tuple unpack
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue
            if hasattr(data[nbr2_t], "time"):
                if data[nbr2_t].time[nbr2_i] <= seed_time:
                    n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))
            else:
                n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))

    n2 = {k: v for k, v in n2.items() if k not in n1}

    neighbors_with_time = []

    # 1-hop: include is_fk_parent
    for (nbr_t, nbr_i), is_fk_parent in n1.items():     # CHANGED: dict iteration
        if hasattr(data[nbr_t], "time"):
            nbr_time = data[nbr_t].time[nbr_i].item()
            relative_time_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0
        neighbors_with_time.append((nbr_t, nbr_i, 1, relative_time_days, None, is_fk_parent))

    # 2-hop: is_fk_parent = False
    for (nbr2_t, nbr2_i), connecting_1hops in n2.items():
        if hasattr(data[nbr2_t], "time"):
            nbr2_time = data[nbr2_t].time[nbr2_i].item()
            relative_time_days = (seed_time - nbr2_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0
        neighbors_with_time.append((nbr2_t, nbr2_i, 2, relative_time_days, connecting_1hops, False))

    return neighbors_with_time
```

### 6. Modify `_process_one_seed` — weighted sampling with three-tier scoring

Replace the entire function. Key changes:
- Access `GLOBAL_NODE_EMBEDDINGS` and scoring weight globals
- Create `np_rng = np.random.RandomState(seed_val)` for weighted sampling
- Three code paths: (a) fallback if 0 neighbors, (b) uniform if
  `GLOBAL_NODE_EMBEDDINGS is None` (Stage 1), (c) weighted if embeddings
  available (Stage 2)
- Weighted path computes composite scores, applies softmax with temperature,
  samples via `np_rng.choice(size, replace=False/True, p=weights)`
- Fallback neighbors get 6-tuple format: `(ft, fi, 3, rel_time, None, False)`
- Before building `final_tokens`, strip `is_fk_parent` (6th element) from
  chosen neighbors — output remains 5-tuples for downstream compatibility
- Local edge-building loop unpacks 3-tuple adjacency:
  `for (nbr_t, nbr_i, _rel) in GLOBAL_ADJ[t_str][i]`

See the full function in the current `utils.py:224-346`.

### 7. Modify `local_nodes_hetero` — accept and pass new params

Add parameters to signature:
```python
def local_nodes_hetero(
    data, K, table_input_nodes, table_input_time,
    undirected=True, num_workers=None,
    node_embeddings=None,          # NEW
    w_fk=5.0,                      # NEW
    w_sim=2.0,                     # NEW
    w_recency=1.0,                 # NEW
    temperature=1.0,               # NEW
):
```

Pass to `init_worker_globals` via `initargs`:
```python
with Pool(
    processes=num_workers,
    initializer=init_worker_globals,
    initargs=(adjacency, all_nodes_all_types, node_embeddings,
              w_fk, w_sim, w_recency, temperature)  # CHANGED
) as pool:
```

Update log message:
```python
sampling_mode = "weighted" if node_embeddings is not None else "uniform"
print(f"  [Sampling:{sampling_mode}] ...")
```

### 8. Modify `RelGTTokens.__init__` — accept new params

Add to constructor signature:
```python
def __init__(self, data, task, K, split="train", undirected=True,
             num_workers=None, precompute=True, precomputed_dir=None,
             train_stage="finetune",
             node_embeddings=None,   # NEW
             w_fk=5.0,               # NEW
             w_sim=2.0,              # NEW
             w_recency=1.0,          # NEW
             temperature=1.0,        # NEW
):
```

Store as instance attributes:
```python
self.node_embeddings = node_embeddings
self.w_fk = w_fk
self.w_sim = w_sim
self.w_recency = w_recency
self.temperature = temperature
```

### 9. Modify `RelGTTokens` precompute logic — force overwrite in Stage 2

Replace the precompute check block:
```python
if self.precompute:
    if self.node_embeddings is not None and os.path.exists(self.precomputed_path):
        # Stage 2: overwrite existing samples with similarity-based resampling
        print(f"[{self.split}] Removing existing HDF5 for resampling: {self.precomputed_path}")
        os.remove(self.precomputed_path)

    if os.path.exists(self.precomputed_path):
        print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
    else:
        print(f"[{self.split}] Precomputing neighbor sampling (K={self.K})...")
        self._precompute_sampling()
```

### 10. Modify `_precompute_sampling` — pass sampling params

In the `local_nodes_hetero` call inside `_precompute_sampling`, add the new
keyword arguments:
```python
S_chunk = local_nodes_hetero(
    data=self.data.to("cpu"),
    K=self.K,
    table_input_nodes=(self.node_type, chunk_node_idxs),
    table_input_time=chunk_times,
    undirected=self.undirected,
    num_workers=self.num_workers,
    node_embeddings=self.node_embeddings,  # NEW
    w_fk=self.w_fk,                        # NEW
    w_sim=self.w_sim,                      # NEW
    w_recency=self.w_recency,              # NEW
    temperature=self.temperature,          # NEW
)
```

---

## Changes to `main_node_ddp.py`

### 1. Update import

```python
from utils import GloveTextEmbedding, RelGTTokens, precompute_node_embeddings
```

### 2. Add argparse arguments

After the existing `--train_stage` argument:
```python
parser.add_argument("--train_stage", type=str, default="finetune",
                    choices=["finetune", "similarity_resample"])  # CHANGED: add choice
parser.add_argument("--encoder_weights_path", type=str, default=None,
                    help="Path to Stage 1 checkpoint for loading NeighborTfsEncoder weights")
parser.add_argument("--w_fk", type=float, default=5.0,
                    help="Weight for FK-parent priority in weighted sampling")
parser.add_argument("--w_sim", type=float, default=2.0,
                    help="Weight for similarity priority in weighted sampling")
parser.add_argument("--w_recency", type=float, default=1.0,
                    help="Weight for recency priority in weighted sampling")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for weighted sampling")
```

### 3. Rename `data` to `raw_data` for HeteroData variable

The variable returned by `make_pkey_fkey_graph` was previously called `data` but
gets shadowed by the `RelGTTokens` dict. Rename to `raw_data`:

```python
raw_data, col_stats_dict = make_pkey_fkey_graph(...)
```

### 4. Add Stage 2 embedding precomputation block

After `make_pkey_fkey_graph`, before `RelGTTokens` construction:

```python
# Stage 2: pre-compute node embeddings from Stage 1 encoder weights
node_embeddings = None
if args.train_stage == "similarity_resample":
    if args.encoder_weights_path is None:
        raise ValueError("--encoder_weights_path is required for similarity_resample stage")
    node_type_map = {nt: idx for idx, nt in enumerate(raw_data.node_types)}
    col_names_dict = {
        nt: raw_data[nt].tf.col_names_dict
        for nt in raw_data.node_types if hasattr(raw_data[nt], 'tf')
    }
    node_embeddings = precompute_node_embeddings(
        data=raw_data,
        encoder_weights_path=args.encoder_weights_path,
        node_type_map=node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
        channels=args.channels,
        device=f"cuda:{local_rank}",
    )
```

### 5. Update `RelGTTokens` construction

Pass `raw_data` and the new sampling params:

```python
data = {
    split: RelGTTokens(
        data=raw_data,                  # CHANGED: was data
        task=task,
        K=args.num_neighbors,
        split=split,
        undirected=True,
        precompute=args.precompute,
        precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
        num_workers=args.num_workers,
        train_stage=args.train_stage,
        node_embeddings=node_embeddings,  # NEW
        w_fk=args.w_fk,                  # NEW
        w_sim=args.w_sim,                # NEW
        w_recency=args.w_recency,        # NEW
        temperature=args.temperature)     # NEW
        for split in ["train", "val", "test"]
    }
```

### 6. Add wandb logging for best test metrics

In both the `finetune` and `similarity_resample` blocks, after computing
`test_metrics`, add:

```python
wandb.log({
    **{f"best_val_{k}": v for k, v in val_metrics.items()},
    **{f"best_test_{k}": v for k, v in test_metrics.items()}
})
```

### 7. Add `similarity_resample` training block

After the `finetune` block, add an `elif args.train_stage == "similarity_resample":`
block with the same training loop structure:
- Same epoch loop calling `train_supervised` and `test`
- Same best model tracking with `val_metrics[tune_metric]`
- Saves checkpoint to `similarity_resampled.pt` (not `finetuned.pt`)
- Final evaluation on val + test with wandb logging
- Print prefix `[Stage2]` for log clarity

---

## New file: `run_pipeline.sh`

Two-stage bash script that runs Stage 1, verifies the checkpoint was saved, then
runs Stage 2 with configurable scoring weights.

Usage:
```bash
# Default: rel-f1 / driver-top3
./run_pipeline.sh

# Custom: dataset, task, batch_size, stage1_epochs, stage2_epochs
./run_pipeline.sh rel-avito ad-ctr 64 15 10

# Override scoring weights via env vars
W_FK=10.0 W_SIM=3.0 TEMPERATURE=0.5 ./run_pipeline.sh
```

See `run_pipeline.sh` in the repo for the full script.

---

## What does NOT change

- `encoders.py` — NeighborTfsEncoder is used as-is
- `model.py` — no model architecture changes
- `local_module.py` — untouched
- HDF5 schema — same datasets (types, indices, hops, times, edges, edges_offsets)
- `collate()` — same batching logic
- `__getitem__()` — same sample retrieval

The downstream model consumes exactly the same data format. Only the sampling
distribution changes.
