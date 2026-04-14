# PASS Sampler Integration — Reimplementation Guide

This document describes every change made on the `PASS` branch relative to baseline commit `fedd4fe5d0066baad912037925942368d11076e0` (original RelGT). Pass this file to a fresh Claude session on a clean branch and ask it to recreate the PASS sampler integration from scratch.

## Goal

Integrate the **Performance-Adaptive Sampling Strategy (PASS)** learned sampler (Yoon et al., KDD 2021, LinkedIn) into RelGT as an alternative to the default random precomputed neighbor sampler. When `--sampler pass` is used, PASS learns — via REINFORCE — which neighbors of each seed are most informative, fully replacing RelGT's random K-sampling.

**Important constraint for the reimplementation:** keep `utils.py` as close to the original as possible. Do not mutate the existing random sampling logic (`RelGTTokens`, `_process_one_seed`, etc.). Instead, add parallel structures (new dataset class, new worker, new helper) alongside the originals. The goal is to add PASS *on top of* the original source rather than replacing its machinery.

---

## Scope of changes

Five files touched:

| File | Change type |
|------|-------------|
| `pass_sampler.py` | **New file** — PASS sampler module |
| `pass_sampler.md` | **New file** — design doc (optional to regenerate) |
| `utils.py` | **Additive only** — new worker function, new helper, new dataset class. Original `RelGTTokens` left untouched. |
| `model.py` | **Add new method** `forward_with_preencoded_tfs` on `RelGT`. Existing `forward` untouched. |
| `main_node_ddp.py` | **Additive + conditional switch** — new CLI args, branching dataset construction, new `train_pass`/`test_pass` functions, wiring in main loop. |

No files are deleted. No existing function bodies on the hot random-sampling path are modified.

---

## Architecture (what runs at each stage)

### Stage 1: Candidate scope precomputation (CPU, offline, once per split)

Before training, a new `RelGTTokensOnline` dataset precomputes a **candidate scope** per seed node via the existing 2-hop BFS machinery in `utils.py`. Unlike the original `RelGTTokens` (which precomputes final K-sized subgraphs), this dataset stores a larger pool of `S = sample_scope` candidates. The actual K selection happens on GPU during training.

Budget cascade:

```
full 2-hop neighborhood
  -> existing gather_1_and_2_hop_vectorized (with its internal caps of
     5000 1-hop and 1000 2-hop-per-1-hop, deduped, temporal-filtered)
  -> capped at S = sample_scope (default 3000), random subsample if over
  -> per seed: stored in HDF5 with {scope_types, scope_indices, scope_hops,
     scope_times, scope_count, seed_type, seed_index, seed_time}
```

If a seed has fewer than `S` valid 2-hop neighbors, all are kept and `scope_count` records the actual count. Zero-neighbor seeds fall back to random graph-wide nodes with `hop=3` sentinel — same fallback that `_process_one_seed` already uses.

### Stage 2: PASS selection (GPU, per training step)

`PASSHeteroSampler` (new module) holds:
- A shared reference to RelGT's `tfs_encoder` (per-type TorchFrame encoders — **not** a copy; both the sampler and RelGT share parameters, included once in the optimizer)
- Its own learnable params: `Ws` (embed_dim × hidden_dim), `as_` (2-element vector), `type_embeddings` (num_types × embed_dim)

Per step, for each seed:
1. Encode all `S` scope candidates and the seed through shared `tfs_encoder` + type_embeddings
2. Project through `Ws`, compute dot-product importance `q_imp = (Ws·h_i) · (Ws·h_j)`
3. Compute uniform baseline `q_rand = 1/N(i)`
4. Combine: `q_tilde = softmax(as_)[0]*q_imp + softmax(as_)[1]*q_rand`
5. Mask padding positions, clamp non-negative, normalize via `Categorical`
6. Sample `K-1` indices via `torch.multinomial` **without replacement** (fallback to with-replacement only when a row has fewer valid candidates than K-1)
7. Prepend seed → final `[B, K]` subgraph

### Stage 3: RelGT forward on selected subgraph

A new `forward_with_preencoded_tfs` method on `RelGT` skips `tfs_encoder` and consumes the already-encoded `[B, K, channels]` tensor from PASS (avoids double-encoding). All other encoders (type, hop, time, pe) run normally. Returns `x_set` pre-head so the training loop can capture its gradient for REINFORCE.

### Stage 4: REINFORCE update

Task loss backprops through RelGT down to `x_set`, whose gradient is captured via `.retain_grad()`. PASS's REINFORCE loss uses that gradient as the upstream reward signal per Yoon et al. Theorem 4.1:

```
∇θ L_sample ≈ (dL/dh) · E[ ∇θ log q(j|i) · h_j ]
```

Source/target embeddings are **detached** in PASS's forward so REINFORCE gradients flow only to `Ws` and `as_`, never back through `tfs_encoder` (the encoder is updated by the task loss, not REINFORCE).

### Stage 5: DDP bypass

Because `tfs_encoder` is called outside the DDP-wrapped `forward`, DDP would flag those params as "unused" and desync. The training loop therefore calls `model.module` directly and **manually `all_reduce`s** gradients across ranks for both model params and sampler own params before `optimizer.step()`.

---

## File-by-file changes

### 1. `pass_sampler.py` (new file, 272 lines)

Create `PASSHeteroSampler(nn.Module)` with:

**`__init__(tfs_encoder, num_types, embed_dim, hidden_dim=32)`**
- Stores reference to `tfs_encoder` (RelGT's existing module, not a copy)
- `self.type_embeddings = nn.Embedding(num_types, embed_dim)`
- `self.Ws = nn.Parameter(torch.zeros(embed_dim, hidden_dim))` with `xavier_uniform_(gain=1.414)`
- `self.as_ = nn.Parameter(torch.FloatTensor([0.5, 0.5]))`
- State slots `batch_selected`, `batch_dist`, `selected_embeds` (populated in forward, used by reinforce_loss)

**`own_parameters()`** — returns `[self.Ws, self.as_, *self.type_embeddings.parameters()]`. Critical: callers must use this to avoid double-counting `tfs_encoder` params in the optimizer.

**`encode_candidates(scope_batch, data, device)`**
- Inputs: `scope_batch["scope_types"]` `[B, S]`, `scope_batch["scope_indices"]` `[B, S]`, `data` (HeteroData with `.tf` per node type)
- Allocates `encoded_flat = torch.zeros(B*S, embed_dim, device=device)`
- For each `(t_int, encoder)` in `self.tfs_encoder.encoders.items()`:
  - `t_idx = self.tfs_encoder.node_type_map[t_int]`
  - `mask = (scope_types == t_idx)`; skip if empty
  - Gather `local_idxs = scope_indices[mask]`
  - `tf = data[t_int].tf[local_idxs].to(device=device)`
  - Sanitize NaN/Inf in every `tf.feat_dict[st]` tensor via `torch.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)`
  - `out = encoder(tf)`; if `out.dim() == 3 and out.shape[1] == 1`, squeeze dim 1
  - Scatter back via `flat_positions = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)`; assign with `.to(encoded_flat.dtype)` (dtype cast critical for AMP — encoder output may be bf16 while destination is float32)
- Reshape to `[B, S, embed_dim]`, add `self.type_embeddings(scope_types.to(device))`, return

**`encode_seeds(scope_batch, data, device)`** — same pattern for `seed_type` `[B]` and `seed_index` `[B]`, output `[B, embed_dim]`.

**`forward(seed_embeds, candidate_embeds, scope_counts, K)`**
- `B, S, D = candidate_embeds.shape`
- `source = seed_embeds.unsqueeze(1).expand(B, S, D).reshape(B*S, D)`
- `target = candidate_embeds.reshape(B*S, D)`
- **Detach** source and target, cast to `self.Ws.dtype` before matmul: `ss = torch.mm(source.detach().to(self.Ws.dtype), self.Ws)`, same for `tt`. The detach implements Theorem 4.1 (h_i, h_j treated as constants in REINFORCE); the dtype cast prevents bf16 @ f32 errors outside autocast.
- `q_imp = torch.bmm(ss.unsqueeze(1), tt.unsqueeze(2)).squeeze(2).reshape(B, S)`
- `q_rand = (1.0 / scope_counts.unsqueeze(1).clamp(min=1).float()).expand(B, S)`
- `as_w = F.softmax(self.as_, dim=0)`; `q_tilde = as_w[0]*q_imp + as_w[1]*q_rand`
- Build `pad_mask = (arange(S) >= scope_counts.unsqueeze(1))` on device; `q_tilde.masked_fill_(pad_mask, 0.0)`
- `q_tilde = q_tilde.clamp(min=0.0) + 1e-9`; re-apply `masked_fill(pad_mask, 0.0)` to re-zero padding after epsilon
- `dist = torch.distributions.Categorical(probs=q_tilde)`; `probs = dist.probs`
- Sample `num_select = K - 1` **without replacement** via `torch.multinomial(probs, num_select, replacement=False)`. Guard: if `scope_counts.min() < num_select`, sample with replacement for the whole batch, then overwrite rows where `scope_counts >= num_select` with without-replacement samples.
- Save `self.batch_selected = selected`, `self.batch_dist = dist`, `self.selected_embeds = torch.gather(candidate_embeds, 1, selected.unsqueeze(-1).expand(-1, -1, D))`
- Return `(selected, dist)`

**`reinforce_loss(loss_up)`**
- `loss_up` is `[B, embed_dim]` — the gradient `dL/dh` (from `x_set.grad.mean(dim=1)` or equivalent — see training loop below for exact shape)
- `logp = self.batch_dist.log_prob(self.batch_selected.T).T` → `[B, K-1]`
- `sel_embeds = self.selected_embeds.detach()` → `[B, K-1, embed_dim]`
- `X = (logp.unsqueeze(2) * sel_embeds).mean(dim=1)` → `[B, embed_dim]`
- Cast both `loss_up` and `X` to `.float()` (mixed precision safety) then `torch.bmm(loss_up.unsqueeze(1).float(), X.unsqueeze(2).float())` → `[B, 1, 1]`
- Return `.mean()`

Full docstrings referencing paper equations 4-7 should match `pass_sampler.md`.

---

### 2. `utils.py` (additive)

**Do not modify** `_process_one_seed`, `RelGTTokens`, `build_adjacency_csr`, `gather_1_and_2_hop_vectorized`, `_sample_fallback_nodes`, `init_worker_globals`, `_build_all_nodes_compact`, or any of the `GLOBAL_*` module-level state. Add the new pieces below *alongside* the originals.

#### (a) New worker function: `_process_one_seed_scope`

Mirrors `_process_one_seed` but returns the full candidate scope instead of K-selected subgraph. Place near `_process_one_seed`.

Signature takes tuple `(sample_scope, seed_node_type, seed_node_idx, seed_time, seed_val, row_idx)`. Same preamble: seed `random`/`np.random`, assert CSR adjacency in `GLOBAL_ADJ`, call `gather_1_and_2_hop_vectorized(GLOBAL_ADJ, GLOBAL_TIME_ARRAYS, GLOBAL_TYPE_TO_IDX, GLOBAL_NODE_TYPES, seed_node_type, seed_node_idx, seed_time)`. Raise if legacy dict-of-sets adjacency detected.

Flow:
- `T_hat_list = list(T_hat)`
- `one_hop = [n for n in T_hat_list if n[2] == 1]`
- `two_hop = [n for n in T_hat_list if n[2] == 2]`
- `combined = one_hop + two_hop`; `actual_count = len(combined)`
- If `actual_count >= sample_scope`: `chosen = random.sample(combined, sample_scope)`
- Elif `actual_count > 0`: `chosen = combined` (will be zero-padded below)
- Else: fallback via `_sample_fallback_nodes(GLOBAL_ALL_NODES, sample_scope, rng=random)`, build tuples `(ft, fi, 3, rel_time, None)` where `rel_time = (seed_time - ft_time) / (60*60*24)` if `ft` in time_arrays else 0; `actual_count = len(chosen)`

Output arrays (all of length `sample_scope`, zero-padded):
- `out_types` int16, `out_indices` int32, `out_hops` int8, `out_times` float32
- Fill by iterating `chosen[:sample_scope]`: `out_types[j] = GLOBAL_TYPE_TO_IDX[t_str]`, etc.
- `scope_count = min(actual_count, sample_scope)`
- `seed_type_idx = GLOBAL_TYPE_TO_IDX[seed_node_type]`
- Return `(row_idx, seed_type_idx, seed_node_idx, seed_time, scope_count, out_types, out_indices, out_hops, out_times)`

#### (b) New helper: `build_batch_edge_index_from_selected`

Builds a batched `edge_index` on CPU for `[B, K]` selected nodes using CSR adjacency. Same induced-subgraph logic as inside `_process_one_seed` (lines ~504-543 in baseline), but batched.

Signature: `(neighbor_types, neighbor_indices, csr_adj, idx_to_type, type_to_idx)`

Logic:
- `nt_np = neighbor_types.cpu().numpy()`; `ni_np = neighbor_indices.cpu().numpy()`
- `all_edges = []`; `batch_vec_parts = []`; `node_offset = 0`
- For each sample `b` in batch:
  - Build `local_composite = {(int(nt_np[b,j]) << 32) | int(ni_np[b,j]): j for j in range(K)}`
  - `local_ck_array = np.array(list(local_composite.keys()), dtype=np.int64)`
  - For each `j_src in range(K)`:
    - `t_str = idx_to_type[int(nt_np[b, j_src])]`; `i = int(ni_np[b, j_src])`
    - `csr_nt = csr_adj[t_str]`; `s = int(csr_nt["offsets"][i])`; `e = int(csr_nt["offsets"][i+1])`
    - Skip if `s == e`
    - Compute composite keys for neighbors via `(nbr_types_arr.astype(int64) << 32) | nbr_indices_arr.astype(int64)`
    - `mask = np.isin(nbr_ck, local_ck_array)`
    - For each matching `ck`, append `(j_src + node_offset, local_composite[int(ck)] + node_offset)`
  - Append to `all_edges`, extend `batch_vec_parts` with `np.full(K, b)`, bump `node_offset += K`
- Concatenate edges to `[2, total_E]`, return `(edge_index.long(), batch_vec.long())`. Empty fallback returns `torch.zeros((2, 0), dtype=torch.long)`.

#### (c) New dataset class: `RelGTTokensOnline(Dataset)`

Place at the end of `utils.py` (after `RelGTTokens`). Precomputes scopes to HDF5, loads per-item on `__getitem__`, similar to `RelGTTokens` but scope-based.

**`__init__(data, task, K, sample_scope=512, split="train", undirected=True, num_workers=None, precomputed_dir=None, train_stage="finetune")`**

Mirrors `RelGTTokens.__init__`:
- Store args, resolve `table_input = get_node_train_table_input(table, task)`, extract `node_type`, `node_idxs`, `target`, `time`, `transform`
- `self.node_types = self.data.node_types`; build `node_type_to_index` / `index_to_node_type`
- `self.max_neighbor_hop = 2 + 1`
- Call `self._create_global_mappings()` — identical to `RelGTTokens` version: walks every node in every type, builds `type_local_to_global` and `global_to_type_local` dicts keyed by `(type_idx, local_idx)` / global int. Use `data[nt]['x'].size(0)` if `'x'` present else `data[nt].num_nodes`.
- `self.precomputed_path = self._construct_precomputed_path()` — builds `{precomputed_dir}/scope_{sample_scope}/{split}.h5`, `os.makedirs` parent
- `self.train_stage = train_stage`
- Rank-aware precompute: `rank = int(os.environ.get("RANK", 0))`; if file exists print skip; elif rank 0 call `self._precompute_scope()`; if `torch.distributed.is_initialized()`, barrier with `device_ids=[int(os.environ.get("LOCAL_RANK", 0))]`; non-rank-0 raise if file still missing

**`get_global_index(type_idxs, local_idxs)`** — same as `RelGTTokens`: loops pairs, looks up in `self.type_local_to_global`.

**`__len__`** — `len(self.node_idxs)`

**`_precompute_scope()`**
- `data_cpu = self.data.to("cpu")`
- `csr_adj = build_adjacency_csr(data_cpu, undirected=self.undirected)`
- Build `time_arrays = {nt: data_cpu[nt].time.numpy().copy() for nt in data_cpu.node_types if hasattr(data_cpu[nt], "time")}`
- `all_nodes_compact = _build_all_nodes_compact(data_cpu)`
- `num_workers = self.num_workers or max(1, min(cpu_count()-1, total))`
- Build `all_tasks`: for each `(i, node_idx_t)`, compute `seed_t = self.time[i].item() if self.time is not None else 0.0`, `seed_val = hash((self.node_type, node_idx, seed_t, self.sample_scope)) & 0xffffffff`, append `(self.sample_scope, self.node_type, node_idx, seed_t, seed_val, i)`
- Preallocate numpy arrays: `all_scope_types[total, S] int16`, `all_scope_indices[total, S] int32`, `all_scope_hops[total, S] int8`, `all_scope_times[total, S] float32`, `all_scope_counts[total] int16`, `all_seed_types[total] int16`, `all_seed_indices[total] int32`, `all_seed_times[total] float32`
- `ctx = get_context("fork")`; `Pool(processes=num_workers, initializer=init_worker_globals, initargs=(csr_adj, all_nodes_compact, node_types, time_arrays))`
- `imap_unordered(_process_one_seed_scope, all_tasks, chunksize=chunksize)` with `chunksize = max(1, total // (num_workers * 4))`, wrap in tqdm
- Write each result into preallocated arrays by `row_idx`
- Atomic HDF5 write: `tmp_path = self.precomputed_path + f".tmp.{os.getpid()}"`, create all 8 datasets, `os.rename` to final, unlink tmp on exception

**`__getitem__(idx)`**
- Open HDF5 read-only, return tuple `(sample, label)` where sample is a dict of torch tensors:
  - `scope_types` long, `scope_indices` long, `scope_hops` long, `scope_times` float, `scope_count` int, `seed_type` int, `seed_index` int, `global_idx = idx`
- `label = self.target[idx] if self.target is not None else None`

**`collate(batch)`**
- Unzip `samples, labels`; stack each tensor field across batch (`torch.stack`) or wrap as `torch.tensor` for scalar fields
- Output dict with batched `scope_types [B,S]`, `scope_indices [B,S]`, `scope_hops [B,S]`, `scope_times [B,S]`, `scope_count [B]`, `seed_type [B]`, `seed_index [B]`, `global_idx [B]`, `labels` (stacked if present, else None)

---

### 3. `model.py` — `RelGT.forward_with_preencoded_tfs`

Add a new method on `RelGT`, directly after the existing `forward`. Copy-paste `forward` and replace the `tfs_encoder` call with the preencoded input.

```python
def forward_with_preencoded_tfs(
    self,
    neighbor_types,
    node_indices,
    neighbor_hops,
    neighbor_times,
    preencoded_tfs,              # [B, K, channels] — already through tfs_encoder
    edge_index=None,
    batch=None,
):
    neighbor_tfs = self.layer_norm_tfs(preencoded_tfs)
    neighbor_types = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
    neighbor_hops = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
    neighbor_times = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
    neighbor_subgraph_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))

    cat_list = [neighbor_types, neighbor_hops, neighbor_times, neighbor_tfs, neighbor_subgraph_pe]
    if self.ablate_idx is not None:
        cat_list.pop(self.ablate_idx)
    x_set = torch.cat(cat_list, dim=-1)
    x_set = self.in_mixture(x_set)

    x = x_set[:, 0, :]
    for i, conv in enumerate(self.convs):
        x_set = conv(x_set, x, node_indices)
        x_set = self.ffs[i](x_set)

    return x_set       # pre-head — caller will apply self.head and capture x_set.grad
```

Do **not** modify the existing `forward`. Leave it alone. (The diff may show cosmetic whitespace-only edits to `forward`; those are unintentional — ignore them in the reimplementation.)

---

### 4. `main_node_ddp.py` — wiring

#### (a) Imports

```python
from utils import (
    GloveTextEmbedding, RelGTTokens, RelGTTokensOnline,
    build_adjacency_csr, build_batch_edge_index_from_selected,
)
from pass_sampler import PASSHeteroSampler
```

Keep the existing `from contextlib import nullcontext` etc. — no other import changes.

#### (b) New CLI args

Add to `argparse` block:

```python
parser.add_argument("--sampler", type=str, default="pass", choices=["random", "pass"],
                    help="Sampling strategy: 'random' (default precomputed) or 'pass' (learned PASS-GNN)")
parser.add_argument("--sample_scope", type=int, default=3000,
                    help="PASS: candidate scope size per seed node")
parser.add_argument("--pass_hidden_dim", type=int, default=128,
                    help="PASS: hidden dim for attention projection")
```

#### (c) Conditional dataset construction

Wrap the existing `data = { split: RelGTTokens(...) ... }` block:

```python
if args.sampler == "pass":
    data = {
        split: RelGTTokensOnline(
            data=data, task=task,
            K=args.num_neighbors,
            sample_scope=args.sample_scope,
            split=split,
            undirected=True,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.sampling_workers,
            train_stage=args.train_stage)
        for split in ["train", "val", "test"]
    }
    data_cpu = data["train"].data.to("cpu")
    csr_adj = build_adjacency_csr(data_cpu, undirected=True)
    idx_to_type = data["train"].index_to_node_type
    type_to_idx = data["train"].node_type_to_index
else:
    data = {
        split: RelGTTokens(
            data=data, task=task,
            K=args.num_neighbors,
            split=split,
            undirected=True,
            precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.sampling_workers,
            train_stage=args.train_stage)
        for split in ["train", "val", "test"]
    }
```

The `csr_adj`, `idx_to_type`, `type_to_idx` are module-level captures used inside `train_pass`/`test_pass`.

#### (d) Sampler construction + optimizer

After the `model = DDP(...)` line and before the optimizer:

```python
pass_sampler = None
if args.sampler == "pass":
    pass_sampler = PASSHeteroSampler(
        tfs_encoder=model.module.tfs_encoder,
        num_types=len(data["train"].node_types),
        embed_dim=args.channels,
        hidden_dim=args.pass_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pass_sampler.own_parameters()),
        lr=base_lr, weight_decay=args.weight_decay
    )
    hetero_data = data["train"].data       # raw HeteroData handle for tf lookup
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
```

Critical: the sampler shares `model.module.tfs_encoder` — do NOT wrap params with `list(...) + list(pass_sampler.parameters())` as that would double-register `tfs_encoder`. Use `own_parameters()`.

#### (e) `train_pass(epoch)` — new function

Full body (see diff for reference). Key steps per batch:

1. **Encode under autocast**: `amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()`. Inside `amp_ctx`, call `pass_sampler.encode_candidates(scope_batch, hetero_data, device)` and `pass_sampler.encode_seeds(...)`.
2. **Select outside autocast**: `scope_counts = scope_batch["scope_count"].to(device)`; `selected_idx, _ = pass_sampler(seed_embeds, candidate_embeds, scope_counts, K)`. The sampler's internal `Ws` matmul casts detached inputs to `Ws.dtype`, so it is safe to call outside `amp_ctx`.
3. **Gather scope fields to device**: `scope_types`, `scope_indices`, `scope_hops`, `scope_times`, `seed_type`, `seed_index`. Use `torch.gather(scope_*, 1, selected_idx)` to pull selected rows.
4. **Prepend seed** to build `[B, K]`: concat `seed_type.unsqueeze(1)` with `selected_types`, same for indices, and prepend `torch.zeros(B, 1)` for hops/times (seed is hop 0, relative time 0).
5. **Build preencoded_tfs**: `torch.cat([seed_embeds.unsqueeze(1), pass_sampler.selected_embeds], dim=1)` → `[B, K, D]`. Reusing the embeddings computed in step 1 avoids a second tfs_encoder pass.
6. **Build edge_index on CPU**: `build_batch_edge_index_from_selected(neighbor_types, neighbor_indices, csr_adj, idx_to_type, type_to_idx)`, move both to device.
7. **Compute node_indices**: take the first column of `neighbor_types` / `neighbor_indices` (the seed), convert to Python lists, call `data["train"].get_global_index(first_types, first_indices)`, wrap in `torch.tensor(..., dtype=torch.long, device=device)`.
8. **Forward under autocast**: `optimizer.zero_grad()`, then `x_set = model.module.forward_with_preencoded_tfs(neighbor_types, node_indices, neighbor_hops, neighbor_times, preencoded_tfs, edge_index=edge_index, batch=batch_vec)`. Call `x_set.retain_grad()`. Compute `pred = model.module.head(x_set)`, reshape with `.view(-1)` if `pred.size(1) == 1`, `task_loss = loss_fn(pred.float(), labels)`.
9. **Task backward**: `task_loss.backward()`.
10. **REINFORCE**: if `x_set.grad is not None`, `chain_grad = x_set.grad.detach()`, `sample_loss = pass_sampler.reinforce_loss(chain_grad)`, `sample_loss.backward()`.
    - Note: `chain_grad` has shape `[B, K, D]`. `reinforce_loss` expects `[B, D]`. **Bug caveat** — the current implementation passes the full `[B, K, D]` tensor, so `torch.bmm(loss_up.unsqueeze(1).float(), X.unsqueeze(2).float())` would shape-mismatch. In practice, `x_set.grad` is not None here, and callers observed it working because `loss_up[:, 0, :]` happens to be used... **Double-check this when reimplementing** — if shapes mismatch, apply `chain_grad = x_set.grad.mean(dim=1).detach()` or `chain_grad = x_set.grad[:, 0, :].detach()` (the seed token's gradient is the most natural choice for the paper's formulation).
11. **Manual all_reduce** (DDP bypass): if `world_size > 1`, loop over `list(model.parameters()) + list(pass_sampler.own_parameters())`, all_reduce each `.grad` with `op=dist.ReduceOp.AVG`.
12. **Clip + step**: `clip_grad_norm_(list(model.parameters()) + list(pass_sampler.own_parameters()), max_norm=1.0)`, `optimizer.step()`.
13. **Logging**: same wandb log as `train_supervised` plus `sample_loss` if computed.
14. Respect `adjusted_max_steps = max(1, args.max_steps_per_epoch // world_size)`.

Don't forget `train_sampler.set_epoch(epoch)` at the top and `pass_sampler.train()` alongside `model.train()`.

#### (f) `test_pass(loader, eval_model, epoch, desc)` — new function

`@torch.no_grad()` decorator. Mirror `train_pass` without backward/REINFORCE/optimizer:
- Use `eval_model` directly (already `model.module`)
- `pass_sampler.eval()`
- For each scope_batch: encode, select, gather, prepend seed, build preencoded_tfs, build edge_index, compute node_indices, `eval_model.forward_with_preencoded_tfs(...)` under `amp_ctx`, apply `eval_model.head`, apply task-type-specific postprocessing (regression clamp, sigmoid for classification)
- Collect `pred_list` and `idx_list` per batch (`.detach().float().cpu().numpy()`)
- Gather across ranks: `gathered = [None]*world_size if local_rank==0 else None`, `dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)`
- Rank 0: allocate `all_preds = np.full((len(loader.dataset),), -100.0)`, scatter from `gathered` by index, return
- Non-rank-0: return `None`

#### (g) Main loop wiring

Replace the fixed `train_supervised` / `test` calls with conditional pointers:

```python
_train_fn = train_pass if args.sampler == "pass" else train_supervised
_test_fn  = test_pass if args.sampler == "pass" else test

for epoch in range(1, args.epochs + 1):
    train_loss = _train_fn(epoch)
    dist.barrier()
    eval_model = model.module
    val_pred = _test_fn(loader_dict["val"], eval_model=eval_model, epoch=epoch, desc="Val")
    ...
```

Same for the final post-training evaluation (`final_val_preds`, `final_test_preds`).

#### (h) Checkpointing

On best-val improvement:
```python
torch.save(state_dict, os.path.join(output_path, "finetuned.pt"))
if args.sampler == "pass":
    torch.save(pass_sampler.state_dict(), os.path.join(output_path, "pass_sampler.pt"))
```

On reload after training:
```python
if args.sampler == "pass":
    sampler_sd = torch.load(os.path.join(output_path, "pass_sampler.pt"), map_location=device)
    own_keys = {k: v for k, v in sampler_sd.items() if not k.startswith("tfs_encoder.")}
    pass_sampler.load_state_dict(own_keys, strict=False)
```

Broadcast after load (DDP param sync):
```python
if args.sampler == "pass" and pass_sampler is not None:
    for p in pass_sampler.own_parameters():
        dist.broadcast(p.data, src=0)
```

---

## Verification checklist (after reimplementation)

1. `python main_node_ddp.py --sampler random ...` must work identically to baseline (regression test — the original random path is untouched).
2. `python main_node_ddp.py --sampler pass --dataset rel-hm --task user-churn --sample_scope 1200 --num_neighbors 300 ...` should:
   - Precompute HDF5 scopes on first run under `{cache_dir}/precomputed/{dataset}/{task}/scope_1200/{split}.h5`
   - Train with `train_loss` and `sample_loss` both logged to wandb
   - Produce val/test metrics
3. Gradient flow audit: after one training step,
   - `pass_sampler.Ws.grad` should be non-zero (from REINFORCE)
   - `pass_sampler.as_.grad` should be non-zero (from REINFORCE)
   - `pass_sampler.type_embeddings.weight.grad` should be non-zero
   - `model.module.tfs_encoder.*.grad` should be non-zero (from task loss, not from REINFORCE — source/target are detached)
4. Multi-GPU DDP run: verify losses converge consistently across ranks (manual all_reduce working).
5. AMP run (`--amp`): no dtype mismatch errors. Particular hotspots:
   - `encode_candidates`/`encode_seeds` scatter cast: `.to(encoded_flat.dtype)`
   - `forward` Ws matmul: `.to(self.Ws.dtype)` after detach
   - `reinforce_loss` bmm: `.float()` on both operands

---

## Known caveats / things to re-examine

1. **`chain_grad` shape in `train_pass`** — see step 10 above. Verify the shapes match `reinforce_loss`'s `[B, D]` expectation. The seed token's gradient (`x_set.grad[:, 0, :]`) is the theoretically cleanest choice.
2. **Whitespace in `model.py` `forward`** — current diff has trailing-whitespace-only edits. Skip those; add only `forward_with_preencoded_tfs`.
3. **Default hyperparams** — recent commits on the branch tweaked defaults for 8×A100 40GB runs (e.g. `sample_scope=3000`, `pass_hidden_dim=128`, `num_layers` reverted to 1 per RelGT paper). Don't treat these as load-bearing; they're tuning, not correctness. The paper default for the sampler hidden dim is 32.
4. **`RelGTTokensOnline` does not write a trailing newline** in the branch file — trivial, ignore.
5. **`_process_one_seed_scope` requires CSR adjacency** — the legacy dict-of-sets path is explicitly raised. The baseline `_process_one_seed` has the same requirement, so no new constraint.

---

## Paper reference

Yoon, M., Gervet, T., Shi, B., Niu, S., He, Q., & Yang, J. (2021). **Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks.** KDD 2021.

Key equations (4-7) implemented in `PASSHeteroSampler.forward`. Theorem 4.1 (REINFORCE gradient estimator) implemented in `PASSHeteroSampler.reinforce_loss`.
