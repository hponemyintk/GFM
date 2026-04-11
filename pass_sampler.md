# PASS Sampler Implementation Summary

Implementation of Performance-Adaptive Sampling Strategy (PASS) from Yoon et al., KDD 2021, adapted for heterogeneous graphs and integrated with RelGT.

## Overview

When `--sampler pass` is set, the PASS sampler completely replaces RelGT's default random precomputed sampling. PASS learns which neighbors are most informative for each seed node, while RelGT only processes the subgraph it receives — it has no say in neighbor selection.

## Architecture

### Candidate Scope Precomputation (CPU, offline)

Before training, `RelGTTokensOnline` precomputes a candidate scope for each seed node via 2-hop BFS on the heterogeneous graph. The BFS is capped by a budget cascade to keep compute tractable:

```
full neighborhood (can be 100K+)
  -> 5000 max 1-hop neighbors (random subsample)
    -> per 1-hop neighbor: 1000 max 2-hop expansion (random subsample)
      -> dedup + temporal filter (time <= seed_time) + merge
        -> 1200 scope (--sample_scope, random subsample, stored in HDF5)
          -> 300 final (--num_neighbors K, learned PASS selection on GPU)
```

The 2-hop BFS (`gather_1_and_2_hop_vectorized`) uses CSR adjacency built with `undirected=True`, so both FK->PK and PK->FK edges are traversed. Deduplication is enforced at every level:
- 1-hop: CSR is deduplicated at build time via composite keys
- 2-hop: `n2_connecting` dict keyed by `(type, idx)` naturally deduplicates
- Cross-hop: seed and all 1-hop nodes are explicitly excluded from 2-hop candidates

If a seed has fewer than 1200 neighbors within 2 hops, all are kept and `scope_count` records the actual count. Zero-neighbor seeds fall back to random graph-wide node sampling (hop=3 sentinel).

### PASSHeteroSampler (GPU, per training step)

`PASSHeteroSampler` (in `pass_sampler.py`) implements the PASS policy adapted for heterogeneous graphs.

**Shared encoder**: The sampler holds a reference to RelGT's `tfs_encoder` (per-type TorchFrame encoders). This is not a copy — both the sampler and RelGT share the same parameters, included once in the optimizer via `model.parameters()`.

**Learnable parameters** (sampler's own, via `own_parameters()`):
- `Ws` (embed_dim x hidden_dim): projection matrix for dot-product attention (paper Eq. 4)
- `as_` (2-element vector): learnable weights balancing importance vs. uniform sampling (paper Eq. 6)
- `type_embeddings` (num_types x embed_dim): added to encoded features for cross-type attention

**Sampling policy** (paper Eq. 4-7):
1. Encode all scope candidates and seed through shared `tfs_encoder` + `type_embeddings`
2. Project seed and candidates via `Ws`, compute dot-product importance scores: `q_imp = (Ws @ h_i) . (Ws @ h_j)`
3. Compute uniform scores: `q_rand = 1/N(i)`
4. Combine: `q_tilde = softmax(as_)[0] * q_imp + softmax(as_)[1] * q_rand`
5. Mask padding positions, clamp to non-negative, normalize
6. Sample K-1 indices via `torch.multinomial` **without replacement** (unique neighbors). Falls back to with-replacement only for seeds with fewer valid candidates than K-1.

**Output**: Hard index selections `[B, K-1]` into the scope. PASS probabilities are NOT passed to RelGT — they are only used for REINFORCE gradient computation.

## Training Integration (train_pass)

The training loop in `main_node_ddp.py` orchestrates PASS and RelGT:

1. **Encode** candidates and seeds through shared `tfs_encoder` (inside AMP context)
2. **Select** K-1 neighbors via PASS attention policy (outside AMP — inputs cast to float32 for `Ws` matmul)
3. **Assemble** subgraph: gather selected types/indices/hops/times, prepend seed -> [B, K] nodes
4. **Reuse** pre-encoded embeddings as `preencoded_tfs` (avoids double-encoding)
5. **Build** `edge_index` on CPU from CSR adjacency for the selected subgraph
6. **Forward** through `model.module.forward_with_preencoded_tfs()` (bypasses DDP hooks since `tfs_encoder` was used outside the model forward)
7. **Task loss** backward through RelGT -> captures `x_set.grad`
8. **REINFORCE loss**: uses `x_set.grad` as upstream signal, computes `log_prob * h_j` for sampled actions, dots with upstream gradient (paper Theorem 4.1). Source/target embeddings are detached so REINFORCE gradients flow only to `Ws` and `as_`, not back through `tfs_encoder`.
9. **Manual all_reduce** across ranks (since DDP hooks were bypassed)
10. **Single optimizer step** updates both model and sampler parameters

## Evaluation (test_pass)

Same pipeline as training but:
- `@torch.no_grad()` — no REINFORCE computation
- Uses `model.module` directly (unwrapped from DDP)
- Predictions gathered across ranks via `dist.gather_object`

## Key Design Decisions

- **Shared tfs_encoder**: PASS and RelGT share the same feature encoders. The `tfs_encoder` gets gradients from the task loss (through `forward_with_preencoded_tfs`), not from REINFORCE (source/target are detached in the PASS forward).
- **DDP bypass**: `train_pass` calls `model.module` directly and manually all-reduces all gradients, because the `tfs_encoder` is used outside the DDP-wrapped forward, which would cause DDP to flag it as unused.
- **Decoupled attention**: PASS decides **who** is in the subgraph; RelGT's local attention decides **how much** to attend to each member. The two attention mechanisms are independent.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | `random` | `random` (precomputed) or `pass` (learned) |
| `--sample_scope` | `1200` | Candidate scope size per seed (precomputed) |
| `--pass_hidden_dim` | `32` | Hidden dim for PASS attention projection |
| `--num_neighbors` | `300` | K: final subgraph size (seed + K-1 selected) |
