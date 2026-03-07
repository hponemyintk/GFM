# Task 2: Separate Backbone from Prediction Head

## Objective

Split `RelGT` in `model.py` so the backbone (encoder + transformer layers) is decoupled from the prediction head. The backbone returns a representation; heads are attached separately.

## Context

Currently `RelGT` has a single `self.head` MLP (line 258-263 in `model.py`) with fixed `out_channels`. For foundation model pretraining, we need:
- `encode()` returning the FULL sequence `[B, K, channels]` for masked token prediction
- `forward()` returning seed-only `[B, channels]` → head → prediction (backward compat)
- The task-specific head to be optional

## Files to Modify

### 1. `model.py` — `RelGT`

**Add `encode()` method — returns FULL sequence:**

```python
def encode(self,
           neighbor_types, node_indices, neighbor_hops,
           neighbor_times, grouped_tf_dict,
           edge_index=None, batch=None):
    """Returns full backbone sequence [B, K, channels] without head.

    IMPORTANT: Returns the full K-length sequence, not just the seed node.
    This is needed for masked token prediction (Task 5) where we predict
    at all masked positions, not just position 0.
    """
    neighbor_tfs = self.layer_norm_tfs(self.tfs_encoder(grouped_tf_dict, neighbor_types))
    neighbor_types = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
    neighbor_hops = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
    neighbor_times = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
    neighbor_subgraph_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))

    cat_list = [neighbor_types, neighbor_hops, neighbor_times, neighbor_tfs, neighbor_subgraph_pe]
    if self.ablate_idx is not None:
        cat_list.pop(self.ablate_idx)
    x_set = torch.cat(cat_list, dim=-1)
    x_set = self.in_mixture(x_set)

    x = x_set[:, 0, :]  # seed for global attention
    for i, conv in enumerate(self.convs):
        x_set = conv(x_set, x, node_indices)
        x_set = self.ffs[i](x_set)

    return x_set  # [B, K, channels] — full sequence
```

**NOTE on shape**: Currently `self.ffs[i]` contains `BatchNorm1d(hidden_channels * h_times)` which expects `[B, C]` not `[B, K, C]`. The current code works because after `conv()` the output for `full` mode is `[B, 2*channels]` (local + global concatenated), which `ffs` reduces to `[B, channels]`. So the output after ffs is `[B, channels]`, NOT `[B, K, channels]`.

**This means `encode()` should return `[B, channels]`** — the seed representation after the transformer layers. The masking head in Task 5 needs to work with this. Re-examine whether we need the full sequence or just the seed. If masked token prediction requires per-position predictions, we need to rethink where to tap into the network (before vs after the RelGTLayer).

**Resolution**: Two encode modes:
```python
def encode(self, ..., return_full_sequence=False):
    ...
    x_set = self.in_mixture(x_set)  # [B, K, channels]

    if return_full_sequence:
        # Return pre-RelGTLayer full sequence for masked token prediction
        # The masking head operates on this directly
        return x_set

    x = x_set[:, 0, :]
    for i, conv in enumerate(self.convs):
        x_set = conv(x_set, x, node_indices)
        x_set = self.ffs[i](x_set)
    return x_set  # [B, channels] after RelGTLayers
```

**Refactor `forward()` to use `encode()`:**
```python
def forward(self, neighbor_types, node_indices, neighbor_hops,
            neighbor_times, grouped_tf_dict, edge_index=None, batch=None):
    x_set = self.encode(neighbor_types, node_indices, neighbor_hops,
                        neighbor_times, grouped_tf_dict, edge_index, batch)
    return self.head(x_set)
```

**Make `self.head` optional:**
- Add `out_channels: Optional[int] = None` parameter
- Only create `self.head` if `out_channels` is not None
- `forward()` raises error if head is None
- Foundation model code calls `encode()` directly

### 2. `model.py` — New `MaskedTokenHead` class

```python
class MaskedTokenHead(nn.Module):
    """Predicts masked neighbor token features from backbone representations.

    Operates on the FULL sequence [B, K, channels] before RelGTLayers.
    Uses its own small transformer to process the sequence, then predicts
    at masked positions.
    """
    def __init__(self, channels: int, num_node_types: int):
        super().__init__()
        self.type_predictor = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, num_node_types),
        )
        self.feature_predictor = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x_set):
        """
        Args:
            x_set: [B, K, channels] full sequence from encode(return_full_sequence=True)
        Returns:
            dict with "type_pred" [B, K, num_types] and "feature_pred" [B, K, channels]
        """
        return {
            "type_pred": self.type_predictor(x_set),
            "feature_pred": self.feature_predictor(x_set),
        }
```

## DDP Notes

- `MaskedTokenHead` is a standard `nn.Module` with parameters — DDP handles it automatically
- Wrap both backbone and masking head in DDP, or combine into a single wrapper module
- The `encode(return_full_sequence=True)` path skips RelGTLayers (and therefore skips global attention / VQ). This means during masked token pretraining, the VQ codebook is NOT used. This is actually fine — the masking head trains the encoders and mixture MLP; the VQ codebook can be initialized/trained in a second phase or used only at fine-tune time.
- Alternatively, if you want global attention during pretraining, use `encode(return_full_sequence=False)` and have the masking head work with the seed representation only. But this limits masked prediction to seed-only.

## What NOT to Change

- Do NOT modify `RelGTLayer` — it's already generic (DDP fixes in Task 7)
- Do NOT modify any encoder — they're already table-agnostic
- Do NOT remove `self.head` — keep it for backward compatibility with `main_node_ddp.py`

## Testing

See `agent_tasks/testing_plan.md` — Phase 1, Task 2 tests.

## Dependencies

- None (can be done in parallel with Task 1)
- Task 5 (masking) depends on the encode() interface defined here
- Task 7 (DDP fixes) modifies RelGTLayer but not the backbone interface
