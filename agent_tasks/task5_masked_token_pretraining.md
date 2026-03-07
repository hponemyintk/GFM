# Task 5: Masked Token Pretraining Objective

## Objective

Design and implement the self-supervised masked token prediction objective for pretraining. This replaces supervised task-specific losses and enables zero-shot transfer.

## Context

Inspired by the Relational Transformer (RT) paper (arXiv 2510.06377) which uses masked token prediction to pretrain on relational data and achieves 93% of supervised AUROC in zero-shot settings.

RelGT's 5-token decomposition per neighbor node is:
1. **Type** — table name (GloVe embedding)
2. **Hop** — hop distance (learned embedding)
3. **Time** — relative time (positional encoding)
4. **Features** — table-agnostic TensorFrame encoding (shared transformer with CLS readout)
5. **GNN PE** — subgraph structural position (GIN)

The masking strategy must work with this tokenization.

## Masking Strategy

### What to Mask

For each seed node's K-length neighbor sequence, randomly select `mask_ratio` (e.g., 15%) of the neighbor positions (NOT the seed at position 0). For each masked position, replace its token inputs with learned mask tokens.

**Mask the following for selected positions:**
- **Type**: Replace type index with a special `mask_type_idx` (already exists in `NeighborNodeTypeEncoder` as `self.mask_idx`)
- **Features**: Replace the TensorFrame row with zeros (or a learned mask embedding)
- **Hop**: Replace with a special mask hop value (e.g., `max_hop + 1`, already reserved)
- **Time**: Replace with a special mask time value (e.g., -1.0, already handled by `NeighborTimeEncoder` which detects `rel_time < 0`)

**Do NOT mask:**
- Position 0 (the seed node) — this is the prediction target
- GNN PE — this depends on the subgraph structure which remains intact
- The `edge_index` — structural connections are preserved

### Reconstruction Targets

For each masked position, predict:
1. **Type reconstruction**: Predict which table the masked neighbor came from
   - Target: original `neighbor_types[masked_positions]`
   - Loss: Cross-entropy over `num_node_types`
2. **Feature reconstruction**: Predict the CLS representation of the masked neighbor's features
   - Target: the feature encoder output for the ORIGINAL (unmasked) TensorFrame at that position
   - Loss: MSE or cosine similarity loss
   - This requires a two-pass approach: first encode unmasked features to get targets, then encode masked inputs for predictions

## Files to Create/Modify

### 1. New file: `masking.py`

**`apply_masking(batch, mask_ratio, mask_type_idx, max_hop)`**

```python
def apply_masking(batch: dict, mask_ratio: float = 0.15,
                  mask_type_idx: int = None, seed: int = None) -> Tuple[dict, dict]:
    """
    Apply random masking to a collated batch.

    Args:
        batch: output of RelGTTokens.collate()
        mask_ratio: fraction of neighbor positions to mask (0-1)
        mask_type_idx: the mask token index for types
        seed: optional random seed

    Returns:
        masked_batch: same structure as batch, with masked positions replaced
        mask_targets: dict with original values for masked positions
            - "mask_positions": [B, K] bool tensor (True = masked)
            - "original_types": [num_masked] long tensor
            - "original_feature_targets": [num_masked, channels] float tensor
    """
    B, K = batch["neighbor_types"].shape

    # Generate mask: don't mask position 0 (seed node)
    mask = torch.zeros(B, K, dtype=torch.bool)
    num_to_mask = max(1, int((K - 1) * mask_ratio))
    for b in range(B):
        positions = torch.randperm(K - 1)[:num_to_mask] + 1  # skip position 0
        mask[b, positions] = True

    # Save original values
    mask_targets = {
        "mask_positions": mask,
        "original_types": batch["neighbor_types"][mask].clone(),
        "original_hops": batch["neighbor_hops"][mask].clone(),
        "original_times": batch["neighbor_times"][mask].clone(),
    }

    # Create masked batch (clone to avoid modifying original)
    masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

    # Apply masks
    masked_batch["neighbor_types"][mask] = mask_type_idx
    masked_batch["neighbor_hops"][mask] = -1  # mask value for hops
    masked_batch["neighbor_times"][mask] = -1.0  # triggers mask in NeighborTimeEncoder

    # For TensorFrames: need to handle in collate or here
    # The grouped_tfs need masked positions' TFs replaced with zeros
    # This is the trickiest part — see implementation notes below

    return masked_batch, mask_targets
```

**Implementation note on TensorFrame masking:**
The `grouped_tfs` in the batch dict groups TensorFrames by node type. Masking requires:
1. Identify which entries in `grouped_tfs` correspond to masked positions
2. Replace those TensorFrame rows with zero-filled TFs of the same schema
This requires cross-referencing `grouped_indices` (which maps flat positions to (batch, neighbor) pairs) with the mask.

**`compute_reconstruction_loss(model_output, mask_targets, num_node_types)`**

```python
def compute_reconstruction_loss(predictions: dict, targets: dict,
                                 num_node_types: int) -> torch.Tensor:
    """
    Compute masked token reconstruction loss.

    Args:
        predictions: output of MaskedTokenHead
            - "type_pred": [B, K, num_node_types] logits
            - "feature_pred": [B, K, channels] predicted features
        targets:
            - "mask_positions": [B, K] bool
            - "original_types": [num_masked] long
            - "original_feature_targets": [num_masked, channels] float

    Returns:
        scalar loss
    """
    mask = targets["mask_positions"]

    # Type prediction loss (cross-entropy)
    type_logits = predictions["type_pred"][mask]  # [num_masked, num_types]
    type_loss = F.cross_entropy(type_logits, targets["original_types"])

    # Feature reconstruction loss (MSE)
    feat_pred = predictions["feature_pred"][mask]  # [num_masked, channels]
    feat_target = targets["original_feature_targets"]
    feat_loss = F.mse_loss(feat_pred, feat_target)

    # Combined loss (equal weighting; can tune)
    return type_loss + feat_loss
```

### 2. `model.py` — `MaskedTokenHead`

See Task 2 for the head architecture. The head needs to output:
- Type predictions: `[B, K, num_node_types]` (one logit per type per position)
- Feature predictions: `[B, K, channels]` (reconstructed feature vector)

```python
class MaskedTokenHead(nn.Module):
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
            x_set: [B, K, channels] from backbone (full sequence, not just seed)
        Returns:
            dict with "type_pred" and "feature_pred"
        """
        return {
            "type_pred": self.type_predictor(x_set),
            "feature_pred": self.feature_predictor(x_set),
        }
```

**IMPORTANT**: The backbone's `encode()` must return the FULL sequence `[B, K, channels]` for masking (not just the seed node at position 0). This is different from the supervised setting where only position 0 matters. Task 2 must account for this — `encode()` should return the full `x_set` before the seed-node selection step.

### 3. Feature target generation (two-pass)

To get reconstruction targets for features, we need the feature encoder's output for the ORIGINAL (unmasked) inputs. Two approaches:

**Option A: Two-pass (clean but slower)**
1. Forward pass 1: encode original (unmasked) batch through `tfs_encoder` only → save as targets
2. Forward pass 2: encode masked batch through full backbone → get predictions
3. Compute loss between predictions at masked positions and targets from pass 1

**Option B: Stop-gradient (efficient)**
1. During collation, encode original TFs → detach as targets
2. Mask the batch
3. Forward through backbone
4. Loss between backbone output at masked positions and detached targets

**Recommendation:** Option A is cleaner. The `tfs_encoder` forward is much cheaper than the full backbone, so the overhead is small.

## Design Decisions

### Mask ratio
- Start with 15% (BERT-style)
- Can experiment with higher ratios (30-50%) as in MAE
- Higher ratios = harder task = potentially better representations, but may destabilize training

### Loss weighting between type and feature reconstruction
- Start with equal weighting (1:1)
- Type loss is cross-entropy (typically ~2-5 range)
- Feature loss is MSE (scale depends on feature magnitudes)
- May need to normalize or tune the ratio

### What about hop and time reconstruction?
- Optional — these carry less semantic information than type and features
- Hop reconstruction is trivial (only 4 values: 0, 1, 2, 3)
- Time reconstruction is harder but may not help representation learning
- Start without them, add if needed

## Testing

- Verify `apply_masking` never masks position 0
- Verify mask ratio is approximately correct (within sampling variance)
- Verify `compute_reconstruction_loss` produces a scalar loss that decreases during training
- Verify feature targets are detached (no gradient flow through targets)
- Verify masked positions in the batch actually have mask tokens (not original values)

## Dependencies

- Task 2 (backbone separation) — `encode()` must return full `[B, K, channels]` sequence
- Task 1 (namespacing) — `num_node_types` in the union map determines type prediction head size
- Task 4 (training script) — calls `apply_masking` and `compute_reconstruction_loss`
