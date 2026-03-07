# Foundation Model Design Decisions

## Goal
Transform RelGT from single-dataset supervised training to a foundation model that:
- Trains on ALL RelBench datasets simultaneously (leave-one-out)
- Uses self-supervised masked token prediction (inspired by RT paper, arXiv 2510.06377)
- Evaluates via zero-shot prediction on held-out dataset (no fine-tuning)

## Key Architecture Decisions

### 1. Table Name Namespacing
- Format: `"{dataset_name}__{table_name}"` to prevent collisions across datasets
- Applied in both node type encoder and data loading

### 2. Self-Supervised Objective: Masked Token Prediction
- Chosen over mixed regression/classification supervised heads
- Avoids the loss balancing problem between regression and classification tasks
- Random masking of K-1 neighbor positions
- Reconstruction loss: type cross-entropy + feature MSE
- Feature targets generated via two-pass with detach

### 3. Backbone/Head Separation
- `encode()` method returns representations without task-specific head
- `MaskedTokenHead` class for pretraining
- Original supervised head remains optional for fine-tuning

### 4. Multi-Dataset Data Loading
- `build_union_mappings()`: load all datasets, namespace, merge node type maps
- Temperature-scaled batch sampling: `p_i ~ size_i^alpha` with `alpha=0.5` (square-root scaling)
- InterleavedIterator cycles through datasets

### 5. VQ Codebook (Global Attention)
- Shared centroids across ALL datasets (desirable - learns universal structural patterns)
- On unseen datasets: centroids still work because they capture structural motifs, not dataset-specific features
- EMA-updated buffers (not parameters) - critical DDP consideration

### 6. GloVe Precomputation (IMPLEMENTED - commit 58cdc8d)
- Precompute all table name embeddings at init as buffers
- Eliminates CPU SentenceTransformer bottleneck in forward pass (major DDP perf win)
- SentenceTransformer is local variable in __init__, never stored as submodule, garbage collected
- Buffer: [num_types+1, 300] — ~120KB for all RelBench tables. Negligible.
- Forward: buffer index + Linear projection. No CPU-GPU sync.

### 7. SharedEmbeddingEncoder Dimension Discovery (IMPLEMENTED - commit 58cdc8d)
- torch_frame's `StatType.EMB_DIM` stores embedding dimension per column in col_stats_dict
- SharedEmbeddingEncoder scans col_stats_dict at init to discover all unique embedding dims
- Creates `nn.ModuleDict` with one `nn.Linear(dim, out_channels)` per unique dim
- No hardcoded 300d assumption. Handles GloVe (300), BERT (768), any future embedder.
- User must use same text embedder for inference as training (enforced by col_stats_dict)

### 8. Z-Score Normalization
- torch_frame's LinearEncoder already does Z-score internally using MEAN/STD from col_stats
- Our manual Z-score in NeighborTfsEncoder._normalize_numerical replicates this correctly
- For unseen tables at inference: user provides col_stats_dict (computed via make_pkey_fkey_graph)
- No batch-level fallback needed — stats always available from data pipeline

### 9. Zero-Shot Evaluation
- Target column treated as masked token
- Forward pass with masked target, reconstruct prediction
- Map reconstructed features to task label space

## Reference Papers
- RelGT: arXiv:2505.10960
- RT (masked token pretraining for relational data): arXiv:2510.06377
