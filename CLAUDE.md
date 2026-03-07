# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Relational Graph Transformer (RelGT)** — a graph transformer architecture for multi-table relational data represented as heterogeneous temporal graphs. Based on the paper [arXiv:2505.10960](https://arxiv.org/abs/2505.10960). It operates on the [RelBench](https://relbench.stanford.edu/) benchmark.

## Running Experiments

Training uses PyTorch DDP (Distributed Data Parallel) via `torchrun`:

```bash
# Single GPU example
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
    main_node_ddp.py \
    --dataset rel-f1 --task driver-top3 \
    --batch_size 512 --channels 512 --num_neighbors 300 \
    --num_layers 4 --epochs 10 --lr 0.0001 \
    --ff_dropout 0.3 --attn_dropout 0.3 \
    --gt_conv_type full --precompute \
    --out_dir results/my_run --run_name my_run

# Batch experiments via shell scripts
bash expts/run-large-base-experiments.sh
```

Key CLI arguments: `--dataset`, `--task`, `--gt_conv_type` (local/global/full), `--ablate` (none/type/hop/time/tfs/gnn), `--num_neighbors` (K, sequence length), `--num_centroids`, `--gnn_pe_dim`.

## Architecture

The model has a **multi-element tokenization** strategy that decomposes each node into 5 token components, then combines local attention over sampled subgraphs with global attention to learnable centroids.

### Core Modules

- **`model.py`** — `RelGT` (top-level model) and `RelGTLayer` (single transformer block).
  - `RelGT.forward()` encodes 5 token types (type, hop, time, features, GNN PE), concatenates them, mixes via MLP, then passes through RelGT layers and a prediction head.
  - `RelGTLayer` supports three `conv_type` modes: `local` (subgraph attention only), `global` (centroid attention only), `full` (both concatenated).

- **`encoders.py`** — Five encoder classes for the tokenization:
  - `NeighborNodeTypeEncoder` — GloVe-based table name embeddings (table-agnostic via text).
  - `NeighborHopEncoder` — Learned embeddings for hop distances.
  - `NeighborTimeEncoder` — Positional encoding for temporal information.
  - `NeighborTfsEncoder` — Table-agnostic feature encoder using shared encoders per stype (numerical, categorical, multicategorical, timestamp, embedding) + a shared transformer with CLS token readout.
  - `GNNPEEncoder` — GIN-based positional encoding on subgraph structure.

- **`local_module.py`** — `LocalModule`: multi-layer transformer encoder for local (subgraph) attention with attention-weighted neighbor aggregation. Uses Flash Attention via `F.scaled_dot_product_attention`.

- **`codebook.py`** — `VectorQuantizerEMA`: EMA-updated vector quantization codebook for global centroid attention (adapted from GOAT).

- **`utils.py`** — `GloveTextEmbedding` (sentence-transformers wrapper) and `RelGTTokens` (Dataset class that handles neighbor sampling, tokenization, precomputation, and batching via a custom `collate` method).

- **`main_node_ddp.py`** — Training script with DDP, wandb logging, and evaluation. Supports binary classification, regression, and multilabel classification tasks.

### Data Flow

1. `RelGTTokens` precomputes/samples K-hop neighbor subgraphs per seed node and stores tokens (types, hops, times, TensorFrames, subgraph edges).
2. Custom `collate` groups TensorFrames by node type for batched encoding.
3. The 5 encoders produce `[B, K, channels]` tensors each, which are concatenated and mixed.
4. `RelGTLayer` runs local attention (over the K-length sequence) and optionally global attention (against VQ centroids).
5. The seed node's representation (position 0) is used for the final prediction.

## Environment Setup

Requires Python 3.12, PyTorch with CUDA, PyG (torch_geometric, pyg_lib, torch_scatter, etc.), relbench, torch_frame, sentence-transformers, wandb, einops, pynvml, and h5py. See README.md for full Micromamba-based setup.

## Active Branch Context

The `table_agnostic_model` branch modifies the node/feature encoders to be table-agnostic — using shared encoders (hashing for categoricals, shared MLPs for numericals, GloVe for type names) instead of per-table encoders, enabling generalization to unseen tables.

Key changes on this branch:
- **GloVe precomputation**: `NeighborNodeTypeEncoder` precomputes all embeddings at init into a buffer (no CPU SentenceTransformer in forward)
- **Dimension-aware embedding projectors**: `SharedEmbeddingEncoder` discovers `EMB_DIM` from `col_stats_dict`, creates one `nn.Linear` per unique dim
- **Z-score normalization**: per-table mean/std as buffers in `NeighborTfsEncoder`, NaN imputed with column mean
- **Validation guards**: contiguous index check, ambiguous 2D reshape detection, safe name collision detection, `_num_zscore_tables` buffer for inference guard

Foundation model plan (7 tasks) in `agent_tasks/`. Architecture decisions in `summaries/architecture_changes_and_tradeoffs.md`.
