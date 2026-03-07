# Codebase Status

## Branch: table_agnostic_model

### Completed Changes
- Table-agnostic encoders (commit eb5ead8): SharedNumericalEncoder, SharedCategoricalEncoder (hash-based), SharedMultiCategoricalEncoder, SharedTimestampEncoder, SharedEmbeddingEncoder
- Z-score normalization for numerical features (commit 3bae093): per-table mean/std as buffers
- GloVe precompute + dimension-aware embedding encoder (commit 58cdc8d):
  - NeighborNodeTypeEncoder: precompute all GloVe embeddings into buffer at init, removed SentenceTransformer as submodule
  - SharedEmbeddingEncoder: discovers EMB_DIM from col_stats_dict via StatType.EMB_DIM, creates one Linear per unique dim
  - Plumbed emb_dims through TableAgnosticStypeEncoder and NeighborTfsEncoder
- M1 Mac compatibility changes in main_node_ddp.py, utils.py, encoders.py (uncommitted)
- Project permissions in .claude/settings.json

### Encoder Audit Result (2026-03-07)
**All 5 encoders are fully table-agnostic.** No per-table learnable parameters.
See `memories/encoder_audit.md` for full details.

### Model-Level Blockers (in model.py, NOT in encoders)
| Issue | Component | Severity | Fix |
|---|---|---|---|
| c_idx per-node buffer | RelGTLayer:65 | Blocks multi-dataset with global/full conv | Eliminate c_idx, use _ema_cluster_size (Task 7) |
| Fixed task head | RelGT.head:258 | Blocks multi-task training | Per-task ModuleDict or self-supervised pretraining |
| num_nodes sizing | RelGT.__init__:166 | Tied to c_idx | Removed when c_idx eliminated |

**Workaround**: `conv_type="local"` bypasses global attention entirely.

### Planned but NOT Implemented (7 Tasks in agent_tasks/)
All task details are in `agent_tasks/` folder with full implementation instructions.

| Phase | Task | Status |
|-------|------|--------|
| 1 | Task 1: Table name namespacing + GloVe precompute | **Partially done** (GloVe precompute done in 58cdc8d, namespacing pending — data pipeline concern) |
| 1 | Task 2: Backbone/head separation | Not started |
| 2 | Task 3: Multi-dataset data loading | Not started |
| 2 | Task 5: Masked token pretraining | Not started |
| 2 | Task 7: DDP global attention fixes | Not started |
| 3 | Task 4: Foundation training script | Not started |
| 4 | Task 6: Zero-shot evaluation | Not started |

Critical path: Task 1 -> Task 3 -> Task 4 -> Task 6

### Key Files
- `model.py` — RelGT and RelGTLayer
- `encoders.py` — 5 token encoders (type, hop, time, features, GNN PE)
- `utils.py` — RelGTTokens dataset, GloveTextEmbedding, neighbor sampling
- `codebook.py` — VectorQuantizerEMA (VQ for global attention)
- `local_module.py` — LocalModule transformer encoder
- `main_node_ddp.py` — Training script with DDP
- `agent_tasks/PLAN.md` — Overall execution plan
- `agent_tasks/testing_plan.md` — 40+ tests

### Testing Plan
Full testing plan in `agent_tasks/testing_plan.md` with tests T1.1-T1.8, T2.1-T2.7, T3.1-T3.7, T5.1-T5.7, T7.1-T7.8, T4.1-T4.10, T6.1-T6.6, R1-R3.
