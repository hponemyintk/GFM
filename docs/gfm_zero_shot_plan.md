# Plan — Evolve current RelGT pretrain into a true Graph Foundation Model (zero-shot to unseen schemas)

## North star

Pretrain a **single RelGT backbone** that, when handed a new RelBench-style relational dataset (different tables, different columns, different categorical levels) **at adoption time**, produces useful per-row embeddings without retraining the backbone. Two downstream consumption paths must work:

1. **Embedding extraction → TabPFN.** Run the backbone on the new dataset's seed nodes, get `[B, channels]` embeddings, hand them to TabPFN as features. No backbone gradient computation post-pretrain.
2. **Embedding + fine-tuned head.** Same embedding extraction, but train a fresh `Linear(channels, 1)` (or small MLP) on the new dataset's labels. Optionally unfreeze the backbone after some warmup.

Pretraining objective stays multi-task supervised (current setup) plus likely auxiliary self-supervised losses to get a generalization-ready backbone. Per-task heads at pretrain time are throwaways — they stop existing the moment we go zero-shot. Only the backbone matters.

## Why the current code can't do this (snapshot of the gap)

The pretrained backbone has *learned* dependencies on the specific tables / columns / categorical levels it was trained on. Pull any of these out and a forward pass crashes. Concretely:

| Site | What's tied to the training schema | Failure on unseen data |
|---|---|---|
| `encoders.NeighborTfsEncoder.encoders` | `nn.ModuleDict` keyed by prefixed type name (`rel-f1::drivers`); per-type torch_frame `ResNet`/`TableAgnosticStypeEncoder` instances built for specific columns + stypes | KeyError on type lookup; even if name matched, encoder shape doesn't match new columns |
| `NeighborNodeTypeEncoder` (in `model.RelGT`) | `Linear(|T|, channels)` indexed by `unified_type_map` integer ids | New types have no row → IndexError or silently wrong embedding |
| Categorical embedding tables inside per-type encoders | `nn.Embedding(num_categories, channels)` per categorical column, learned IDs | New driver/customer/etc. IDs aren't in the table |
| `model.py:65` — `c_idx` buffer | `torch.randint(0, num_centroids, (num_nodes_total,))` — **size pinned to total training-time node count** | Indexing it with a new dataset's node-id explodes; semantics also wrong |
| `task_tokens.py` per-task `target_mean/std` | Z-score stats fitted on training labels | New task targets denormalize against the wrong distribution |
| `multi_task_head.task_heads` | Per-task `Linear` heads | No head exists for the new task — by design |

The first three rows are the architectural blockers. The rest are configuration / state that adoption code needs to handle but don't require backbone changes.

---

## Architectural changes — backbone-only, scoped to enable zero-shot

We need to remove every "per-X" learned parameter where X is "table", "column", "categorical level", or "training-time-node-id". The backbone's parameter count must depend ONLY on architectural constants (channels, layers, heads, num_centroids), not on the schema.

### A1. Schema-agnostic column encoding (highest priority)

**Goal:** one shared encoder operates on any (column_name, column_value, stype) triple. Replace `NeighborTfsEncoder.encoders[type]` ModuleDict.

**What we have:** `encoders.TableAgnosticStypeEncoder` — already shared across tables. **What's still per-table:** `NeighborTfsEncoder` wraps it with a per-type ModuleDict because torch_frame's `ResNet`/transformer wrapper holds per-table column-stat affine layers.

**Concrete refactor:**

1. **Merge `TableAgnosticStypeEncoder` directly into `NeighborTfsEncoder.forward`.** Drop the per-type ModuleDict. The forward becomes:

   ```python
   def forward(self, batch_dict, neighbor_types):
       # Group rows by stype (numerical | categorical | timestamp | text)
       # NOT by table. Each stype-group runs through a single shared
       # encoder layer that takes (column_name_embedding, column_value).
       ...
       per_row_emb = self.shared_stype_encoder(grouped_by_stype)
       # Then the existing column-attention transformer (already shared)
       # mixes columns within each row.
       row_emb = self.shared_transformer(per_row_emb)
   ```

2. **Column-name encoder.** Reuse the existing GloVe column-name embedding pathway from `TableAgnosticStypeEncoder`. Each column contributes `(name_embedding, value_embedding)` pairs to the row sequence. No per-column learned parameters; everything keys on the text of the column name.

3. **Per-stype value encoder (no per-column state):**
   - `numerical`: `Linear(1, channels)` shared. Z-score the value first using running stats *of the value alone* — but we can't fit stats per-column at adoption time. Two options:
     - (a) Compute z-score per-batch (stateless). Cheapest, slight train/test distribution drift.
     - (b) The model's first BN absorbs the scaling — `BatchNorm1d` over the value feature works because it's column-agnostic.
   - `categorical`: TEXT-encode the level. e.g., `glove("driver_id_42")` or `text_encoder(level_string)`. Can't use `nn.Embedding` because that's ID-specific. **This is the single biggest behavioral change** — categorical columns lose their learned per-level richness, but gain zero-shot transfer.
   - `multicategorical`: same as categorical, then mean-pool.
   - `timestamp`: `Linear(1, channels)` over the (timestamp − seed_time) delta. Already stateless.
   - `embedding` (text/dense vector input): pass through a `Linear(emb_dim, channels)` shared across all text columns.

4. **Drop torch_frame's per-table affine layers.** The torch_frame `Dataset.__init__` builds per-column stats that flow into the encoder's `LinearEncoder`. We bypass this by feeding raw values + name embeddings.

**Refactor cost:** ~2-3 days. Touches `encoders.py` (rewrite `NeighborTfsEncoder`), `model.py` (the encoder feeds into the GT layers — interface should stay `[B, K, channels]`), `gfm_data/collate.py` (no longer groups by node type, groups by stype). Tests in `test_emb_dims_discovery`, `test_embedding_encoder`, `test_type_encoder` need rewrite.

**Risk:** the per-table parameter capacity was doing real work on imbalanced tables. Removing it likely hurts in-distribution accuracy. Acceptable cost for zero-shot transfer.

### A2. Type-agnostic node-type encoding

**Goal:** replace `Linear(|T|, channels)` with a function of the **text** of the type name.

**Current:** `NeighborNodeTypeEncoder` in `encoders.py` does `W_type @ onehot(type_id)`.

**Replacement:**
```python
self.type_text_proj = nn.Linear(glove_dim, channels)
def forward(self, type_strs: List[str]):
    # type_strs: e.g. ["rel-f1::drivers", "rel-f1::races", ...] per row
    name_text = [t.split("::")[-1] for t in type_strs]
    glove = self.glove_encode(name_text)  # [B, glove_dim]
    return self.type_text_proj(glove)
```

Same trick used for column names already — extends to types. No `unified_type_map` dependency at inference time. Inputs become strings, not ids. `cache.node_types` and `unified_type_map` still useful for indexing during sampling, but they no longer need to match the training-time vocabulary at the encoder level.

**Refactor cost:** ~half-day. Touches `encoders.NeighborNodeTypeEncoder`, `gfm_data/collate.py` (need to pass type strings in batch dict), `gfm_data/task_tokens.py` (already has `cache.index_to_node_type` available).

### A3. Stateless or dynamic centroid assignment

**Goal:** drop the per-training-node `c_idx` buffer.

**Current (`model.py:65`):**
```python
c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
self.register_buffer("c_idx", c)
```

`num_nodes` is the sum of all training-dataset node counts. This makes the model size scale with the training dataset and breaks completely on new node ids.

**Replacement options (pick one):**

1. **Compute centroid assignment on the fly per batch.** In `RelGTLayer.forward`:
   ```python
   q_x = self.lin_query_g(x)   # current
   distances = pairwise_dist(q_x, self.vq._embedding)  # [B, num_centroids]
   c_idx_batch = distances.argmin(dim=-1)              # [B]
   centroid_count = torch.bincount(c_idx_batch, minlength=self.num_centroids)
   ```
   No buffer needed. The VQ codebook itself (centroids) stays as the only learned global state — that's what we want.

2. **Hash-based assignment.** `c_idx = hash((type, id)) % num_centroids`. Fastest, no compute cost, but loses any "similar nodes get similar centroids" property.

Option 1 is the principled choice. Compute cost is `[B, num_centroids]` distance matrix — at `num_centroids=4096` and `B=512` that's 2M ops per layer, negligible.

**Refactor cost:** ~half-day. Touches `model.RelGTLayer.forward`, removes the `register_buffer("c_idx", ...)` and the `__init__` arg threading.

### A4. Categorical levels — text encoding (decision point)

**Goal:** make categorical embedding open-vocab.

**Current:** torch_frame's `EmbeddingEncoder` builds an `nn.Embedding(num_levels, channels)` per column.

**Two choices:**

- **(a) Text-of-level encoding** (true open-vocab): embed the *string* of each categorical value. Works only when categorical levels have meaningful text (country names, product categories, status enums). For opaque IDs (`driver_id=823`, `user_id=42`), text embeddings of the integer string are essentially noise.
- **(b) Fixed random projection of integer level** (Hashed embedding): `nn.Embedding(LARGE_PRIME, channels)` indexed by `hash(level) % LARGE_PRIME`. Open-vocab via collision. Loses identity info but consistent across datasets.

**Recommendation:** (a) for stype `categorical_text` (already used for product names etc.), (b) for `categorical_id`. Need a sub-stype distinction in `gfm_data/stypes.py` to route them.

**Refactor cost:** ~1 day. Touches `encoders.SharedCategoricalEncoder`, `gfm_data/stypes.py` (add the categorical_id vs categorical_text distinction), `gfm_data/tf_store.py` (changes how categorical levels are stored — text vs int).

### A5. Drop `MultiTaskHead` from the saved backbone

The pretrained model checkpoint should contain backbone-only. Currently `MultiTaskRelGT` holds backbone + heads. Saving with `MultiTaskRelGT.state_dict()` includes per-task heads that don't transfer.

**Fix:** add `MultiTaskRelGT.save_backbone(path)` and `MultiTaskRelGT.load_backbone(path)` that pickle/unpickle only `self.backbone.state_dict()`. Adoption code constructs a fresh `RelGT(...)` with the same architectural constants and loads.

**Refactor cost:** ~30 min. Pure plumbing.

---

## Pretraining objective changes

The current setup is **supervised multi-task per-row prediction**. For zero-shot transfer, the backbone needs to learn representations that generalize, not memorize per-task structure. Add self-supervised auxiliary losses.

### B1. Masked column prediction

For a fraction of nodes per batch, **mask one or more column values at the input** (replace with a learned `[MASK]` token) and ask the backbone to predict the original from the surrounding token sequence. Loss is per-stype:

- numerical: MSE on z-scored value
- categorical: cross-entropy over text embedding (or contrastive against a batch of negatives)
- text: cosine sim to original GloVe embedding

This is BERT-style for tabular data. Already explored in literature (TaCo, TabBERT). Forces the column-attention transformer to learn meaningful inter-column structure that transfers.

**Implementation cost:** ~2 days. Add a `MaskedColumnPrediction` head to `MultiTaskRelGT`, augment collate to randomly mask, write an auxiliary loss term. Would also lift `--loss_balance` mode to weight masked-prediction vs supervised tasks.

### B2. Masked node prediction (subgraph dropout + reconstruction)

For a fraction of K-1 neighbors, drop them from the local subgraph and ask the model to predict their *type and ID* given the surroundings. Encourages the backbone to capture structural regularities (parent-child, common patterns).

**Implementation cost:** ~2 days. Augment shard build or do dropout at runtime. New auxiliary head + loss.

### B3. Contrastive seed-node pretext

For each seed node, generate two random K-neighbor samplings of its subgraph. Run both through the backbone. Maximize cosine similarity between the two embeddings; minimize against batch negatives. NCE loss. Forces invariance to neighborhood-sampling noise → embeddings depend on structural signal, not which 300 neighbors happened to get sampled.

**Implementation cost:** ~1.5 days. Modify the sampler to yield two views per seed; add a contrastive head + InfoNCE loss.

### B4. Loss schedule

Pretrain runs over a mix of:
- 50% supervised multi-task (current)
- 25% masked column prediction
- 15% masked node prediction
- 10% contrastive seed

Tunable. Total loss is `w_sup * L_sup + w_mask_col * L_mask_col + ...`. Use `--loss_balance uncertainty` to learn task weights — already in place.

---

## Adoption protocol

### C1. Embedding extraction API

Add `tools/extract_embeddings.py`:

```python
# Inputs: pretrained backbone .pt, new dataset name, seed table, K
# Process: build TF store + shards for the new dataset (phases 1+2),
#          load backbone, run forward with task_id=None, save [B, C] tensor
# Output: results/<run>/<new_ds>/<task>_embeddings.{pt,parquet}
```

Reuses `MultiTaskRelGT.forward(task_id=None)` (already merged in `bdff105`). The dataset-load + cache-build path needs a "fresh dataset that wasn't in pretrain" mode. Adoption-time pre-flight is the existing phase 1 + phase 2 + phase 3-startup minus the training loop.

**Implementation cost:** ~1 day. Mostly tooling.

### C2. TabPFN downstream pipeline

Once embeddings are extracted:

```python
import tabpfn
X_train, y_train = embeddings[train_idx], labels[train_idx]
X_test = embeddings[test_idx]
clf = tabpfn.TabPFNClassifier()  # or TabPFNRegressor
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
```

That's it — TabPFN consumes the embeddings as if they were features. Document this in `docs/adoption_tabpfn.md` with a concrete example using rel-f1 (treat it as "unseen" by extracting from a backbone trained on rel-event/rel-amazon/etc.).

**Implementation cost:** ~half-day. Just tooling + a notebook/script.

### C3. Fine-tuned head pipeline

```python
backbone = MultiTaskRelGT.load_backbone(pretrained_path)
backbone.eval().requires_grad_(False)   # frozen by default
head = nn.Linear(channels, 1)
optim = torch.optim.AdamW(head.parameters(), lr=1e-3)

for epoch in epochs:
    for batch in new_dataset_loader:
        with torch.no_grad():
            emb = backbone(batch, task_id=None)  # [B, C]
        pred = head(emb).squeeze(-1)
        loss = loss_fn(pred, batch.labels)
        loss.backward()
        optim.step()
```

Plus an optional `--unfreeze_after_epoch` knob to start backbone gradient flow after the head has warmed up (LIaR-style, prevents catastrophic forgetting).

**Implementation cost:** ~1 day. New script `tools/finetune_new_task.py`. New regression test that loads an existing backbone, fine-tunes a head on rel-f1.driver-position with the backbone frozen, verifies non-trivial val MAE.

### C4. Held-out dataset evaluation harness

To validate zero-shot capability, **hold out one RelBench dataset entirely** during pretrain. Pretrain on the other 6, evaluate on the held-out one via:

- (i) Embedding → TabPFN
- (ii) Embedding → fine-tuned head (frozen backbone)
- (iii) Embedding → fine-tuned head (unfrozen after warmup)
- (iv) Single-task RelGT trained from scratch (paper baseline)

Compare. If (i)/(ii) come within 10% of (iv), zero-shot transfer is working. If they don't, either the backbone needs more capacity / data / pretrain steps, or the architectural changes regressed in-domain quality so far that there's no room left for transfer.

**Implementation cost:** ~1 day. New script `scripts/holdout_eval.sh` running the four conditions.

---

## Migration phases (concrete commits)

Each phase is a self-contained PR with passing tests. Total: ~3 weeks of focused work.

### Phase 1 — schema-agnostic encoder refactor (1 week)

- A1 (column encoder unification)
- A2 (type-name text encoding)
- A4 (categorical text/hash encoding)
- A5 (backbone-only checkpoint)

End state: `MultiTaskRelGT` constructs with `num_tasks` and `task_type_ids` only — NO `col_names_dict` or `col_stats_dict` plumbing. Backbone state dict is portable across datasets.

Tests added:
- `test_encoder_zero_shot_shapes`: build encoder on dataset A's schema, run forward on dataset B's schema, assert no key/index errors and output shape `[B, K, channels]`.
- `test_backbone_save_load_portable`: save backbone trained on dataset A, load into a fresh model construction for dataset B.

Existing tests rewritten:
- `test_emb_dims_discovery`, `test_embedding_encoder`, `test_type_encoder`, `test_column_semantic_embedding`, `test_multi_task_dataset` — all need updates for the new collate/encoder API.

### Phase 2 — `c_idx` removal (2 days)

- A3 (dynamic centroid assignment)

End state: `RelGT.__init__` doesn't take `num_nodes`. Forward computes centroid assignment per batch.

Tests added:
- `test_dynamic_c_idx_matches_static_on_pretrain_data`: on the SAME dataset, assert per-batch dynamic c_idx is consistent with what the buffer-based `c_idx` would have produced (within tolerance — they won't be bit-for-bit because the static one was random init).
- `test_dynamic_c_idx_works_on_new_node_ids`: assert no IndexError when called with node-id ranges outside training scope.

### Phase 3 — self-supervised auxiliary losses (1 week)

- B1 (masked column)
- B2 (masked node)
- B3 (contrastive seed)
- B4 (loss schedule)

End state: pretraining mixes supervised + 3 self-supervised objectives. `--ssl_weight` etc. as launcher knobs.

Tests added:
- `test_masked_column_loss_decreases`: small synthetic dataset, train 100 steps, loss decreases.
- `test_contrastive_loss_pulls_views_together`: same.
- `test_no_label_leak_in_masked_prediction`: assert masked rows don't see their own label.

### Phase 4 — adoption tooling (3-4 days)

- C1 (extract_embeddings.py)
- C2 (TabPFN integration script + doc)
- C3 (finetune_new_task.py)

Tests added:
- `test_extract_embeddings_outputs_shape`: end-to-end smoke; load backbone, extract on rel-f1, assert `[N, channels]` tensor.
- `test_finetune_head_converges`: 5-epoch fine-tune on rel-f1.driver-top3 from a pretrained-on-rel-event backbone, assert val AUROC > 0.6.

### Phase 5 — held-out validation + tuning (1+ week, depends on results)

- C4 (held-out eval harness)
- Run pretrain over 6 datasets, eval on 7th
- Tune SSL weights, K, channels until zero-shot numbers are reasonable

This phase is research, not engineering. Could expand if first results are poor.

---

## What to keep AS-IS

These current pieces translate cleanly and don't need changes:

- **VQ-EMA codebook** (`codebook.py`) — global centroids are dataset-independent; the all-reduce DDP sync we just merged is correct.
- **Per-task heads** (`heads/multi_task_head.py`) — used at pretrain time, dropped at adoption time. The `task_id=None` extraction path is the adoption hook.
- **Sampler / shard pipeline** (`gfm_data/sampler.py`, `tools/precompute_shards.py`) — works on any dataset with the right `name_prefix` and `unified_type_map`, both of which are determined at runtime. The K-aware sentinel fix we just merged makes shard caching robust to per-run K changes.
- **Z-score regression normalization** — per-task on training. Adoption code re-fits on the new task's labels; same machinery, different stats. No backbone change.
- **Multi-task DDP infrastructure** (chunked load, watchdog, checkpoint, all-reduce) — schema-agnostic, keeps working.

---

## Open research questions

1. **How much capacity does the schema-agnostic encoder lose vs per-table?** Likely 5-15% on in-domain metrics. May need to compensate with a wider backbone (channels=768 or 1024).

2. **Is text-of-categorical sufficient for ID-heavy schemas?** rel-amazon's `user_id` and `item_id` are opaque integers. Text embedding `"42"` vs `"43"` is meaningless. Hashed embedding (A4 option b) loses identity info but is at least consistent. **TBD whether this hurts a lot or a little.**

3. **Does masked-column pretraining transfer to label prediction?** Strong evidence in BERT-style for text and tables (TabBERT, TURL). Less evidence for relational/temporal data. Worth piloting on rel-event before full pretrain.

4. **How much pretrain data is enough?** RelBench v2's 7 datasets total ~50M training rows. May need more: synthetic relational graphs from logs/datasets/etc. Could be a separate effort.

5. **What's the right downstream protocol — TabPFN or fine-tune?**
   - TabPFN: zero training cost, but limited to small `n_train` (~1k-10k rows). Fits adoption budget for small RelBench tasks but not rel-amazon-scale.
   - Fine-tuned head: scales to any train size, ~minutes of compute on a single GPU. More flexible but requires labeled adoption data.
   - Hybrid: extract embeddings, fit small MLP, use TabPFN as a regularizer. Possible but speculative.

6. **What's the cross-row temporal handling at adoption time?** The current pipeline uses temporal cutoffs to prevent leakage. Adoption-time labeled data has its own temporal structure. The seed-time filter on the sampler still works since it's input-agnostic — but the `target_mean/std` should be fitted on adoption-time train rows, NOT on the new dataset's full labels (would leak val/test).

---

## Risk register (most important first)

1. **In-domain metric regression from removing per-table parameters.** Tracked in Phase 1's parity tests. If RelBench in-domain numbers degrade > 15%, it's not worth the trade.

2. **Masked-prediction objectives don't help.** If SSL losses drive train_loss but don't transfer to held-out, we get worse-than-random zero-shot. Pilot on a small held-out before committing the full pretrain.

3. **VQ centroids don't generalize across datasets.** Current centroids are EMA over training nodes. New dataset's nodes get assigned to centroids fitted on training distribution — may produce poor global attention. Could mitigate by clustering on raw GloVe embeddings of (type+column-text) instead of learned EMA.

4. **Compute budget.** Full pretrain on 6 RelBench v2 datasets at paper-config (channels=512, layers=4, K=300, batch=256) on 8×A100 takes 1-3 days. Adding SSL losses + multi-view contrastive doubles it. Need realistic budget upfront.

5. **TabPFN doesn't accept high-dim features well.** TabPFN was trained on `~100` numerical features. Our embeddings are 512-d. May need a learned projector or PCA before feeding to TabPFN. Test early in Phase 4.

---

## Concrete first-week milestones

Day 1-2: Phase 1 — A2 (type-name text encoding) + A5 (backbone save/load). Smallest, highest-confidence changes. Tests pass at this point should already include a "load on different name_prefix" smoke.

Day 3-4: Phase 1 — A1 (column encoder unification). The big refactor. End of day 4: can run multi-task pretrain on rel-f1+rel-event, get test metrics, compare to current per-task ModuleDict baseline. Expect 5-10% degradation on F1/AUROC; document.

Day 5: Phase 2 — A3 (`c_idx` removal). Quick win. Tests + smoke.

Day 6-7: Phase 4 (skipping ahead) — C1/C2 (extraction + TabPFN). Even before SSL losses, validate the adoption path with the supervised-only backbone. Run on a held-out dataset (e.g., pretrain on rel-event, extract on rel-f1, TabPFN). If this beats RDL baseline on rel-f1.driver-top3 even slightly, we know the path works.

By end of week 1: a working adoption pipeline at modest quality. Phases 3 (SSL) and 5 (held-out tuning) are the path to good quality.

---

## Reference implementation pointers

- TableAgnosticStypeEncoder: `encoders.py:407` (already partially implementing A1)
- Column-name embedding: `encoders.py:GloveColumnNameEmbedding`
- Type-id one-hot: `encoders.NeighborNodeTypeEncoder` (replace per A2)
- c_idx buffer: `model.py:65` (replace per A3)
- Embedding extraction path: `heads/multi_task_head.MultiTaskRelGT.forward` with `task_id=None` (already in `bdff105`)
- Pretrained backbone state dict: currently the full `MultiTaskRelGT` includes heads; need backbone-only save (per A5)

Adoption-time data flow once Phase 1+2 land:

```
new dataset → TF store (phase 1, ~1 hr per dataset)
            → shards (phase 2, ~10 min per task)
            → DataLoader → backbone(task_id=None) → [B, channels] → save
                        ↓
       TabPFN.fit(emb, y) OR head.fit(emb, y)
```

No backbone parameters change at adoption time. Schema lives entirely in the encoder's input pipeline.
