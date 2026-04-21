# Distilled Sampler — Final Plan

Upgrade RelGT's subgraph sampler from uniform random to a **Pre-Softmax Logit Distillation Sampler** that learns to predict the teacher's last-layer seed-to-candidate attention logits, then at inference uses temperature-scaled Gumbel-Top-K (without replacement) over a larger scope pool.

Executed in three isolated phases, selected by `--run_mode`:

| Phase | Mode | Trains | Freezes | Data |
|-------|------|--------|---------|------|
| 1 | `teacher` | RelGT (all) | — | Random K=300 subgraphs (existing HDF5) |
| 2 | `distill` | DistillSampler | RelGT (all) | Random K=300 subgraphs (existing HDF5) |
| 3a | `joint` (setup) | — | Teacher + sampler | Scope-3000 pools + phase-1 adjacency |
| 3b | `joint` (train) | convs + ffs + in_mixture + layer_norm_pe + head | 4 base encoders + 4 base LNs + pe_encoder weights | Curated K=300 subgraphs (written by 3a) |

The separation prevents the sampler from co-adapting to an in-training transformer and partially mitigates covariate shift by freezing the embedding-input pathway from phase 1 onward.

---

## 1. Teacher logit extraction

**Target:** last `EncoderLayer`'s row-0 pre-softmax attention logits, mean over heads.

In `local_module.py` — `EncoderLayer.forward(x, attn_bias=None, extract_seed_logits=False)`:

```python
Q_seed = Q[:, :, 0:1, :]                              # [B, H, 1, d_h]
dots   = (Q_seed @ K.transpose(-2, -1)) / math.sqrt(head_dim)
seed_logits = dots.squeeze(2).mean(dim=1)             # [B, L]   (mean over heads)
```

Always returns `(x, seed_logits)` with `seed_logits=None` when the flag is off. The manual `Q_seed @ K^T` is computed in addition to `F.scaled_dot_product_attention`, not replacing it — the forward path is unchanged.

In `LocalModule.forward(batched_data, pretrain_token=False, extract_seed_logits=False)`:
- Pass `extract_seed_logits=True` only to the **last** enc_layer (earlier layers operate on less-contextualized features; last-layer logits best reflect the teacher's final selection behavior).
- Return `(output, seed_logits)`.

In `model.py` — `RelGTLayer.forward(..., extract_seed_logits=False)` and `RelGT.forward(..., extract_seed_logits=False, return_base_concat=False)`:
- Backward-compatible return: without flags, returns `x_set` as before. With either flag, returns `(x_set, extras_dict)` where `extras_dict` may contain `'seed_logits'` and/or `'base_concat'`.
- `base_concat` is `cat([layer_norm_type(type_emb), layer_norm_hop(hop_emb), layer_norm_time(time_emb), layer_norm_tfs(tfs_emb)], dim=-1)` — dim `4*channels`, **PE excluded**.

---

## 2. DistillSampler module (`distill_sampler.py`)

```python
class DistillSampler(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.Ws = nn.Linear(embed_dim, hidden_dim, bias=False)

    def forward(self, base_concat: Tensor) -> Tensor:
        # base_concat: [B, K, embed_dim]
        proj = self.Ws(base_concat)                              # [B, K, hidden_dim]
        seed = proj[:, 0:1, :]                                   # [B, 1, hidden_dim]
        q_imp = torch.bmm(seed, proj.transpose(1, 2)).squeeze(1) # [B, K]
        return q_imp

    @staticmethod
    def distillation_loss(q_imp: Tensor, teacher_logits: Tensor) -> Tensor:
        # Exclude seed self-entry at index 0
        return F.mse_loss(q_imp[:, 1:], teacher_logits[:, 1:])
```

`embed_dim = 4 * args.channels`. `hidden_dim = args.channels`.

---

## 3. Data pipeline (`utils.py`)

**No change** to the existing `RelGTTokens` — phase 1 and phase 2 both consume the current K=300 HDF5 with adjacency.

**New `RelGTScopeTokens`** — identical to `RelGTTokens` but:
- `sample_scope` replaces `K` in the token arrays.
- HDF5 stores only `types`, `indices`, `hops`, `times` at shape `[N, sample_scope]`. **No `edges`/`edges_offsets`, no `node_indices`, no `tfs` expansion at collate.**
- `_process_one_seed_scope`: reuses the existing gather/fallback logic with `K := sample_scope`, but skips the edge-index build.
- Separate on-disk path: `{cache_dir}/precomputed/{dataset}/{task}/scope_{sample_scope}/{split}.h5`.

**New `curate_subgraphs(...)` helper** — takes a trained teacher + sampler, iterates the scope dataset, applies Gumbel-Top-K (or deterministic top-K for val/test and recommended for train too), rebuilds `edge_index` on the selected K nodes using the cached `GLOBAL_ADJ`, and writes a phase-3 HDF5 with the same format as the phase-1 HDF5.

Phase-3 HDF5 path: `{cache_dir}/precomputed/{dataset}/{task}/curated_{K}/{split}.h5`.

---

## 4. Orchestration (`main_node_ddp.py`)

### New CLI args

```
--run_mode              {teacher,distill,joint}  default: teacher
--sample_scope          int                      default: 3000
--sample_temp           float                    default: 1.0
--teacher_ckpt          str                      default: auto-derived from out_dir
--sampler_ckpt          str                      default: auto-derived from out_dir
```

`--use_gnnpe_in_sampler` is **not** introduced. PE is excluded from the sampler by design.

### Mode dispatch

```python
if args.run_mode == "teacher":
    train_teacher()      # existing loop, unchanged
elif args.run_mode == "distill":
    train_sampler()      # load teacher ckpt, freeze, train sampler
elif args.run_mode == "joint":
    ensure_scope_precomputed()
    ensure_curated_split()   # rank-0 writes curated HDF5 if missing; all ranks barrier
    train_phase3()           # standard finetune on curated HDF5 with partial freeze
```

### Freeze mechanics in phase 3

```python
frozen_modules = [
    model.type_encoder, model.hop_encoder, model.time_encoder, model.tfs_encoder,
    model.layer_norm_type, model.layer_norm_hop, model.layer_norm_time, model.layer_norm_tfs,
    model.pe_encoder,
]
for m in frozen_modules:
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

# Re-apply m.eval() after every model.train() to keep BN running stats frozen:
def set_frozen_eval(model):
    for m in frozen_modules: m.eval()

# In loop:
model.train()
set_frozen_eval(model.module)
```

`GNNPEEncoder`'s `torch.randn(total_nodes, 1)` input draws a fresh tensor per forward — freezing its `nn.Parameter`s does not affect this; the random-per-forward behavior is preserved, matching phase-1 behavior.

Optimizer filters frozen params:
```python
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=base_lr, weight_decay=args.weight_decay,
)
```

DDP's `find_unused_parameters=True` is already set — frozen params won't block sync.

### Distill mode training step

```python
teacher.eval()                     # BN stats frozen
# teacher has no DDP wrap; plain replicated module per rank
with torch.no_grad():
    x_set, extras = teacher(..., extract_seed_logits=True, return_base_concat=True)
teacher_logits = extras["seed_logits"]   # [B, K]
base_concat    = extras["base_concat"]   # [B, K, 4*channels]

q_imp = sampler(base_concat)             # [B, K]
loss  = DistillSampler.distillation_loss(q_imp, teacher_logits)
loss.backward(); opt.step()
```

### Phase-3 curate step (rank 0 only, with dist.barrier())

```python
# Loads phase-1 teacher + phase-2 sampler, iterates scope loader, writes curated HDF5.
# Deterministic top-K for val/test.
# For train: deterministic top-K by default (reproducible dataset). 
# Stochastic Gumbel-Top-K at T=args.sample_temp available via --curate_stochastic flag (ablation).
```

Gumbel-Top-K (no replacement):
```python
if stochastic:
    u = torch.rand_like(q_imp)
    gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    noisy = q_imp[:, 1:] / args.sample_temp + gumbel[:, 1:]
    _, sel = torch.topk(noisy, k=K-1, dim=1)
else:
    _, sel = torch.topk(q_imp[:, 1:], k=K-1, dim=1)
sel = sel + 1   # shift past seed at index 0
```

Selected candidates are gathered from the scope's token arrays, and edge_index is rebuilt from `GLOBAL_ADJ` on those K nodes (seed at position 0, selected at 1..K-1).

---

## 5. Unit tests (`test_distillation.py`)

Behavioral (not just structural):

1. **Manual Q·K^T matches reference**: on a tiny handcrafted input with `num_heads=2, K=4, d_h=8`, assert the extracted `seed_logits` equals a numpy-computed `(Q[:,:,0,:] @ K^T / √d_h).mean(dim=1)` to within 1e-5.
2. **Distillation loss decreases**: synthetic teacher logits + random base_concat → train sampler 100 steps → assert final loss < 0.5 × initial loss.
3. **Freeze flags** in phase 3: assert the four base encoders + four LNs + pe_encoder have `requires_grad=False`, and convs/head have `requires_grad=True`.
4. **BN-eval persistence**: after `model.train(); set_frozen_eval(model)`, assert `model.tfs_encoder.training == False` and `model.convs[0].training == True`.
5. **Gumbel-Top-K shape**: `selected_idx.shape == (B, K-1)`.
6. **Deterministic top-K reproducibility**: with `stochastic=False`, same `q_imp` → identical `sel` across two calls.

---

## 6. Run order for rel-event user-repeat

Convenience driver (single GPU, all three phases sequentially):

```bash
bash expts/run-distilled-sampler-user-repeat.sh [GPU_ID]
```

The script uses the same hyperparameters as `expts/run-large-base-experiments.sh`
(`num_layers=4`, `channels=512`, `num_neighbors=300`, `ff_dropout=attn_dropout=0.3`,
`lr=1e-4`, `warmup_steps=10`, `max_steps_per_epoch=500`, `epochs=10`, `num_workers=8`,
`seed=0`, `--precompute`, `gt_conv_type=full`, `ablate=none`) but with
`batch_size=32`. All three phases share `out_dir`, so phase 2 auto-loads `phase1.pt`
and phase 3 auto-loads `phase1.pt` + `sampler.pt`.

Equivalent manual invocation:

```bash
# Phase 1 — train teacher
torchrun --nproc_per_node=N main_node_ddp.py \
    --dataset rel-event --task user-repeat \
    --run_mode teacher --epochs 10 \
    --out_dir results/distilled_sampler

# Phase 2 — train sampler (loads phase1.pt automatically)
torchrun --nproc_per_node=N main_node_ddp.py \
    --dataset rel-event --task user-repeat \
    --run_mode distill --epochs 5 \
    --out_dir results/distilled_sampler

# Phase 3 — curate (rank-0 one-shot) + retrain transformer/head
torchrun --nproc_per_node=N main_node_ddp.py \
    --dataset rel-event --task user-repeat \
    --run_mode joint --epochs 10 \
    --sample_scope 3000 --sample_temp 1.0 \
    --out_dir results/distilled_sampler
```

---

## 7. Open decisions deferred to after first results

- **Head-reduction** of teacher logits: starting with **mean**; switch to max if top-K recall is weak after phase 2.
- **Train-split curate stochasticity**: starting **deterministic**; `--curate_stochastic` enables Gumbel-Top-K at T=`sample_temp` as an ablation.
- **Temperature tuning**: default `T=1.0`. Before phase 3, measure `teacher_logits.std(dim=1).mean()` on a val batch; if σ ≫ 1 the default may be too exploit-heavy, adjust with a sweep `{0.5, 1.0, 3.0}`.
- **Scope pool size**: default 3000. Memory-bound for precompute; can be reduced to 1500 if disk pressure.

---

## 8. Non-goals / explicitly out of scope

- Joint sampler-and-model end-to-end backprop through Gumbel-Top-K (not differentiable in the hard path; relaxations like Gumbel-Softmax top-K exist but out of scope for v1).
- Listwise ranking loss alternatives to MSE (flagged as future work if MSE's top-K recall is poor).
- Cross-dataset sampler transfer.
- GNNPE in sampler inputs.
