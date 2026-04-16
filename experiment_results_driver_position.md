# ogPASS vs dev-kyaw — driver-position results (MAE)

Head-to-head on `rel-f1 / driver-position` (regression). Tune metric is **MAE**
(lower = better); we also report RMSE (lower) and R² (higher). 10 seeds (0–9)
per branch per sweep.

Common config: batch_size 32, num_layers 4, channels 512, max_steps_per_epoch
1000, lr 1e-4, warmup_steps 100, ff_dropout 0.3, attn_dropout 0.3, single-GPU
DDP, precompute cache cleared between seeds. ogPASS uses the 3-phase schedule
with 20/20/60 % of epochs spent on warmup / sampler_only / joint.

<!-- BEGIN SWEEP n1_ep10 -->
## Sweep `n1_ep10` — K=1, epochs=10

_K=1 seed-only. PASS `sampler_only` is a no-op; schedule shape kept._

### Aggregated (mean ± std over 10 seeds)

| branch | mae ↓ | rmse ↓ | r2 ↑ |
|---|---|---|---|
| dev-kyaw (n=10) | 4.2466 ± 0.1087 | 5.1373 ± 0.1271 | 0.0273 ± 0.0481 |
| ogPASS (n=10) | 4.0934 ± 0.1086 | 4.9401 ± 0.1072 | 0.1007 ± 0.0388 |

### Δ (ogPASS − dev-kyaw)  — negative is better for mae/rmse, positive for r2

| Δmae | Δrmse | Δr2 |
|---|---|---|
| -0.1533 | -0.1972 | +0.0734 |

### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)

| mae | rmse | r2 |
|---|---|---|
| 1.00× | 0.84× | 0.81× |

### Per-seed

#### dev-kyaw

| seed | mae | rmse | r2 |
|---:|---:|---:|---:|
| 0 | 4.1142 | 4.9568 | 0.0950 |
| 1 | 4.3156 | 5.2205 | -0.0039 |
| 2 | 4.0960 | 4.9914 | 0.0823 |
| 3 | 4.1935 | 5.1177 | 0.0353 |
| 4 | 4.2589 | 5.1149 | 0.0363 |
| 5 | 4.3285 | 5.2410 | -0.0118 |
| 6 | 4.2010 | 5.0889 | 0.0461 |
| 7 | 4.4272 | 5.3239 | -0.0441 |
| 8 | 4.3562 | 5.2909 | -0.0311 |
| 9 | 4.1754 | 5.0271 | 0.0691 |

#### ogPASS

| seed | mae | rmse | r2 |
|---:|---:|---:|---:|
| 0 | 4.2651 | 5.0869 | 0.0468 |
| 1 | 4.0564 | 4.9258 | 0.1063 |
| 2 | 4.0745 | 4.9138 | 0.1106 |
| 3 | 3.9802 | 4.8083 | 0.1484 |
| 4 | 4.1097 | 4.9396 | 0.1012 |
| 5 | 4.0637 | 4.9253 | 0.1065 |
| 6 | 4.2250 | 5.0499 | 0.0607 |
| 7 | 4.1080 | 4.9870 | 0.0839 |
| 8 | 4.1565 | 5.0281 | 0.0687 |
| 9 | 3.8946 | 4.7365 | 0.1736 |

### Notes on Exp1

At K=1 there are no neighbors for the sampler to pick, so the sampling
module is effectively inert. Both branches run the seed-only path, yet
ogPASS wins MAE by **−0.153** with comparable std. This can't come from
"better sampling" — it has to come from elsewhere:

- **Schedule mismatch.** dev-kyaw runs 10 straight joint-training epochs.
  ogPASS runs 2 warmup (task-only) + 2 sampler_only (task frozen) + 6 joint.
  The 2 sampler_only epochs freeze the task model entirely, so the task
  model effectively sees **8** training epochs on ogPASS vs **10** on
  dev-kyaw. Despite seeing fewer gradient updates, ogPASS still wins —
  which is surprising and suggests a different source.
- **Different init / optimizer state.** The two branches may differ in
  how they build the task model (e.g. codebook init, param groups,
  layernorm defaults). Any K=1 difference at the regression floor is a
  baseline offset between the branches, independent of sampling.
- **Implication.** Treat the K=1 delta as a **per-branch baseline offset**
  when reading Exp2/Exp3. If ogPASS beats dev-kyaw by ~0.15 MAE at K>1,
  that's not evidence of sampler value — it's the same baseline offset
  we see at K=1. Real sampler value would show up as ogPASS pulling
  further ahead at K=10/50 than it already is at K=1.

<!-- END SWEEP n1_ep10 -->

### Notes on Exp2

Exp2 confirms that `driver-position` has almost no graph signal:

- **dev-kyaw barely improves K=1 → K=10:** 4.247 → 4.224, Δ = −0.02.
  Adding 9 random neighbors buys essentially nothing.
- **ogPASS gets worse K=1 → K=10:** 4.093 → 4.203, Δ = +0.11.
  The 9 sampled neighbors are actively hurting the model.
- **Δmae shrinks from −0.153 to −0.022.** After subtracting the K=1
  baseline offset, the sampler effect is **+0.131** (harmful).
- **Variance ratio ~0.5×** — ogPASS is more stable, same "regularizer
  not discriminator" pattern from driver-top3.

### Summary table (offset-adjusted)

| K | ep | n | dev-kyaw MAE | ogPASS MAE | Δmae | Δmae − K=1 offset |
|---|---|---|---|---|---|---|
| 1  | 10 | 10 | 4.2466 | 4.0934 | −0.1533 | (reference) |
| 10 | 20 |  4 | 4.2243 | 4.2028 | −0.0215 | **+0.1318** (sampler hurts) |

The last column isolates the sampler's contribution from the branch-level
baseline offset. The sampler is not helping — and on this task, the graph
itself barely helps either.

Exp3 (K=50) was still running at time of writing. Given the K=10 result
and near-zero graph signal, Exp3 is unlikely to change the picture.

<!-- BEGIN SWEEP n10_ep20 -->
## Sweep `n10_ep20` — K=10, epochs=20

_ogPASS 3-phase: warmup=4, sampler_only=4, joint=12._

### Aggregated (mean ± std over 4 seeds)

| branch | mae ↓ | rmse ↓ | r2 ↑ |
|---|---|---|---|
| dev-kyaw (n=4) | 4.2243 ± 0.2625 | 5.2233 ± 0.2777 | -0.0071 ± 0.1057 |
| ogPASS (n=4) | 4.2028 ± 0.1306 | 5.1356 ± 0.1311 | 0.0280 ± 0.0500 |

### Δ (ogPASS − dev-kyaw)  — negative is better for mae/rmse, positive for r2

| Δmae | Δrmse | Δr2 |
|---|---|---|
| -0.0215 | -0.0877 | +0.0351 |

### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)

| mae | rmse | r2 |
|---|---|---|
| 0.50× | 0.47× | 0.47× |

### Per-seed

#### dev-kyaw

| seed | mae | rmse | r2 |
|---:|---:|---:|---:|
| 0 | 4.4512 | 5.4996 | -0.1141 |
| 1 | 4.1248 | 5.1798 | 0.0117 |
| 2 | 4.4226 | 5.3580 | -0.0575 |
| 3 | 3.8987 | 4.8559 | 0.1314 |

#### ogPASS

| seed | mae | rmse | r2 |
|---:|---:|---:|---:|
| 0 | 4.1525 | 5.0694 | 0.0534 |
| 1 | 4.2013 | 5.1436 | 0.0255 |
| 2 | 4.0750 | 5.0139 | 0.0740 |
| 3 | 4.3826 | 5.3155 | -0.0407 |

<!-- END SWEEP n10_ep20 -->

---

## Conclusion: driver-position is not the right task for sampler evaluation

The paper (RelGT, 2505.10960v2, Table 1) reports only 2.61% MAE
improvement from RDL → RelGT on `driver-position` with K=300. Our K=1
seed-only baseline already reaches MAE ~4.09, close to the paper's K=300
RDL result of 4.02. The entire graph signal on this task is worth ~0.17
MAE — well inside seed-to-seed noise.

F1 driver finishing position is dominated by tabular features on the
driver node itself (grid position, constructor quality, recent form).
The relational graph adds redundant information.

## Task selection for sampler experiments

For a learned sampler to demonstrate value, a task needs all three:
1. **Strong graph signal** — K=300 must substantially beat K=1
2. **Clean target distribution** — not degenerate or bimodal
3. **Large candidate neighborhood** — need >> K candidates so the
   sampler has room to discriminate

### RelBench regression tasks (from paper Table 1)

| task | RelGT % gain | label distribution | 2-hop neighborhood | verdict |
|---|---|---|---|---|
| rel-trial site-success | 18.4% | bimodal (0 or 1) | mean=66, **median=2**, mode=1 | dead: most nodes have ≤ 2 neighbors |
| rel-avito ad-ctr | 15.9% | concentrated near 0 | ? | bad labels |
| rel-hm item-sales | 4.3% | ? | ? | moderate signal at best |
| rel-f1 driver-position | 2.6% | clean | plenty | too weak (confirmed above) |
| everything else | < 2.3% | — | — | too weak |

### RelBench classification tasks (from paper Table 1)

| task | RelGT % AUC gain | class balance | 2-hop neighborhood | verdict |
|---|---|---|---|---|
| **rel-f1 driver-top3** | **10.6%** | ~80/20 imbalanced | plenty | **best available** (AP handles imbalance) |
| rel-f1 driver-dnf | 4.5% | imbalanced (DNF is rare) | plenty | weaker signal |
| rel-avito user-clicks | 3.6% | ? | ? | moderate |
| everything else | < 1% | — | — | too weak |

### Recommendation

**`rel-f1 / driver-top3` with AP as the primary metric remains the best
available RelBench task for evaluating the PASS sampler.**

- 10.6% AUC gain = real graph signal; enough headroom for a sampler
- AP (Average Precision) is the correct metric for imbalanced binary
  classification; it is also what RelBench uses as canonical metric
- Plenty of 2-hop neighbors for the sampler to discriminate
- Existing infrastructure: 15-seed baselines already in
  `experiment_results.md` with multiple sweep configs

The class imbalance is not a problem for AP — it is a problem for F1
and accuracy, which is why the earlier experiments showed noisy F1/Acc
while AP and AUC were more stable.

No regression task in RelBench combines strong graph signal, clean labels,
and dense neighborhoods. If a regression evaluation is required, a
synthetic benchmark with controllable graph signal would be needed.
