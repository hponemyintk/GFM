# ogPASS vs dev-kyaw — driver-top3 baseline-fix results

Head-to-head on `rel-f1 / driver-top3` (binary classification). Tune metric is
**AP** (higher = better); we also report AUC, F1, and Accuracy. 5 seeds (0–4)
per branch per sweep.

**Fixes applied:** REINFORCE EMA baseline (--use_reinforce_baseline),
tune_metric changed from AUC to AP.

Common config: batch_size 32, num_layers 4, channels 512, max_steps_per_epoch
1000, lr 1e-4, warmup_steps 100, ff_dropout 0.3, attn_dropout 0.3, single-GPU
DDP, precompute cache cleared between seeds. ogPASS uses the 3-phase schedule
with 20/20/60 % of epochs spent on warmup / sampler_only / joint.

<!-- BEGIN SWEEP n1_ep20 -->
## Sweep `n1_ep20` — K=1, epochs=20

_K=1 seed-only. PASS `sampler_only` is a no-op; schedule shape kept._

### Aggregated (mean ± std over 5 seeds)

| branch | AP ↑ | AUC ↑ | F1 ↑ | Acc ↑ |
|---|---|---|---|---|
| dev-kyaw (n=5) | 0.3170 ± 0.0056 | 0.7710 ± 0.0085 | 0.0524 ± 0.0216 | 0.7716 ± 0.0215 |
| ogPASS (n=5) | 0.3058 ± 0.0094 | 0.7613 ± 0.0067 | 0.1298 ± 0.2331 | 0.7813 ± 0.0268 |

### Δ (ogPASS − dev-kyaw) — positive is better for all metrics

| ΔAP | ΔAUC | ΔF1 | ΔAcc |
|---|---|---|---|
| -0.0112 | -0.0097 | +0.0774 | +0.0096 |

### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)

| AP | AUC | F1 | Acc |
|---|---|---|---|
| 1.69× | 0.78× | 10.79× | 1.25× |

### Per-seed

#### dev-kyaw

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3211 | 0.7780 | 0.0546 | 0.7617 |
| 1 | 0.3112 | 0.7673 | 0.0333 | 0.7603 |
| 2 | 0.3241 | 0.7821 | 0.0333 | 0.7603 |
| 3 | 0.3121 | 0.7632 | 0.0860 | 0.7658 |
| 4 | 0.3163 | 0.7645 | 0.0548 | 0.8099 |

#### ogPASS

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.2935 | 0.7522 | 0.0328 | 0.7562 |
| 1 | 0.3130 | 0.7686 | 0.0000 | 0.8237 |
| 2 | 0.3000 | 0.7574 | 0.5460 | 0.7824 |
| 3 | 0.3166 | 0.7664 | 0.0368 | 0.7837 |
| 4 | 0.3058 | 0.7620 | 0.0333 | 0.7603 |

<!-- END SWEEP n1_ep20 -->

<!-- BEGIN SWEEP n10_ep20 -->
## Sweep `n10_ep20` — K=10, epochs=20

_ogPASS 3-phase: warmup=4, sampler_only=4, joint=12._

### Aggregated (mean ± std over 5 seeds)

| branch | AP ↑ | AUC ↑ | F1 ↑ | Acc ↑ |
|---|---|---|---|---|
| dev-kyaw (n=5) | 0.3325 ± 0.1156 | 0.7008 ± 0.0873 | 0.2397 ± 0.2118 | 0.7650 ± 0.0552 |
| ogPASS (n=5) | 0.3505 ± 0.0141 | 0.7984 ± 0.0093 | 0.2868 ± 0.2592 | 0.7647 ± 0.0644 |

### Δ (ogPASS − dev-kyaw) — positive is better for all metrics

| ΔAP | ΔAUC | ΔF1 | ΔAcc |
|---|---|---|---|
| +0.0180 | +0.0975 | +0.0471 | -0.0003 |

### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)

| AP | AUC | F1 | Acc |
|---|---|---|---|
| 0.12× | 0.11× | 1.22× | 1.17× |

### Per-seed

#### dev-kyaw

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.2766 | 0.6249 | 0.3241 | 0.7300 |
| 1 | 0.5003 | 0.8234 | 0.4500 | 0.7879 |
| 2 | 0.2641 | 0.6669 | 0.3979 | 0.6873 |
| 3 | 0.4016 | 0.7595 | 0.0000 | 0.8237 |
| 4 | 0.2198 | 0.6293 | 0.0263 | 0.7961 |

#### ogPASS

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3356 | 0.7913 | 0.5308 | 0.7273 |
| 1 | 0.3526 | 0.7896 | 0.4838 | 0.6708 |
| 2 | 0.3685 | 0.8103 | 0.4047 | 0.7893 |
| 3 | 0.3585 | 0.8062 | 0.0000 | 0.8237 |
| 4 | 0.3371 | 0.7945 | 0.0145 | 0.8127 |

<!-- END SWEEP n10_ep20 -->

<!-- BEGIN SWEEP n50_ep20 -->
## Sweep `n50_ep20` — K=50, epochs=20

_ogPASS 3-phase: warmup=4, sampler_only=4, joint=12._

### Aggregated (mean ± std over 5 seeds)

| branch | AP ↑ | AUC ↑ | F1 ↑ | Acc ↑ |
|---|---|---|---|---|
| dev-kyaw (n=5) | 0.3398 ± 0.1270 | 0.7260 ± 0.0869 | 0.3308 ± 0.2015 | 0.7331 ± 0.1072 |
| ogPASS (n=5) | 0.3710 ± 0.0426 | 0.8043 ± 0.0278 | 0.4178 ± 0.1344 | 0.7675 ± 0.0446 |

### Δ (ogPASS − dev-kyaw) — positive is better for all metrics

| ΔAP | ΔAUC | ΔF1 | ΔAcc |
|---|---|---|---|
| +0.0312 | +0.0782 | +0.0869 | +0.0344 |

### Variance ratio (ogPASS std ÷ dev-kyaw std — lower = ogPASS more stable)

| AP | AUC | F1 | Acc |
|---|---|---|---|
| 0.34× | 0.32× | 0.67× | 0.42× |

### Per-seed

#### dev-kyaw

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.2893 | 0.7251 | 0.4276 | 0.5537 |
| 1 | 0.5224 | 0.8072 | 0.4737 | 0.8072 |
| 2 | 0.4123 | 0.8150 | 0.4738 | 0.7369 |
| 3 | 0.2026 | 0.6151 | 0.0000 | 0.8237 |
| 4 | 0.2723 | 0.6678 | 0.2791 | 0.7438 |

#### ogPASS

| seed | AP | AUC | F1 | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3357 | 0.7776 | 0.4831 | 0.7259 |
| 1 | 0.4247 | 0.8370 | 0.4167 | 0.8264 |
| 2 | 0.4086 | 0.8297 | 0.1878 | 0.7975 |
| 3 | 0.3513 | 0.7973 | 0.5275 | 0.7631 |
| 4 | 0.3348 | 0.7797 | 0.4737 | 0.7245 |

<!-- END SWEEP n50_ep20 -->
