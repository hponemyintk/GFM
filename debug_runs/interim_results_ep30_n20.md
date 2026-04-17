# ogPASS vs dev-kyaw — Interim Results (20 seeds, 30 epochs)

Generated: 2026-04-17 09:21  
Dataset: `rel-f1 / driver-top3` | Tune metric: AP (higher = better)  
Config: batch=32, layers=4, channels=512, lr=1e-4, K swept over [50,10,30,20,40,300]  
ogPASS 3-phase schedule: warmup=6, sampler\_only=6, joint=18 epochs

## Cross-K Summary (Test AP)

| K | status | dev-kyaw AP | ogPASS AP | Δ (og−dev) | dev-kyaw std | ogPASS std | var ratio |
|---|---|---|---|---|---|---|---|
| 50 | ✅ 40/40 | 0.4027 ± 0.0481 | 0.3758 ± 0.0388 | **-0.0268** dev-kyaw | 0.0481 | 0.0388 | 0.81× |
| 10 | ✅ 40/40 | 0.3556 ± 0.0937 | 0.3689 ± 0.0241 | **+0.0133** ✅ ogPASS | 0.0937 | 0.0241 | 0.26× |
| 30 | 🔄 32/40 | 0.4127 ± 0.1109 | 0.3794 ± 0.0256 | **-0.0333** dev-kyaw | 0.1109 | 0.0256 | 0.23× |
| 20 | ⏳ pending | — | — | — | — | — | — |
| 40 | ⏳ pending | — | — | — | — | — | — |
| 300 | ⏳ pending | — | — | — | — | — | — |

Variance ratio < 1.0 = ogPASS is more stable.

## K=50 (complete)

### Val AP (best epoch) vs Test AP

| branch | n | best val AP | test AP | gap |
|---|---|---|---|---|
| dev-kyaw | 20 | 0.4979 | 0.4027 | -0.0953 |
| ogPASS | 20 | 0.5899 | 0.3758 | -0.2141 |

### Per-seed Test AP

| seed | dev-kyaw | ogPASS |
|---|---|---|
| 0 | 0.4024 | 0.3602 |
| 1 | 0.4076 | 0.3528 |
| 2 | 0.3688 | 0.4136 |
| 3 | 0.4017 | 0.3354 |
| 4 | 0.3032 | 0.3815 |
| 5 | 0.3899 | 0.3702 |
| 6 | 0.4372 | 0.4329 |
| 7 | 0.4026 | 0.4031 |
| 8 | 0.4042 | 0.3296 |
| 9 | 0.3960 | 0.3831 |
| 10 | 0.3428 | 0.4085 |
| 11 | 0.4379 | 0.3992 |
| 12 | 0.5250 | 0.3463 |
| 13 | 0.4303 | 0.4004 |
| 14 | 0.3907 | 0.3690 |
| 15 | 0.4220 | 0.4305 |
| 16 | 0.4711 | 0.3356 |
| 17 | 0.3312 | 0.3847 |
| 18 | 0.3793 | 0.4037 |
| 19 | 0.4093 | 0.2763 |

## K=10 (complete)

### Val AP (best epoch) vs Test AP

| branch | n | best val AP | test AP | gap |
|---|---|---|---|---|
| dev-kyaw | 20 | 0.3936 | 0.3556 | -0.0380 |
| ogPASS | 20 | 0.4453 | 0.3689 | -0.0764 |

### Per-seed Test AP

| seed | dev-kyaw | ogPASS |
|---|---|---|
| 0 | 0.3580 | 0.3841 |
| 1 | 0.2594 | 0.3636 |
| 2 | 0.2705 | 0.3821 |
| 3 | 0.4568 | 0.3502 |
| 4 | 0.4154 | 0.3734 |
| 5 | 0.3433 | 0.3752 |
| 6 | 0.3590 | 0.4002 |
| 7 | 0.4411 | 0.4120 |
| 8 | 0.3278 | 0.3642 |
| 9 | 0.3255 | 0.3797 |
| 10 | 0.4406 | 0.3786 |
| 11 | 0.5876 | 0.3224 |
| 12 | 0.3993 | 0.3931 |
| 13 | 0.1669 | 0.3393 |
| 14 | 0.3932 | 0.3656 |
| 15 | 0.2815 | 0.3553 |
| 16 | 0.3286 | 0.3716 |
| 17 | 0.3240 | 0.3905 |
| 18 | 0.2238 | 0.3609 |
| 19 | 0.4096 | 0.3164 |

## K=30 (partial — 32/40 runs)

### Val AP (best epoch) vs Test AP

| branch | n | best val AP | test AP | gap |
|---|---|---|---|---|
| dev-kyaw | 16 | 0.4664 | 0.4127 | -0.0537 |
| ogPASS | 16 | 0.5602 | 0.3794 | -0.1808 |

### Per-seed Test AP

| seed | dev-kyaw | ogPASS |
|---|---|---|
| 0 | 0.3858 | 0.3998 |
| 1 | 0.4317 | 0.3840 |
| 2 | 0.5009 | 0.3431 |
| 3 | 0.3709 | 0.3789 |
| 4 | 0.2684 | 0.3610 |
| 5 | 0.6267 | 0.3682 |
| 6 | 0.5284 | 0.3589 |
| 7 | 0.3599 | 0.3312 |
| 8 | 0.5715 | 0.3634 |
| 9 | 0.3499 | 0.3979 |
| 10 | 0.2877 | 0.4136 |
| 11 | 0.3798 | 0.3685 |
| 12 | 0.2966 | 0.4231 |
| 13 | 0.2695 | 0.3767 |
| 14 | 0.4910 | 0.3934 |
| 15 | 0.4841 | 0.4087 |

## Training Curves

![K=50](training_curves_n50_ep30.png)

![K=10](training_curves_n10_ep30.png)

**Key observations:**
- K=10: ogPASS val AP improves continuously through joint phase — no collapse
- K=50: ogPASS val AP drops sharply at joint phase onset, converging to dev-kyaw level
- ogPASS consistently lower test AP than best val AP (val→test gap ~0.18 vs ~0.08 for dev-kyaw)
- ogPASS has much lower variance across seeds at all K values
