# ogPASS vs dev-kyaw — Experiment Results

Multiple 15-seed sweeps comparing the PASS-GNN learned sampler (`ogPASS` branch)
against random neighbor sampling (`dev-kyaw` branch) on `rel-f1` /
`driver-top3`. All metrics are `Best Test` snapshots selected by best val AUC.

`num_neighbors` is K = total subgraph size **including the seed token**.
K=1 means seed-only (no neighbors).

## Common config

- batch_size 32, num_layers 4, channels 512
- max_steps_per_epoch 1000, lr 1e-4, warmup_steps 100
- ff_dropout 0.3, attn_dropout 0.3
- single-GPU DDP, RelBench precompute cache cleared between seeds
- seeds 0–14 (n=15 per branch)

## Sweep matrix

| sweep | num_neighbors | epochs | sampler config |
|---|---|---|---|
| **baseline_n50_ep10** | 50 | 10 | ogPASS: joint-from-step-1, original buggy sampler (Xavier `Ws`, no softmax) |
| **fix12_n30_ep30**    | 30 | 30 | ogPASS: joint-from-step-1, fix1 (softmax q_imp) + fix2 (zero-init `Ws`) applied |
| **warmup_n10_ep30**   | 10 | 30 | ogPASS: 3-phase schedule (10 task-warmup → 10 sampler-only on frozen task → 10 joint), K=10 |
| **seedonly_n1_ep30**  | 1  | 30 | Seed-only ablation. PASS sampler short-circuits (`K-1=0` branch added in `pass_sampler.py`). Tests the model's tabular-prior floor with zero graph context. |

## Aggregated results (mean ± std over 15 seeds)

| sweep | branch | AP | F1 | AUC | Acc |
|---|---|---|---|---|---|
| baseline_n50_ep10 | devkyaw | **0.3713 ± 0.0608** | 0.3921 ± 0.1707 | 0.7653 ± 0.0451 | 0.7320 ± 0.0676 |
| baseline_n50_ep10 | ogpass  | 0.3603 ± 0.0303 | **0.4104 ± 0.1484** | **0.7894 ± 0.0314** | **0.7369 ± 0.0547** |
| fix12_n30_ep30    | devkyaw | **0.4064 ± 0.0663** | **0.4547 ± 0.0550** | 0.7873 ± 0.0251 | 0.7367 ± 0.0670 |
| fix12_n30_ep30    | ogpass  | 0.3604 ± 0.0367 | 0.4279 ± 0.1154 | **0.7883 ± 0.0222** | **0.7430 ± 0.0551** |
| warmup_n10_ep30   | devkyaw | **0.3940 ± 0.1050** | 0.3609 ± 0.1521 | 0.7682 ± 0.0652 | **0.7821 ± 0.0598** |
| warmup_n10_ep30   | ogpass  | 0.3613 ± 0.0274 | **0.3650 ± 0.1644** | **0.7934 ± 0.0195** | 0.7566 ± 0.0525 |
| warmup_n1_ep10    | devkyaw | 0.2931 ± 0.0603 | 0.2317 ± 0.1495 | 0.7269 ± 0.0605 | 0.7360 ± 0.0380 |
| seedonly_n1_ep30  | devkyaw | **0.3243 ± 0.0246** | 0.1711 ± 0.1739 | **0.7691 ± 0.0294** | **0.7702 ± 0.0185** |
| seedonly_n1_ep30  | ogpass  | 0.3174 ± 0.0367 | **0.1886 ± 0.1763** | 0.7601 ± 0.0392 | 0.7421 ± 0.0409 |

### Δ (ogpass − devkyaw)

| sweep | ΔAP | ΔF1 | ΔAUC | ΔAcc |
|---|---|---|---|---|
| baseline_n50_ep10 | −0.0110 | +0.0183 | **+0.0241** | +0.0049 |
| fix12_n30_ep30    | **−0.0460** | −0.0268 | +0.0010 | +0.0063 |
| warmup_n10_ep30   | −0.0327 | +0.0041 | **+0.0252** | −0.0255 |
| seedonly_n1_ep30  | −0.0069 | +0.0175 | −0.0090 | −0.0281 |

### Variance ratio (ogpass std ÷ devkyaw std — lower = ogpass more stable)

| sweep | AP | F1 | AUC | Acc |
|---|---|---|---|---|
| baseline_n50_ep10 | 0.50× | 0.87× | **0.70×** | 0.81× |
| fix12_n30_ep30    | 0.55× | **2.10×** | 0.88× | 0.82× |
| warmup_n10_ep30   | **0.26×** | 1.08× | **0.30×** | 0.88× |
| seedonly_n1_ep30  | 1.49× | 1.01× | 1.33× | 2.21× |

## Per-seed tables

### baseline_n50_ep10

#### devkyaw

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3563 | 0.4724 | 0.7912 | 0.6309 |
| 1 | 0.3204 | 0.1529 | 0.7627 | 0.8017 |
| 2 | 0.3951 | 0.5358 | 0.8237 | 0.7231 |
| 3 | 0.4122 | 0.3169 | 0.6961 | 0.7328 |
| 4 | 0.3040 | 0.3546 | 0.7498 | 0.6791 |
| 5 | 0.3919 | 0.5195 | 0.7963 | 0.6942 |
| 6 | 0.5192 | 0.4459 | 0.6593 | 0.7603 |
| 7 | 0.4106 | 0.4857 | 0.7864 | 0.7521 |
| 8 | 0.3481 | 0.4251 | 0.7607 | 0.6460 |
| 9 | 0.4156 | 0.4548 | 0.7938 | 0.7590 |
| 10 | 0.2995 | 0.4636 | 0.7408 | 0.6143 |
| 11 | 0.4058 | 0.5587 | 0.8139 | 0.8499 |
| 12 | 0.3279 | 0.0764 | 0.7566 | 0.8003 |
| 13 | 0.2841 | 0.0471 | 0.7373 | 0.7769 |
| 14 | 0.3784 | 0.5714 | 0.8105 | 0.7603 |

#### ogpass

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3815 | 0.2654 | 0.8074 | 0.7865 |
| 1 | 0.3870 | 0.5075 | 0.8104 | 0.6818 |
| 2 | 0.3744 | 0.4341 | 0.8140 | 0.7989 |
| 3 | 0.3225 | 0.3686 | 0.7533 | 0.7782 |
| 4 | 0.3365 | 0.4821 | 0.7806 | 0.6804 |
| 5 | 0.3327 | 0.5085 | 0.7917 | 0.6804 |
| 6 | 0.3685 | 0.5000 | 0.8120 | 0.7658 |
| 7 | 0.3512 | 0.4933 | 0.7797 | 0.6887 |
| 8 | 0.3736 | 0.5251 | 0.8054 | 0.7135 |
| 9 | 0.3304 | 0.0000 | 0.6981 | 0.8196 |
| 10 | 0.3557 | 0.3022 | 0.8021 | 0.7837 |
| 11 | 0.3560 | 0.4927 | 0.7888 | 0.6653 |
| 12 | 0.3557 | 0.2547 | 0.7991 | 0.7824 |
| 13 | 0.3361 | 0.4871 | 0.7729 | 0.6722 |
| 14 | 0.4431 | 0.5354 | 0.8258 | 0.7562 |

### fix12_n30_ep30

#### devkyaw

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.4897 | 0.5012 | 0.8287 | 0.7094 |
| 1 | 0.4295 | 0.4467 | 0.7622 | 0.8499 |
| 2 | 0.3525 | 0.5097 | 0.8000 | 0.7218 |
| 3 | 0.5105 | 0.4315 | 0.7960 | 0.6226 |
| 4 | 0.3161 | 0.4978 | 0.7569 | 0.6915 |
| 5 | 0.4059 | 0.4165 | 0.7837 | 0.6680 |
| 6 | 0.3728 | 0.4770 | 0.7927 | 0.6405 |
| 7 | 0.5200 | 0.5360 | 0.8358 | 0.7782 |
| 8 | 0.4582 | 0.4321 | 0.7722 | 0.7466 |
| 9 | 0.4130 | 0.4542 | 0.7971 | 0.7782 |
| 10 | 0.3377 | 0.5286 | 0.7855 | 0.7052 |
| 11 | 0.4167 | 0.4167 | 0.8020 | 0.8072 |
| 12 | 0.3098 | 0.3769 | 0.7409 | 0.7769 |
| 13 | 0.3872 | 0.4536 | 0.7748 | 0.7245 |
| 14 | 0.3758 | 0.3422 | 0.7817 | 0.8306 |

#### ogpass

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3405 | 0.4973 | 0.7760 | 0.7410 |
| 1 | 0.3032 | 0.4664 | 0.7510 | 0.6722 |
| 2 | 0.3251 | 0.2537 | 0.7659 | 0.7893 |
| 3 | 0.3315 | 0.3231 | 0.7592 | 0.7865 |
| 4 | 0.3189 | 0.1163 | 0.7619 | 0.7906 |
| 5 | 0.3961 | 0.4842 | 0.7887 | 0.7300 |
| 6 | 0.3644 | 0.5090 | 0.8046 | 0.7369 |
| 7 | 0.4408 | 0.4947 | 0.8283 | 0.6708 |
| 8 | 0.3608 | 0.3750 | 0.7902 | 0.7934 |
| 9 | 0.3466 | 0.4296 | 0.7831 | 0.7769 |
| 10 | 0.3745 | 0.5202 | 0.8112 | 0.7713 |
| 11 | 0.3898 | 0.4554 | 0.8061 | 0.6047 |
| 12 | 0.3474 | 0.4877 | 0.7883 | 0.7424 |
| 13 | 0.3587 | 0.5204 | 0.8005 | 0.7893 |
| 14 | 0.4076 | 0.4859 | 0.8095 | 0.7493 |

### seedonly_n1_ep30

#### devkyaw

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3432 | 0.0761 | 0.7858 | 0.7658 |
| 1 | 0.3164 | 0.0377 | 0.7816 | 0.7893 |
| 2 | 0.3311 | 0.2915 | 0.7805 | 0.7590 |
| 3 | 0.3194 | 0.0333 | 0.7734 | 0.7603 |
| 4 | 0.3299 | 0.1161 | 0.7670 | 0.8113 |
| 5 | 0.3927 | 0.5864 | 0.8280 | 0.7824 |
| 6 | 0.3073 | 0.0535 | 0.7628 | 0.7562 |
| 7 | 0.3416 | 0.3521 | 0.7808 | 0.7466 |
| 8 | 0.3135 | 0.0382 | 0.7570 | 0.7920 |
| 9 | 0.3161 | 0.0333 | 0.7694 | 0.7603 |
| 10 | 0.2833 | 0.4103 | 0.6850 | 0.7466 |
| 11 | 0.3278 | 0.0333 | 0.7795 | 0.7603 |
| 12 | 0.2965 | 0.2157 | 0.7463 | 0.7796 |
| 13 | 0.3237 | 0.0333 | 0.7732 | 0.7603 |
| 14 | 0.3224 | 0.2559 | 0.7661 | 0.7837 |

#### ogpass

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.4087 | 0.4989 | 0.8083 | 0.6763 |
| 1 | 0.3063 | 0.0335 | 0.7668 | 0.7617 |
| 2 | 0.3210 | 0.0414 | 0.7809 | 0.8085 |
| 3 | 0.3332 | 0.3060 | 0.7672 | 0.7438 |
| 4 | 0.3100 | 0.0333 | 0.7697 | 0.7603 |
| 5 | 0.2479 | 0.2205 | 0.6607 | 0.7176 |
| 6 | 0.3321 | 0.1070 | 0.7780 | 0.7700 |
| 7 | 0.3139 | 0.0333 | 0.7692 | 0.7603 |
| 8 | 0.2528 | 0.0733 | 0.6832 | 0.7562 |
| 9 | 0.3048 | 0.2353 | 0.7548 | 0.7493 |
| 10 | 0.3158 | 0.0333 | 0.7675 | 0.7603 |
| 11 | 0.3190 | 0.2500 | 0.7817 | 0.7603 |
| 12 | 0.3250 | 0.0333 | 0.7770 | 0.7603 |
| 13 | 0.3343 | 0.5068 | 0.7949 | 0.6997 |
| 14 | 0.3369 | 0.4225 | 0.7413 | 0.6460 |

### warmup_n10_ep30

#### devkyaw

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.4127 | 0.5682 | 0.8314 | 0.7906 |
| 1 | 0.3724 | 0.2404 | 0.7852 | 0.8085 |
| 2 | 0.4034 | 0.4340 | 0.7755 | 0.8168 |
| 3 | 0.5119 | 0.2346 | 0.8282 | 0.8292 |
| 4 | 0.4730 | 0.4830 | 0.8202 | 0.7700 |
| 5 | 0.3166 | 0.4047 | 0.7599 | 0.7204 |
| 6 | 0.4108 | 0.3204 | 0.8018 | 0.8072 |
| 7 | 0.4586 | 0.4929 | 0.7749 | 0.8044 |
| 8 | 0.2832 | 0.0145 | 0.6445 | 0.8127 |
| 9 | 0.2676 | 0.1616 | 0.7119 | 0.7713 |
| 10 | 0.5850 | 0.5128 | 0.8456 | 0.8691 |
| 11 | 0.2738 | 0.3724 | 0.7218 | 0.7121 |
| 12 | 0.5343 | 0.5069 | 0.8379 | 0.8526 |
| 13 | 0.3647 | 0.3754 | 0.7396 | 0.7204 |
| 14 | 0.2417 | 0.2920 | 0.6443 | 0.6460 |

#### ogpass

| seed | AP | F1 | AUC | Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.3995 | 0.1910 | 0.8214 | 0.8017 |
| 1 | 0.3454 | 0.5440 | 0.7982 | 0.7645 |
| 2 | 0.3331 | 0.3281 | 0.7763 | 0.7631 |
| 3 | 0.3872 | 0.3364 | 0.8142 | 0.7989 |
| 4 | 0.3402 | 0.4396 | 0.7728 | 0.7893 |
| 5 | 0.3613 | 0.4611 | 0.7932 | 0.7328 |
| 6 | 0.3663 | 0.4198 | 0.7962 | 0.7906 |
| 7 | 0.4131 | 0.5242 | 0.8300 | 0.7025 |
| 8 | 0.3807 | 0.3364 | 0.8053 | 0.8044 |
| 9 | 0.3446 | 0.4730 | 0.7755 | 0.6777 |
| 10 | 0.3515 | 0.0152 | 0.7977 | 0.8209 |
| 11 | 0.3155 | 0.0382 | 0.7666 | 0.7920 |
| 12 | 0.3309 | 0.4798 | 0.7728 | 0.6983 |
| 13 | 0.3778 | 0.4028 | 0.7771 | 0.7631 |
| 14 | 0.3725 | 0.4848 | 0.8041 | 0.6488 |

## Interpretation

### Note on the right metric for this task
`driver-top3` is roughly 80/20 imbalanced (predicting all-negative gets
accuracy ≈ 0.80, which is what the seed-only F1 collapses to in many seeds).
For class-imbalanced binary classification, **Average Precision (AP)** is
the right primary metric — threshold-free, sensitive to ranking quality on
the *positive* class, with a random-baseline equal to the prevalence (~0.20).
**AUROC** is a useful secondary but less sensitive on imbalanced data
(it rewards ranking negatives correctly too). **F1** is threshold-dependent
and noisy here. **Accuracy** is uninformative — the prior beats most models.
RelBench itself reports AP as the canonical metric for binary node tasks.

Reading the Δ table through the AP lens: **ogpass loses every sweep**.
The earlier "ogpass wins AUC" framing flattered the policy with the wrong
metric.

### The seed-only floor (n=1, ep=30)
The K=1 ablation tells us how much of `driver-top3` is solvable from the
seed driver's own row features alone, with zero graph context:
- devkyaw n=1: **AP 0.324, AUC 0.769**
- devkyaw n=30 (ep=30): AP 0.406, AUC 0.787

So the *entire* graph signal — going from 0 neighbors to 29 — buys only
**+0.082 AP / +0.018 AUC**. Most of the task's solvability is already in
the tabular prior, which means the headroom for *any* sampler to demonstrate
value is small.

This sharpens the comparison: ogpass at K=10/30/50 sits at AP ≈ 0.36, which
is only **+0.04 above the seed-only floor**. devkyaw at K=30/ep=30 reaches
AP 0.406, which is **+0.08 above the floor** — twice the lift from the same
graph budget.

### Recurring pattern across all four sweeps
- ogPASS wins **AUC** by a small margin (+0.001 to +0.025) on the K>1 sweeps.
- ogPASS wins on **variance** for AP and AUC at K>1, often by 2–4×.
- ogPASS **loses AP** in every K>1 sweep (−0.011 to −0.046).
- At K=1 both branches collapse to within noise of each other (Δ AP −0.007),
  confirming that the K>1 differences come entirely from how the two branches
  use the candidate neighbors.

### Devkyaw response curve to neighbor budget (AP, mean over 15 seeds)

| K | epochs | devkyaw AP | ogpass AP |
|---|---|---|---|
| 1  | 30 | 0.324 | 0.317 |
| 10 | 30 | 0.394 | 0.361 |
| 30 | 30 | **0.406** | 0.360 |
| 50 | 10 | 0.371 | 0.360 |

Random sampling shows the expected diminishing-returns curve: gains from
1 → 10 → 30, then a slight regression at 50 (more random neighbors past
some point = more noise). **ogpass is flat at AP ≈ 0.36 across all
non-trivial K.**

This flatness has two possible readings:
1. *Saturation:* PASS already pulls all the informative neighbors at K=10
   and adding more candidates is correctly downweighted to noise.
2. *No discrimination:* PASS is effectively sampling near-uniformly and
   adding training noise that costs AP.

The data favors reading #2: at every K, ogpass loses to devkyaw on AP. A
truly saturating sampler should at minimum tie devkyaw at K=10 (both pick
9 from a similar pool, PASS in theory picking the best 9). The cleanest
test would be a smaller-K sweep (K=3 or K=5) — if PASS is discriminating,
smaller K should let it pull *ahead* of random because each pick binds
harder.

**Conclusion to date:** the learned PASS sampler behaves as a regularizer
(low-variance, slightly better AUC) rather than a discriminative policy
(better AP). Across +20 epochs, +1 reference-bug fix, a 3-phase schedule
with stationary REINFORCE rewards, and a seed-only floor establishing how
small the headroom is, ogPASS has not produced a single sweep-level win
on AP — the metric that matters for this imbalanced task.

Remaining hypotheses to test (not yet run):
1. **Smaller K (K=3 or K=5)** — directly tests the saturation hypothesis.
   If PASS is discriminating, smaller K should let it beat random because
   each pick binds harder.
2. **REINFORCE baseline** — variance reduction on the policy gradient.
   Currently `loss_up · selected_embed` has no baseline subtraction.
3. **Richer `Ws`** — a single linear projection on the raw tfs embedding may
   not carry enough task-relevant signal for the importance distribution.
