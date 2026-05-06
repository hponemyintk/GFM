# Pretrain-Only 3-Trial Seed Sweep Results

**Date:** 2026-05-03  
**Branch:** `gfm_test_using-table_agnostic_model-branch`  
**Runtime:** 541m 58s total (3 seeds × ~180min each)

## Configuration

| Parameter | Value |
|-----------|-------|
| Seeds | 0, 1, 2 |
| Tasks | 8 (5 rel-f1 + 3 rel-event) |
| Epochs | 20 |
| K (neighbors) | 300 |
| Backbone dim | 512 |
| GPUs | 8× A100 (p4d.24xlarge) |
| Per-task budget | 1000 steps/epoch |
| Total MAX_STEPS | 1000 × 8 = 8000 |
| Script | `pretrain_only_seed_sweep.sh` |
| Output | `results/pretrain_only_3trials.out` |
| Aggregate JSON | `results/pretrain_only_seed_sweep/rel-f1+rel-event/aggregate.json` |

## Training Summary (Per Seed)

| Seed | Best Epoch | Best Val Macro |
|------|------------|----------------|
| 0 | 18 | 0.6004 |
| 1 | 4 | 0.6238 |
| 2 | 14 | 0.6030 |

## Tasks

| Dataset | Task | Type | Primary Metric |
|---------|------|------|----------------|
| rel-f1 | driver-dnf | Binary classification | AUROC |
| rel-f1 | driver-top3 | Binary classification | AUROC |
| rel-f1 | driver-position | Regression (forecasting) | MAE |
| rel-f1 | results-position | Regression (autocomplete) | R² |
| rel-f1 | qualifying-position | Regression (autocomplete) | R² |
| rel-event | user-repeat | Binary classification | AUROC |
| rel-event | user-ignore | Binary classification | AUROC |
| rel-event | user-attendance | Regression (forecasting) | R² |

---

## Scorecard

| Verdict | Count | Tasks |
|---------|-------|-------|
| **Win** | 4 | driver-dnf, driver-top3, results-position, qualifying-position |
| Neutral | 3 | driver-position, user-repeat, user-attendance |
| Loss | 1 | user-ignore |

### Comparison vs Published Baselines (Primary Metrics)

| Dataset | Task | Metric | Ours (mean±SD) | Prev. SOTA | Source | Δ vs SOTA | Verdict |
|---------|------|--------|----------------|------------|--------|-----------|---------|
| rel-f1 | driver-dnf | AUROC↑ | 0.779±0.007 | 0.759±0.041 | RelGT | +0.020 | **Win** |
| rel-f1 | driver-top3 | AUROC↑ | 0.881±0.018 | 0.835±0.034 | RelGT | +0.046 | **Win** |
| rel-f1 | results-position | R²↑ | 0.878±0.125 | 0.394 | GNN | +0.484 | **Win** |
| rel-f1 | qualifying-position | R²↑ | 0.922±0.092 | 0.015 | GNN | +0.907 | **Win** |
| rel-f1 | driver-position | MAE↓ | 3.58±0.56 | 3.92±0.34 | RelGT | −0.34 | Neutral |
| rel-event | user-repeat | AUROC↑ | 0.729±0.023 | 0.769±0.016 | GNN | −0.040 | Neutral |
| rel-event | user-attendance | R²↑ | −0.028±0.053 | 0.003±0.096 | GNN | −0.031 | Neutral |
| rel-event | user-ignore | AUROC↑ | 0.686±0.008 | 0.816±0.004 | RelGT | −0.130 | **Loss** |

---

## Detailed Per-Task Results (All Metrics, All Baselines)

### rel-f1.driver-position (Regression — Forecasting)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| R² | 0.265±0.235 | 0.033 | 0.503 | 0.259 |
| MAE | 3.580±0.563 | 4.091 | 2.977 | 3.672 |
| RMSE | 4.427±0.727 | 5.124 | 3.672 | 4.485 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | MAE | 3.917±0.345 | Table 11 (extended) |
| GNN (RBv2) | MAE | 4.022±0.119 | RelGT paper Table 11 |
| RelGT | R² | — (not reported) | — |
| GNN (RBv2) | R² | — (not reported) | — |

---

### rel-f1.driver-dnf (Binary Classification)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| AUROC | 0.779±0.007 | 0.778 | 0.786 | 0.773 |
| Avg Precision | 0.875±0.009 | 0.866 | 0.884 | 0.874 |
| Accuracy | 0.706±0.001 | 0.707 | 0.705 | 0.705 |
| F1 | 0.827±0.000 | 0.828 | 0.827 | 0.827 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | AUROC | 0.759±0.041 | Table 11 (extended) |
| GNN (RBv2) | AUROC | 0.726±0.003 | RelGT paper Table 11 |

---

### rel-f1.driver-top3 (Binary Classification)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| AUROC | 0.881±0.018 | 0.869 | 0.902 | 0.872 |
| Avg Precision | 0.575±0.059 | 0.515 | 0.633 | 0.578 |
| Accuracy | 0.832±0.015 | 0.847 | 0.818 | 0.831 |
| F1 | 0.423±0.291 | 0.565 | 0.616 | 0.089 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | AUROC | 0.835±0.034 | Table 11 (extended) |
| GNN (RBv2) | AUROC | 0.755±0.006 | RelGT paper Table 11 |

Note: F1 has very high variance (seed 2 = 0.089) due to threshold sensitivity at this class imbalance.

---

### rel-f1.results-position (Regression — Autocomplete)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| R² | 0.878±0.125 | 0.735 | 0.938 | 0.962 |
| MAE | 1.293±0.734 | 2.122 | 1.032 | 0.724 |
| RMSE | 1.711±0.918 | 2.757 | 1.330 | 1.044 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| GNN (RBv2) | R² | 0.394 | RBv2 Table 5 |
| RelGT | R² | — (not reported) | Autocomplete tasks not in RelGT paper |
| RelGT | MAE | — (not reported) | — |
| GNN (RBv2) | MAE | — (not reported) | — |

---

### rel-f1.qualifying-position (Regression — Autocomplete)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| R² | 0.922±0.092 | 0.977 | 0.972 | 0.816 |
| MAE | 1.322±0.868 | 0.783 | 0.861 | 2.323 |
| RMSE | 1.550±0.970 | 0.944 | 1.036 | 2.669 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| GNN (RBv2) | R² | 0.015 | RBv2 Table 5 |
| RelGT | R² | — (not reported) | Autocomplete tasks not in RelGT paper |
| RelGT | MAE | — (not reported) | — |
| GNN (RBv2) | MAE | — (not reported) | — |

Note: Seed 2 is an outlier (R²=0.816 vs 0.977/0.972) — possibly converged to a slightly different local minimum.

---

### rel-event.user-attendance (Regression — Forecasting)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| R² | −0.028±0.053 | −0.088 | −0.006 | 0.010 |
| MAE | 0.442±0.038 | 0.467 | 0.460 | 0.398 |
| RMSE | 0.651±0.017 | 0.670 | 0.644 | 0.639 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | MAE | 0.250±0.003 | Table 12 (extended) |
| GNN (RBv2) | MAE | 0.258±0.006 | RelGT paper Table 12 |
| GNN (RBv2) | R² | 0.003±0.096 | RBv2 Table 9 |
| RelGT | R² | — (not reported) | — |

Note: This task is inherently near-impossible. Best published R²=0.003±0.096 (GNN, RBv2 Table 9). All methods are near zero — this is a task ceiling, not a GFM-specific failure. Our MAE (0.442) is worse than RelGT (0.250) and GNN (0.258), but R² shows all methods explain essentially zero variance.

---

### rel-event.user-repeat (Binary Classification)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| AUROC | 0.729±0.023 | 0.706 | 0.731 | 0.752 |
| Avg Precision | 0.649±0.015 | 0.654 | 0.660 | 0.632 |
| Accuracy | 0.623±0.053 | 0.626 | 0.569 | 0.675 |
| F1 | 0.691±0.023 | 0.687 | 0.671 | 0.716 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | AUROC | 0.761±0.022 | Table 12 (extended) |
| GNN (RBv2) | AUROC | 0.769±0.016 | RelGT paper Table 12 |

---

### rel-event.user-ignore (Binary Classification)

#### Our Results

| Metric | Mean±SD | Seed 0 | Seed 1 | Seed 2 |
|--------|---------|--------|--------|--------|
| AUROC | 0.686±0.008 | 0.685 | 0.678 | 0.694 |
| Avg Precision | 0.227±0.009 | 0.228 | 0.236 | 0.218 |
| Accuracy | 0.870±0.000 | 0.870 | 0.870 | 0.870 |
| F1 | 0.000±0.000 | 0.000 | 0.000 | 0.000 |

#### Published Baselines

| Source | Metric | Value | Table/Reference |
|--------|--------|-------|-----------------|
| RelGT | AUROC | 0.816±0.004 | Table 12 (extended) |
| GNN (RBv2) | AUROC | 0.816±0.011 | RelGT paper Table 12 |

Note: F1=0 across all seeds indicates the model predicts the majority class only (87% negative). AUROC is above random (0.686 vs 0.5) but substantially below baselines. This is the most heavily class-imbalanced task.

---

## Raw Data Reference: All Published Baselines

### RelGT Paper (Tables 11/12 — Extended Results, 5 seeds)

| Dataset | Task | AUROC | MAE | R² |
|---------|------|-------|-----|-----|
| rel-f1 | driver-dnf | 0.759±0.041 | — | — |
| rel-f1 | driver-top3 | 0.835±0.034 | — | — |
| rel-f1 | driver-position | — | 3.917±0.345 | — (not reported) |
| rel-f1 | results-position | — (not in paper) | — | — |
| rel-f1 | qualifying-position | — (not in paper) | — | — |
| rel-event | user-repeat | 0.761±0.022 | — | — |
| rel-event | user-ignore | 0.816±0.004 | — | — |
| rel-event | user-attendance | — | 0.250±0.003 | — (not reported) |

Notes:
- RelGT paper only covers forecasting tasks (the original RelBench v1 tasks).
- Autocomplete tasks (results-position, qualifying-position) are RelBench v2 additions and not present in the RelGT paper.
- RelGT does not report R² for any task — only MAE for regression.

### GNN / RDL Baselines (RelBench v2 Paper + RelGT Paper)

| Dataset | Task | AUROC | MAE | R² | Source Table |
|---------|------|-------|-----|-----|-------------|
| rel-f1 | driver-dnf | 0.726±0.003 | — | — | RelGT Table 11 |
| rel-f1 | driver-top3 | 0.755±0.006 | — | — | RelGT Table 11 |
| rel-f1 | driver-position | — | 4.022±0.119 | — (not reported) | RelGT Table 11 |
| rel-f1 | results-position | — | — | 0.394 | RBv2 Table 5 |
| rel-f1 | qualifying-position | — | — | 0.015 | RBv2 Table 5 |
| rel-event | user-repeat | 0.769±0.016 | — | — | RelGT Table 12 |
| rel-event | user-ignore | 0.816±0.011 | — | — | RelGT Table 12 |
| rel-event | user-attendance | — | 0.258±0.006 | 0.003±0.096 | RelGT Table 12 / RBv2 Table 9 |

Notes:
- GNN AUROC and MAE values come from the RelGT paper's "RDL" column (Tables 11/12).
- GNN R² values for autocomplete tasks come from RelBench v2 paper Table 5.
- GNN R² for user-attendance comes from RelBench v2 paper Table 9.
- No SD reported for R² values from RBv2 Table 5 (results-position, qualifying-position).
- driver-position R² is not reported by any baseline paper.

---

## Key Observations

1. **rel-f1 tasks dominate:** 4/5 rel-f1 tasks are wins or neutral. The multi-task pretrain transfers strongly within the F1 dataset's graph structure.

2. **rel-event tasks underperform:** All 3 rel-event tasks are neutral or loss. user-ignore is the only clear loss (−0.130 AUROC vs RelGT).

3. **Autocomplete R² tasks show massive gains:** results-position and qualifying-position achieve 2× and 61× the GNN baseline respectively. These autocomplete tasks benefit heavily from the shared pretrained backbone.

4. **user-attendance is inherently near-impossible:** Best published R²=0.003±0.096 (GNN from RBv2 Table 9). All methods are near zero — this is a task ceiling, not a GFM-specific failure. However, our MAE (0.442) is notably worse than RelGT (0.250) and GNN (0.258).

5. **High variance on some tasks:** driver-position R² ranges from 0.033 to 0.503 across seeds. results-position ranges from 0.735 to 0.962. More trials would tighten these estimates.

6. **user-ignore class imbalance:** F1=0.000 across all seeds means the model never predicts the positive class. The 87% negative class ratio causes the model to default to majority-class prediction. AUROC still shows discriminative ability (0.686) but at no operating threshold does the model produce positive predictions.

7. **Seed 1 consistently strongest:** Best val macro (0.6238 at epoch 4), best driver-position (R²=0.503), best driver-top3 (AUROC=0.902). Early convergence (epoch 4) suggests this seed landed in a favorable region of the loss landscape.

8. **Baseline coverage gaps:** No published R² for driver-position from any method. RelGT doesn't cover autocomplete tasks at all. Complete apples-to-apples comparison only possible for AUROC (binary tasks) and MAE (driver-position, user-attendance).

## Baselines Sources

- **RelGT:** Tables 11/12 (extended results) from the RelGT paper (arXiv 2505.10960v2). Single-task training with standard deviations from 5 seeds.
- **GNN (RDL):** RelBench v2 paper (arXiv 2602.12606v1) Tables 3, 5, 9; also reported in RelGT paper Tables 11/12 as the "RDL" column.
- **Note:** RelGT does not report R² for any task. GNN R² for autocomplete tasks comes from RBv2 Table 5. GNN R² for user-attendance comes from RBv2 Table 9.

## Plots

- `plots/pretrain_3trial_charts.py` — 3-panel (AUROC + MAE + R²), original
- `plots/pretrain_3trial_charts_v2.py` — 2-panel (AUROC + R²), MAE in table only
- `plots/pretrain_3trial_table.py` — Standalone summary table (color-coded by verdict)
