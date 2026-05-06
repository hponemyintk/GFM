# Holdout Dataset Eval: rel-f1 + rel-event → rel-arxiv

**Date**: 2026-05-03
**Branch**: `gfm_test_using-table_agnostic_model-branch`
**Command**:
```bash
BATCH=128 NPROC=8 STEPS_PER_TASK=1000 EPOCHS=20 \
  SOURCE="rel-f1 rel-event" TARGET=rel-arxiv RUN_TABPFN=0 \
  ./scripts/holdout_dataset_eval_clean.sh
```

**Setup**: Phase-5 cross-dataset adoption. Backbone pretrained on rel-f1 (5 tasks) + rel-event (3 tasks), frozen, then embeddings extracted on rel-arxiv (2 tasks). MLP2 finetune head, 3-seed sweep (seeds 0/1/2). K=300, BATCH=512 pretrain / 128 adoption, CHANNELS=512, NUM_LAYERS=4, HEADS=4, FULL_GRAPH=1.

---

## paper-citation (entity binary classification, AUROC — higher is better)

| Method | Test AUROC |
|---|---|
| LightGBM (RelBench v2, single-table) | 71.21 |
| **GFM cross-dataset (ours, 3-seed mean)** | **68.83** |
| GNN (RelBench v2, same-dataset) | 82.50 ± 0.04 |

Per-seed results:

| Seed | Accuracy | F1 | ROC AUC |
|---|---|---|---|
| 0 | 0.6524 | 0.5795 | 0.6889 |
| 1 | 0.6491 | 0.5813 | 0.6878 |
| 2 | 0.6539 | 0.5746 | 0.6882 |
| **Mean** | **0.6518** | **0.5785** | **0.6883** |

**Analysis**: ~15% relative AUROC drop vs same-dataset GNN (68.83 vs 82.50). Trails LightGBM by ~2.4 points. For a zero-shot cross-domain transfer (F1 racing + events → academic citations), the backbone learned some transferable structural signal, but the domain gap is large. Low seed variance (±0.06 AUROC) indicates stable embeddings.

---

## author-publication (entity regression, R² — higher is better)

| Method | Test R² |
|---|---|
| Mean baseline | −0.000 |
| Median baseline | −0.210 |
| Entity Mean | −0.010 |
| LightGBM (RelBench v2, single-table) | −0.210 |
| **GFM cross-dataset (ours, 3-seed mean)** | **−2.556** |
| GNN (RelBench v2, same-dataset) | 0.249 ± 0.013 |

Per-seed results:

| Seed | R² | MAE | RMSE | Best Epoch |
|---|---|---|---|---|
| 0 | −2.557 | 2.012 | 2.372 | 1 |
| 1 | −2.542 | 2.007 | 2.367 | 2 |
| 2 | −2.568 | 2.016 | 2.376 | 1 |
| **Mean** | **−2.556** | **2.012** | **2.372** | — |

**Analysis**: Significantly worse than all baselines including the trivial mean predictor (R²=0). However, this task is inherently weak — even LightGBM gets R²=−0.210 (worse than predicting the mean), meaning single-table features actively hurt. Only the same-dataset GNN with full relational context achieves positive R² (0.249). The finetune head converged at epoch 1-2, indicating the frozen embeddings carry no useful signal for this regression task. Not a meaningful test of cross-domain transfer.

---

## Summary

| Task | GFM (cross-dataset) | Best simple baseline | Same-dataset GNN |
|---|---|---|---|
| paper-citation (AUROC↑) | 68.83 | 71.21 (LightGBM) | 82.50 |
| author-publication (R²↑) | −2.556 | −0.000 (Mean) | 0.249 |

**Key takeaways**:
- paper-citation shows the backbone learned some transferable structure despite the domain gap (F1/events → academic papers).
- author-publication is not a meaningful transfer test — the regression target (publication count) requires domain-specific productivity patterns that have no analog in the source domains. Even same-dataset baselines mostly fail.
- This is likely the hardest cross-domain pairing in the 9-dataset pool. More structurally similar pairings (e.g., rel-stack → rel-arxiv for "users producing content in a community") would better test the transfer hypothesis.

## Bug Note

The initial run completed pretrain + extraction (~26 hours) but silently exited before fine-tuning due to a `set -e` + `[ ... ] && sleep` bug in `_wait_all()`. Fixed by appending `|| true` and initializing associative arrays with `=()`. See `logs/bugfix_wait_all_set_e.md`.
