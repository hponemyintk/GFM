# Task 6: Zero-Shot Evaluation Pipeline

## Objective

Implement zero-shot prediction on the held-out dataset using ONLY the pretrained backbone — no fine-tuning, no additional training on the held-out dataset.

## Context

After pretraining with masked token prediction across N-1 datasets, we evaluate on the held-out dataset's tasks. The model has never seen this dataset's schema, tables, or labels. The table-agnostic encoders should generalize (GloVe for type names, shared encoders for features, etc.).

## Zero-Shot Strategy

### How to predict without a task-specific head

The pretrained backbone produces a representation for the seed node. For zero-shot prediction, we leverage the masked token prediction capability:

**Approach: Target-as-Masked-Token**

For classification tasks:
1. The task's target column exists as a feature in the seed node's table
2. At inference, mask the target feature for the seed node
3. The model predicts the masked feature → this IS the prediction
4. Map the reconstructed feature to the task's label space

For regression tasks:
1. Same approach — mask the target feature
2. The reconstructed feature value is the regression prediction

**Challenge**: This requires that the target column is present as a feature in the TensorFrame. In RelBench, the target IS a column in one of the tables. During pretraining, this column is part of the data (since we don't use labels, all columns are features). At evaluation, we mask specifically the target column and predict it.

**Alternative Approach: Nearest-Neighbor in Representation Space**

If the target-as-masked-token approach proves too difficult to implement:
1. Compute backbone representations for all nodes in the held-out dataset
2. For each test node, find the K nearest training nodes (from other datasets) in representation space
3. Use the training nodes' task labels as a vote/average for prediction
4. This requires a mapping from representation similarity to task labels, which may not transfer well across datasets

**Recommendation**: Implement the target-as-masked-token approach first. It's more principled and directly tests the foundation model's capability.

## Files to Create/Modify

### 1. New file: `zero_shot.py`

**`zero_shot_evaluate(model, masking_head, held_out_tokens, task, device)`**

```python
def zero_shot_evaluate(
    model: nn.Module,
    masking_head: nn.Module,
    loader: DataLoader,
    task: EntityTask,
    target_col_idx: int,    # which feature column is the target
    target_node_type: str,  # which table the target belongs to
    device: torch.device,
) -> np.ndarray:
    """
    Zero-shot prediction by masking the target feature and predicting it.

    Args:
        model: pretrained backbone (eval mode)
        masking_head: pretrained masked token head (eval mode)
        loader: DataLoader for held-out dataset's val/test split
        task: RelBench task object (for evaluation)
        target_col_idx: index of the target column in the TensorFrame
        target_node_type: namespaced table name where target lives
        device: compute device

    Returns:
        predictions: np.ndarray of predictions for all samples
    """
    model.eval()
    masking_head.eval()

    pred_list = []

    with torch.no_grad():
        for batch in loader:
            # Mask specifically the target feature for seed nodes (position 0)
            masked_batch = mask_target_column(batch, target_col_idx, target_node_type)

            # Forward through backbone
            representations = model.encode(
                masked_batch["neighbor_types"].to(device),
                masked_batch["node_indices"].to(device),
                masked_batch["neighbor_hops"].to(device),
                masked_batch["neighbor_times"].to(device),
                masked_batch["grouped_tf_dict"],
                edge_index=masked_batch["edge_index"].to(device),
                batch=masked_batch["batch"].to(device),
            )

            # Get prediction for seed node (position 0)
            pred = masking_head(representations)
            seed_feature_pred = pred["feature_pred"][:, 0, :]  # [B, channels]

            # Map from feature space to target space
            # This depends on the task type
            pred_list.append(seed_feature_pred.cpu().numpy())

    all_preds = np.concatenate(pred_list, axis=0)
    return all_preds
```

**`mask_target_column(batch, target_col_idx, target_node_type)`**

This function specifically masks the target column in the seed node's TensorFrame. Unlike random masking during pretraining, this is targeted masking of a specific feature.

Implementation details:
1. In the `grouped_tfs`, find the TensorFrame entries for `target_node_type`
2. Identify which rows correspond to seed nodes (position 0 in each sample)
3. Zero out the `target_col_idx` column in those rows
4. Return the modified batch

### 2. Integration with `main_foundation.py`

At the end of each epoch (or after training):

```python
# Zero-shot evaluation on held-out dataset
for (dataset_name, task_name), tokens in held_out_tokens.items():
    task = get_task(dataset_name, task_name)
    loader = DataLoader(tokens, batch_size=args.batch_size, collate_fn=tokens.collate)

    preds = zero_shot_evaluate(
        model=model.module,
        masking_head=masking_head,
        loader=loader,
        task=task,
        target_col_idx=find_target_col_idx(task, tokens),
        target_node_type=f"{dataset_name}__{task.entity_table}",
        device=device,
    )

    if local_rank == 0:
        metrics = task.evaluate(preds, task.get_table("test"))
        print(f"Zero-shot {dataset_name}/{task_name}: {metrics}")
        wandb.log({f"zeroshot_{dataset_name}_{task_name}_{k}": v
                    for k, v in metrics.items()})
```

### 3. Mapping feature predictions to task labels

This is the most nuanced part. The masked token head predicts a `channels`-dimensional feature vector. We need to map this to the actual task label space:

**For binary classification:**
- The target column is typically 0/1
- After feature reconstruction, apply a sigmoid to the first dimension
- Or: train a single linear layer (1 parameter essentially) — this is borderline "not zero-shot" but is extremely minimal

**For regression:**
- The target column is a continuous value
- The reconstructed feature should approximate the original value
- Need to de-normalize (reverse Z-score) using the col_stats

**For multilabel classification:**
- The target is a multi-hot vector
- Reconstruct the full vector, apply sigmoid per label

**Important consideration**: The feature reconstruction head predicts in the model's internal representation space, not the original feature space. We need a mapping from representation space back to the original feature values. This mapping is implicitly learned during pretraining (the reconstruction loss trains the model to predict original features), but the output is `channels`-dimensional, not 1-dimensional.

**Revised approach**: The reconstruction head should predict the original feature VALUE (before encoding), not the encoded representation. This means:
- For numerical targets: predict a scalar (the original column value)
- For categorical targets: predict a class index
- This requires a per-stype decoder in the masked token head

## Open Questions

1. **How to handle the target column at pretraining time**: Should the target column be included as a regular feature during pretraining? YES — during pretraining, there are no "labels" vs "features" distinction. All columns are features. The target column is just another numerical/categorical column that gets masked and reconstructed like any other.

2. **How well does this transfer**: The model learns to reconstruct features of tables it has seen. For the held-out dataset, it's reconstructing features of tables it has NEVER seen. The quality of this transfer depends entirely on how well the table-agnostic encoders generalize.

3. **Baseline comparison**: Compare zero-shot results against:
   - Random prediction
   - Majority class / mean prediction
   - A simple feature-based baseline (e.g., predict mean of numerical columns)
   - The fully supervised single-dataset RelGT (upper bound)

## Testing

- Verify zero-shot pipeline runs end-to-end without errors on a small held-out dataset
- Verify predictions have the correct shape and value range for each task type
- Verify that the target column is properly masked (not leaked) during evaluation
- Verify metrics can be computed via `task.evaluate()`
- Sanity check: zero-shot performance should be better than random

## Dependencies

- Task 2 (backbone separation) — needs `model.encode()` returning full sequence
- Task 3 (data loading) — needs held-out dataset tokens with union mappings
- Task 4 (training script) — integration point for evaluation during/after training
- Task 5 (masking) — reuses masking infrastructure, but targeted instead of random
