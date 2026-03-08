"""Precompute TabPFN unsupervised-permute embeddings for all node types.

Implements the Unsupervised-Permute method with leave-one-fold-out embedding
extraction, adapted from:
  "A Closer Look at TabPFN v2" (Ye et al., arXiv:2502.17361v2)

For each column j of a table, treat column j as a pseudo-target and the
remaining columns as features.  Use K-fold leave-one-fold-out to extract
held-out embeddings for every row, then mean-pool across columns to produce
a fixed-size embedding per row.
"""

import os
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _torchframe_to_dataframe(tf) -> pd.DataFrame:
    """Convert a TorchFrame to a pandas DataFrame preserving dtypes.

    - Numerical/timestamp columns -> float64
    - Categorical columns -> pd.CategoricalDtype (so TabPFN detects them)
    - Multicategorical columns -> exploded into binary indicator columns
    - Embedding columns -> flattened into individual float columns

    Returns a DataFrame where TabPFN's internal preprocessing can correctly
    identify and handle each column type.
    """
    import torch_frame
    from torch_frame.data.multi_nested_tensor import MultiNestedTensor

    feat_dict = tf.feat_dict
    col_names_dict = tf.col_names_dict
    parts = {}

    # ---- Numerical columns --------------------------------------------------
    if torch_frame.numerical in feat_dict:
        t = feat_dict[torch_frame.numerical]
        if t.dim() == 3:
            t = t.squeeze(-1)
        arr = t.cpu().float().numpy()
        names = col_names_dict[torch_frame.numerical]
        for i, name in enumerate(names):
            parts[f"num__{name}"] = arr[:, i].astype(np.float64)

    # ---- Timestamp columns --------------------------------------------------
    if torch_frame.timestamp in feat_dict:
        t = feat_dict[torch_frame.timestamp]
        if t.dim() == 3:
            t = t.squeeze(-1)
        arr = t.cpu().float().numpy()
        names = col_names_dict[torch_frame.timestamp]
        for i, name in enumerate(names):
            parts[f"ts__{name}"] = arr[:, i].astype(np.float64)

    # ---- Categorical columns (preserve as pd.Categorical) -------------------
    if torch_frame.categorical in feat_dict:
        t = feat_dict[torch_frame.categorical]
        if t.dim() == 3:
            t = t.squeeze(-1)
        arr = t.cpu().long().numpy()  # integer category indices
        names = col_names_dict[torch_frame.categorical]
        for i, name in enumerate(names):
            col = arr[:, i]
            # Map -1 (missing) to NaN, keep valid indices as string categories
            cats = []
            for v in col:
                cats.append(str(v) if v >= 0 else None)
            parts[f"cat__{name}"] = pd.Categorical(cats)

    # ---- Multicategorical columns (explode to binary indicators) ------------
    if torch_frame.multicategorical in feat_dict:
        mnt = feat_dict[torch_frame.multicategorical]
        names = col_names_dict[torch_frame.multicategorical]
        assert isinstance(mnt, MultiNestedTensor)
        n_rows = mnt.num_rows
        n_cols = mnt.num_cols
        for col_idx, col_name in enumerate(names):
            # Collect all unique category indices for this column
            all_cats = set()
            for row_idx in range(n_rows):
                vals = mnt[row_idx, col_idx]
                valid = vals[vals >= 0]  # -1 = missing
                all_cats.update(valid.tolist())
            all_cats = sorted(all_cats)
            if not all_cats:
                continue
            # Build binary indicator columns
            for cat_val in all_cats:
                indicator = np.zeros(n_rows, dtype=np.float64)
                for row_idx in range(n_rows):
                    vals = mnt[row_idx, col_idx]
                    if cat_val in vals.tolist():
                        indicator[row_idx] = 1.0
                parts[f"mcat__{col_name}__{cat_val}"] = indicator

    # ---- Embedding columns (flatten each dim as a float column) -------------
    if torch_frame.embedding in feat_dict:
        from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor

        t = feat_dict[torch_frame.embedding]
        if isinstance(t, MultiEmbeddingTensor):
            # values is already [num_rows, total_dims] (all cols concatenated)
            arr = t.values.cpu().float().numpy()
        elif t.dim() == 3:
            n, cols, d = t.shape
            arr = t.reshape(n, cols * d).cpu().float().numpy()
        else:
            arr = t.cpu().float().numpy()
        for i in range(arr.shape[1]):
            parts[f"emb__dim{i}"] = arr[:, i].astype(np.float64)

    if not parts:
        raise ValueError("No extractable features found in TorchFrame")

    df = pd.DataFrame(parts)

    # Impute NaN/Inf with column means for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = ~np.isfinite(df[col].values.astype(float))
        if mask.any():
            col_mean = np.nanmean(
                np.where(np.isfinite(df[col].values.astype(float)),
                         df[col].values.astype(float), np.nan)
            )
            if np.isnan(col_mean):
                col_mean = 0.0
            df.loc[mask, col] = col_mean

    return df


def precompute_tabpfn_embeddings(
    data,
    cache_dir: str,
    max_support_size: int = 3000,
    max_columns: int = 50,
    n_folds: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> int:
    """Compute and attach TabPFN unsupervised-permute embeddings.

    Uses leave-one-fold-out embedding extraction: for each column permutation,
    data is split into K folds. Each fold is used as the query set (test) while
    the remaining folds form the support set (train), ensuring every row's
    embedding is extracted in held-out context.

    Modifies *data* in-place by adding ``data[node_type].tabpfn_emb``
    (a ``[num_nodes, h]`` float tensor) for every node type.

    Node types without a ``.tf`` attribute receive zero embeddings.

    Args:
        data: PyG ``HeteroData`` with ``.tf`` per node type.
        cache_dir: Directory to cache ``.pt`` files.
        max_support_size: Max support rows per fold (subsampled from the
            K-1 training folds if they exceed this).
        max_columns: Max columns to permute over (randomly sampled if
            the table has more).
        n_folds: Number of folds for leave-one-fold-out extraction.
        device: Device for TabPFN inference (``"cpu"`` recommended for
            precomputation).
        seed: Random seed for reproducible column/support-set sampling.

    Returns:
        tabpfn_emb_dim (int): The embedding dimension *h*.
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.base import get_embeddings

    rng = np.random.RandomState(seed)
    os.makedirs(cache_dir, exist_ok=True)
    tabpfn_emb_dim = None
    pending_node_types = []

    for node_type in data.node_types:
        cache_path = os.path.join(cache_dir, f"{node_type}_tabpfn_emb.pt")

        # ---- load from cache ------------------------------------------------
        if os.path.exists(cache_path):
            logger.info("Loading cached TabPFN embeddings for '%s'", node_type)
            emb = torch.load(cache_path, weights_only=True)
            data[node_type].tabpfn_emb = emb
            tabpfn_emb_dim = emb.shape[1]
            continue

        # ---- node types without .tf get zero embeddings ---------------------
        if not hasattr(data[node_type], "tf"):
            logger.warning(
                "Node type '%s' has no .tf — will use zero embeddings", node_type
            )
            pending_node_types.append(node_type)
            continue

        # ---- build DataFrame with proper dtypes -----------------------------
        logger.info("Computing TabPFN embeddings for '%s' ...", node_type)
        tf_obj = data[node_type].tf
        try:
            df = _torchframe_to_dataframe(tf_obj)
        except ValueError as exc:
            logger.warning("  %s — will use zero embeddings", exc)
            pending_node_types.append(node_type)
            continue

        n, d = df.shape
        logger.info("  Table shape: %d rows x %d columns", n, d)

        if d <= 1:
            logger.warning("  Only %d column(s), deferring (zero embeddings)", d)
            pending_node_types.append(node_type)
            continue

        # ---- select columns to permute over ---------------------------------
        col_indices = (
            rng.choice(d, max_columns, replace=False)
            if d > max_columns
            else np.arange(d)
        )

        # ---- assign fold indices for leave-one-fold-out ---------------------
        actual_folds = min(n_folds, n)
        fold_ids = np.arange(n) % actual_folds
        rng.shuffle(fold_ids)

        # ---- column-permutation with leave-one-fold-out ---------------------
        col_embeddings = []
        n_failures = 0
        all_columns = list(df.columns)

        for j_pos, j in enumerate(col_indices):
            target_col = all_columns[j]
            is_categorical = isinstance(
                df[target_col].dtype, pd.CategoricalDtype
            )
            feature_cols = [c for c in all_columns if c != target_col]
            X_all = df[feature_cols]

            if is_categorical:
                # Use category codes as integer labels for classification
                y_all = df[target_col].cat.codes.values.copy()  # -1 = NaN
                # Need at least 2 distinct non-missing classes
                valid_classes = np.unique(y_all[y_all >= 0])
                if len(valid_classes) < 2:
                    continue
            else:
                y_all = df[target_col].values.astype(np.float64)
                # Skip constant pseudo-targets
                finite_y = y_all[np.isfinite(y_all)]
                if len(finite_y) == 0 or np.std(finite_y) < 1e-10:
                    continue

            # Allocate output array for this column's embeddings
            fold_emb = None  # will be [n, h] once we know h

            try:
                for fold_k in range(actual_folds):
                    query_mask = fold_ids == fold_k
                    support_mask = ~query_mask

                    X_support = X_all.iloc[support_mask]
                    y_support = y_all[support_mask]
                    X_query = X_all.iloc[query_mask]

                    if is_categorical:
                        # Filter out rows with missing categories (-1)
                        valid_support = y_support >= 0
                        X_support = X_support.iloc[valid_support]
                        y_support = y_support[valid_support]
                        # Need ≥2 classes in this fold's support
                        if len(np.unique(y_support)) < 2:
                            continue

                    # Subsample support set if too large
                    if len(X_support) > max_support_size:
                        sub_idx = rng.choice(
                            len(X_support), max_support_size, replace=False
                        )
                        X_support = X_support.iloc[sub_idx]
                        y_support = y_support[sub_idx]

                    if is_categorical:
                        model = TabPFNClassifier(
                            device=device, n_estimators=1
                        )
                    else:
                        model = TabPFNRegressor(
                            device=device, n_estimators=1
                        )
                    model.fit(X_support, y_support)
                    emb = get_embeddings(model, X_query, data_source="test")
                    assert emb.ndim == 3, (
                        f"Expected 3D embeddings, got shape {emb.shape}"
                    )
                    emb = emb.mean(axis=0)  # [n_query, h]

                    if fold_emb is None:
                        h = emb.shape[1]
                        fold_emb = np.zeros((n, h), dtype=np.float32)

                    fold_emb[query_mask] = emb

                if fold_emb is not None:
                    col_embeddings.append(fold_emb)
            except Exception as exc:
                n_failures += 1
                logger.warning("  Column %d (%s) failed: %s", j, target_col, exc)
                continue

            if (j_pos + 1) % 10 == 0:
                logger.info(
                    "  Processed %d / %d columns", j_pos + 1, len(col_indices)
                )

        if n_failures > 0:
            logger.warning(
                "  %d / %d columns failed for '%s'",
                n_failures, len(col_indices), node_type,
            )

        # ---- aggregate & store ----------------------------------------------
        if col_embeddings:
            E = np.stack(col_embeddings, axis=0).mean(axis=0)  # [n, h]
            emb_tensor = torch.from_numpy(E).float()
        else:
            logger.error(
                "  ALL columns failed for '%s' — using zero embeddings", node_type
            )
            pending_node_types.append(node_type)
            continue

        data[node_type].tabpfn_emb = emb_tensor
        torch.save(emb_tensor, cache_path)

        h = emb_tensor.shape[1]
        if tabpfn_emb_dim is not None and h != tabpfn_emb_dim:
            raise ValueError(
                f"TabPFN embedding dim mismatch for '{node_type}': "
                f"got {h}, expected {tabpfn_emb_dim}"
            )
        tabpfn_emb_dim = h
        logger.info("  Done: %s -> [%d, %d]", node_type, n, h)

    # ---- handle deferred single-column / empty / no-tf tables ---------------
    for node_type in pending_node_types:
        num = data[node_type].num_nodes
        h = tabpfn_emb_dim if tabpfn_emb_dim is not None else 1
        data[node_type].tabpfn_emb = torch.zeros(num, h)
        cache_path = os.path.join(cache_dir, f"{node_type}_tabpfn_emb.pt")
        torch.save(data[node_type].tabpfn_emb, cache_path)

    if tabpfn_emb_dim is None:
        tabpfn_emb_dim = 1

    return tabpfn_emb_dim
