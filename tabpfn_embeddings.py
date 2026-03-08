"""Precompute TabPFN unsupervised-permute embeddings for all node types.

Implements the Unsupervised-Permute method from:
  "A Closer Look at TabPFN v2" (Ye et al., arXiv:2502.17361v2)

For each column j of a table, treat column j as a pseudo-target and the
remaining columns as features.  Fit TabPFNRegressor on a support subset,
extract embeddings for all rows, then mean-pool across columns to produce
a fixed-size embedding per row.
"""

import os
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_flat_features(tf) -> np.ndarray:
    """Extract all features from a TorchFrame as a flat numpy array.

    Handles numerical, categorical, timestamp, and embedding stypes.
    Returns shape [n_rows, total_features].
    """
    import torch_frame

    parts = []
    feat_dict = tf.feat_dict

    for st in (torch_frame.numerical, torch_frame.categorical, torch_frame.timestamp):
        if st in feat_dict:
            t = feat_dict[st]
            if t.dim() == 3:
                t = t.squeeze(-1)
            parts.append(t.cpu().float().numpy())

    # Always include embedding features (e.g. GloVe) when present
    if torch_frame.embedding in feat_dict:
        t = feat_dict[torch_frame.embedding]
        if t.dim() == 3:
            n, cols, d = t.shape
            t = t.reshape(n, cols * d)
        parts.append(t.cpu().float().numpy())

    if not parts:
        raise ValueError("No extractable features found in TorchFrame")

    X = np.concatenate(parts, axis=1)

    # Impute NaN / Inf with column means
    X = np.where(np.isfinite(X), X, np.nan)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X.astype(np.float32)


def precompute_tabpfn_embeddings(
    data,
    cache_dir: str,
    max_support_size: int = 3000,
    max_columns: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> int:
    """Compute and attach TabPFN unsupervised-permute embeddings.

    Modifies *data* in-place by adding ``data[node_type].tabpfn_emb``
    (a ``[num_nodes, h]`` float tensor) for every node type.

    Node types without a ``.tf`` attribute receive zero embeddings.

    Args:
        data: PyG ``HeteroData`` with ``.tf`` per node type.
        cache_dir: Directory to cache ``.pt`` files.
        max_support_size: Max rows used as TabPFN support set.
        max_columns: Max columns to permute over (randomly sampled if
            the table has more).
        device: Device for TabPFN inference (``"cpu"`` recommended for
            precomputation).
        seed: Random seed for reproducible column/support-set sampling.

    Returns:
        tabpfn_emb_dim (int): The embedding dimension *h*.
    """
    from tabpfn import TabPFNRegressor
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

        # ---- extract flat feature matrix ------------------------------------
        logger.info("Computing TabPFN embeddings for '%s' ...", node_type)
        tf = data[node_type].tf
        try:
            X = extract_flat_features(tf)
        except ValueError as exc:
            logger.warning("  %s — will use zero embeddings", exc)
            pending_node_types.append(node_type)
            continue

        n, d = X.shape
        logger.info("  Table shape: %d rows x %d columns", n, d)

        if d <= 1:
            logger.warning("  Only %d column(s), deferring (zero embeddings)", d)
            pending_node_types.append(node_type)
            continue

        # ---- select columns & support set -----------------------------------
        col_indices = (
            rng.choice(d, max_columns, replace=False)
            if d > max_columns
            else np.arange(d)
        )
        support_size = min(n, max_support_size)
        support_idx = rng.choice(n, support_size, replace=False)

        # ---- column-permutation embedding -----------------------------------
        col_embeddings = []
        n_failures = 0
        for j_pos, j in enumerate(col_indices):
            y = X[:, j].copy()
            X_mj = np.delete(X, j, axis=1)

            # skip constant pseudo-targets
            if np.std(y[support_idx]) < 1e-10:
                continue

            try:
                reg = TabPFNRegressor(device=device, n_estimators=1)
                reg.fit(X_mj[support_idx], y[support_idx])
                emb = get_embeddings(reg, X_mj, data_source="test")
                assert emb.ndim == 3, f"Expected 3D embeddings, got shape {emb.shape}"
                emb = emb.mean(axis=0)  # [n, h]
                col_embeddings.append(emb)
            except Exception as exc:
                n_failures += 1
                logger.warning("  Column %d failed: %s", j, exc)
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
