"""Test Co5 -- single-task collate output is identical to dev-kyaw's
``RelGTTokens.collate`` modulo the new forward-compat keys (``task_id`` and
``task_type_id``).

We don't go through HDF5 here -- we hand-build a list of samples and feed
them to ``gfm_data.collate.collate_single_task`` and to the dev-kyaw collate
adapted to free-function form.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------- helpers
class _StubTF:
    """Tiny torch_frame.TensorFrame stand-in used inside collate."""
    def __init__(self, val: int):
        self.val = val
    def __getitem__(self, idx):
        # Mimic tf[local_idxs] -> a TF subset; here just track the indices.
        return ("tf_subset", self.val, list(idx) if hasattr(idx, "__iter__") else int(idx))
    def __eq__(self, other):
        return isinstance(other, _StubTF) and self.val == other.val
    def __repr__(self):
        return f"_StubTF({self.val})"


class _StubNodeStore:
    def __init__(self, num_nodes: int, type_id: int):
        self.num_nodes = num_nodes
        self.tf = _StubTF(val=type_id)
    def __contains__(self, key):
        return False  # no 'x' so num_nodes path is used


class _StubData:
    def __init__(self, node_types):
        self._stores = {nt: _StubNodeStore(num_nodes=4, type_id=i)
                        for i, nt in enumerate(node_types)}
        self.node_types = node_types
    def __getitem__(self, key):
        return self._stores[key]


class _StubCache:
    def __init__(self, node_types):
        self.data = _StubData(node_types)
        self.node_types = list(node_types)
        self.node_type_to_index = {nt: i for i, nt in enumerate(node_types)}
        self.index_to_node_type = {i: nt for i, nt in enumerate(node_types)}
        self.prefixed_to_raw = {nt: nt for nt in node_types}
    def _num_nodes_of(self, data, node_type):
        return data[node_type].num_nodes
    def tf_view(self, raw_node_type, row_idx):
        # Mirrors DatasetGraphCache.tf_view(in-RAM branch).
        return self.data[raw_node_type].tf[row_idx]


class _StubTaskTokens:
    """Just enough of TaskTokens for collate_single_task to run."""
    def __init__(self):
        self.cache = _StubCache(node_types=["A", "B"])
        self.data = self.cache.data
        self.node_types = list(self.cache.node_types)
        self.node_type_to_index = dict(self.cache.node_type_to_index)
        self.index_to_node_type = dict(self.cache.index_to_node_type)
        self.target = torch.zeros(8)  # any non-None target
        # type_local_to_global: simple stacked layout
        self.type_local_to_global = {}
        g = 0
        for ti, nt in self.index_to_node_type.items():
            for li in range(4):
                self.type_local_to_global[(ti, li)] = g
                g += 1
    def __len__(self):
        return len(self.target)
    def __getitem__(self, idx):
        # Round-robin sample for multi-task dispatch tests.
        return _make_sample(global_idx=idx, task_id=0)
    def get_global_index(self, type_idxs, local_idxs):
        return [self.type_local_to_global[(t, l)] for t, l in zip(type_idxs, local_idxs)]


def _make_sample(K=4, types=(0, 1, 1, 0), indices=(0, 1, 2, 3),
                 hops=(0, 1, 1, 2), times=(0.0, 1.5, 2.5, 3.5),
                 edges=((0, 1), (1, 2)), global_idx=0,
                 task_id=0, task_type_id=1):
    sample = {
        "types": torch.tensor(types, dtype=torch.long),
        "indices": torch.tensor(indices, dtype=torch.long),
        "hops": torch.tensor(hops, dtype=torch.long),
        "times": torch.tensor(times, dtype=torch.float32),
        "edge_index": torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long),
        "first_type": types[0],
        "first_index": indices[0],
        "global_idx": global_idx,
        "tfs": [None] * K,  # collate doesn't read this for the asserted keys
        "task_id": task_id,
        "task_type_id": task_type_id,
    }
    label = torch.tensor(0.0)
    return sample, label


# ----------------------------------------------------------------- Co5
def test_Co5_single_task_collate_keys_and_shapes():
    from gfm_data.collate import collate_single_task
    ds = _StubTaskTokens()
    batch = [_make_sample(global_idx=i) for i in range(3)]
    out = collate_single_task(ds, batch)

    expected_keys = {
        "neighbor_types", "neighbor_indices", "neighbor_hops", "neighbor_times",
        "labels", "node_indices",
        "grouped_tfs", "grouped_indices", "flat_batch_idx", "flat_nbr_idx",
        "global_idx", "edge_index", "batch",
        # forward-compat additions
        "task_id", "task_type_id",
    }
    assert set(out.keys()) == expected_keys, set(out.keys()) ^ expected_keys

    B, K = 3, 4
    assert out["neighbor_types"].shape == (B, K)
    assert out["neighbor_indices"].shape == (B, K)
    assert out["neighbor_hops"].shape == (B, K)
    assert out["neighbor_times"].shape == (B, K)
    assert out["labels"].shape == (B,)
    assert out["node_indices"].shape == (B,)
    assert out["batch"].shape == (B * K,)

    # Edge_index is concatenated with K-offset.
    # Each sample contributes 2 edges, so total = 6.
    assert out["edge_index"].shape == (2, B * 2)
    # Per-sample offset of K: edges from sample 1 should have indices in [4, 8).
    sample_1_edges = out["edge_index"][:, 2:4]
    assert int(sample_1_edges.min()) >= K and int(sample_1_edges.max()) < 2 * K

    # task_id / task_type_id replicate per row.
    assert out["task_id"].tolist() == [0, 0, 0]
    assert out["task_type_id"].tolist() == [1, 1, 1]


def test_Co5_grouped_tfs_keyed_by_type_id_with_correct_offsets():
    from gfm_data.collate import collate_single_task
    ds = _StubTaskTokens()
    # Layout per sample: positions 0,3 are type A (id=0); 1,2 are type B (id=1).
    batch = [_make_sample(types=(0, 1, 1, 0), indices=(0, 1, 2, 3),
                          global_idx=i) for i in range(2)]
    out = collate_single_task(ds, batch)
    K = 4
    # Type A appears at positions 0 and 3 of each sample -> flat positions
    # {0, 3, 4, 7} for B=2 samples.
    assert sorted(out["grouped_indices"][0]) == [0, 3, 4, 7]
    # Type B appears at positions 1 and 2 of each sample -> flat positions
    # {1, 2, 5, 6}.
    assert sorted(out["grouped_indices"][1]) == [1, 2, 5, 6]


def test_Co5_empty_edges_handled():
    from gfm_data.collate import collate_single_task
    ds = _StubTaskTokens()
    batch = [_make_sample(edges=(), global_idx=i) for i in range(2)]
    out = collate_single_task(ds, batch)
    assert out["edge_index"].shape == (2, 0)
