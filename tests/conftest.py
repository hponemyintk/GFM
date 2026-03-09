"""
Shared fixtures for sampling tests.

Mocks ALL heavy dependencies (torch_geometric, sentence_transformers,
relbench) so tests can run fast even on Google Drive filesystems.
"""
import sys
import os
import types

# ── Mock heavy dependencies BEFORE any project imports ──────────

# sentence_transformers
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = None
sys.modules["sentence_transformers"] = _st

# relbench
for mod_name in ("relbench", "relbench.base", "relbench.modeling",
                 "relbench.modeling.graph"):
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["relbench.base"].Dataset = type("Dataset", (), {})
sys.modules["relbench.base"].EntityTask = type("EntityTask", (), {})
sys.modules["relbench.modeling.graph"].get_node_train_table_input = lambda *a, **k: None

# torch_geometric — lightweight mock with real HeteroData-like class
_pyg = types.ModuleType("torch_geometric")
_pyg_data = types.ModuleType("torch_geometric.data")
sys.modules["torch_geometric"] = _pyg
sys.modules["torch_geometric.data"] = _pyg_data


import torch


class _NodeStore:
    """Minimal stand-in for a PyG node/edge store."""
    def __init__(self):
        self._data = {}
        self.num_nodes = 0

    def __setattr__(self, key, val):
        if key.startswith("_") or key == "num_nodes":
            super().__setattr__(key, val)
        else:
            self._data[key] = val

    def __getattr__(self, key):
        if key.startswith("_") or key == "num_nodes":
            return super().__getattribute__(key)
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __contains__(self, key):
        return key in self._data


class MockHeteroData:
    """
    Minimal mock of torch_geometric.data.HeteroData.
    Supports: data["type"], data["src","rel","dst"], .node_types, .edge_types
    """
    def __init__(self):
        self._node_stores = {}
        self._edge_stores = {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if key not in self._edge_stores:
                self._edge_stores[key] = _NodeStore()
            return self._edge_stores[key]
        else:
            if key not in self._node_stores:
                self._node_stores[key] = _NodeStore()
            return self._node_stores[key]

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            self._edge_stores[key] = val
        else:
            self._node_stores[key] = val

    @property
    def node_types(self):
        return list(self._node_stores.keys())

    @property
    def edge_types(self):
        return list(self._edge_stores.keys())

    def to(self, device):
        return self


# Inject the mock
_pyg_data.HeteroData = MockHeteroData
_pyg.data = _pyg_data
# ── End mocks ───────────────────────────────────────────────────

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import build_adjacency_hetero, build_adjacency_csr, init_worker_globals


# ──────────────────────────────────────────────────────────────────
#  Graph fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_hetero():
    """
    Small heterogeneous graph for deterministic tests.

    Nodes:
      user(5)     times=[100, 200, 300, 400, 500]
      product(8)  times=[50, 150, 250, 350, 450, 550, 650, 750]
      category(3) NO time attribute

    Edges (directed, made undirected by build_adjacency):
      user  --buys-->       product   (9 edges)
      product --belongs_to--> category (8 edges)
    """
    data = MockHeteroData()

    data["user"].num_nodes = 5
    data["user"].time = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])

    data["product"].num_nodes = 8
    data["product"].time = torch.tensor(
        [50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0]
    )

    data["category"].num_nodes = 3
    # category intentionally has NO time attribute

    data["user", "buys", "product"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3, 3, 4, 4],
         [0, 1, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )

    data["product", "belongs_to", "category"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7],
         [0, 0, 1, 1, 2, 2, 0, 1]],
        dtype=torch.long,
    )

    return data


@pytest.fixture
def small_adj(small_hetero):
    """Old-format adjacency (dict-of-lists-of-sets) for small_hetero, undirected."""
    return build_adjacency_hetero(small_hetero, undirected=True)


@pytest.fixture
def small_adj_directed(small_hetero):
    """Old-format adjacency, directed."""
    return build_adjacency_hetero(small_hetero, undirected=False)


@pytest.fixture
def hub_hetero():
    """
    Graph with a high-degree hub node for threshold testing.
    A[0] is connected to all 200 B nodes.
    B nodes also form a chain (B[i]->B[i+1]) for 2-hop connectivity.
    """
    data = MockHeteroData()

    num_b = 200
    data["A"].num_nodes = 10
    data["A"].time = torch.arange(10, dtype=torch.float) * 100.0

    data["B"].num_nodes = num_b
    data["B"].time = torch.arange(num_b, dtype=torch.float) * 5.0

    src = torch.zeros(num_b, dtype=torch.long)
    dst = torch.arange(num_b, dtype=torch.long)
    data["A", "connects", "B"].edge_index = torch.stack([src, dst])

    b_src = torch.arange(0, num_b - 1, dtype=torch.long)
    b_dst = torch.arange(1, num_b, dtype=torch.long)
    data["B", "links", "B"].edge_index = torch.stack([b_src, b_dst])

    return data


@pytest.fixture
def isolated_hetero():
    """Graph where node type Z has no edges (completely isolated)."""
    data = MockHeteroData()

    data["X"].num_nodes = 3
    data["X"].time = torch.tensor([100.0, 200.0, 300.0])

    data["Y"].num_nodes = 2
    # Y has no time

    data["Z"].num_nodes = 2
    data["Z"].time = torch.tensor([50.0, 150.0])

    data["X", "rel", "Y"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    # No edges touching Z

    return data


# ──────────────────────────────────────────────────────────────────
#  Helper to set up worker globals for _process_one_seed tests
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def worker_globals_small(small_hetero):
    """Initialize worker globals with CSR adjacency + time arrays for small_hetero."""
    import utils
    csr_adj = build_adjacency_csr(small_hetero, undirected=True)
    all_nodes = []
    for nt in small_hetero.node_types:
        for i in range(small_hetero[nt].num_nodes):
            all_nodes.append((nt, i))
    time_arrays = {}
    for nt in small_hetero.node_types:
        if hasattr(small_hetero[nt], "time"):
            time_arrays[nt] = small_hetero[nt].time.numpy()
    node_types = list(small_hetero.node_types)
    init_worker_globals(csr_adj, all_nodes, node_types=node_types, time_arrays=time_arrays)
    yield
    init_worker_globals(None, None)
    utils.GLOBAL_TIME_ARRAYS = None
    utils.GLOBAL_NODE_TYPES = None
    utils.GLOBAL_TYPE_TO_IDX = None
    utils.GLOBAL_IDX_TO_TYPE = None


@pytest.fixture
def worker_globals_csr(small_hetero):
    """Initialize worker globals with CSR adjacency + time arrays (new path)."""
    import utils
    csr_adj = build_adjacency_csr(small_hetero, undirected=True)
    all_nodes = []
    for nt in small_hetero.node_types:
        for i in range(small_hetero[nt].num_nodes):
            all_nodes.append((nt, i))
    time_arrays = {}
    for nt in small_hetero.node_types:
        if hasattr(small_hetero[nt], "time"):
            time_arrays[nt] = small_hetero[nt].time.numpy()
    node_types = list(small_hetero.node_types)
    init_worker_globals(csr_adj, all_nodes, node_types=node_types, time_arrays=time_arrays)
    yield
    init_worker_globals(None, None)
    utils.GLOBAL_TIME_ARRAYS = None
    utils.GLOBAL_NODE_TYPES = None
    utils.GLOBAL_TYPE_TO_IDX = None
    utils.GLOBAL_IDX_TO_TYPE = None
