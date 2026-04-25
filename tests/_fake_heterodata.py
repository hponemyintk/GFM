"""Tiny duck-typed stand-in for torch_geometric.HeteroData.

Used by the CSR adjacency / sampler unit tests so they don't need the real
torch_geometric available (the project conftest mocks it for fast tests).

Supports just enough of the HeteroData surface to drive
``DatasetGraphCache`` and dev-kyaw's ``build_adjacency_hetero`` /
``gather_1_and_2_hop_with_seed_time`` for equivalence checks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class FakeNodeStore:
    def __init__(self, num_nodes: int, time: Optional[torch.Tensor] = None,
                 x: Optional[torch.Tensor] = None):
        self._fields: Dict[str, object] = {"num_nodes": num_nodes}
        self.num_nodes = num_nodes
        if time is not None:
            assert time.shape == (num_nodes,), \
                f"time must be shape ({num_nodes},), got {tuple(time.shape)}"
            self.time = time
            self._fields["time"] = time
        if x is not None:
            self.x = x
            self._fields["x"] = x

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __getitem__(self, key: str):
        return self._fields[key]


class FakeEdgeStore:
    def __init__(self, edge_index: torch.Tensor):
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        self.edge_index = edge_index

    def __contains__(self, key: str) -> bool:
        return key == "edge_index"


class FakeHeteroData:
    """Behaves like HeteroData for the subset of API we exercise."""

    def __init__(
        self,
        node_stores: Dict[str, FakeNodeStore],
        edge_stores: Dict[Tuple[str, str, str], FakeEdgeStore],
    ):
        self._node_stores = node_stores
        self._edge_stores = edge_stores

    @property
    def node_types(self):
        return list(self._node_stores.keys())

    @property
    def edge_types(self):
        return list(self._edge_stores.keys())

    @property
    def num_nodes(self):
        return sum(s.num_nodes for s in self._node_stores.values())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._node_stores[key]
        return self._edge_stores[key]

    def to(self, device):
        return self


def make_toy_graph() -> FakeHeteroData:
    """Two node types A (4 nodes), B (3 nodes); A-B edges; B has timestamps.

    A0 --- B0 (t=10)
    A0 --- B1 (t=20)
    A1 --- B0 (t=10)
    A1 --- B2 (t=30)
    A2 --- B1 (t=20)
    A3  (no edges)

    Used by C1, S1 etc.
    """
    a = FakeNodeStore(num_nodes=4)
    b_time = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    b = FakeNodeStore(num_nodes=3, time=b_time)
    edges_ab = torch.tensor(
        [[0, 0, 1, 1, 2],
         [0, 1, 0, 2, 1]], dtype=torch.long,
    )
    return FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(edges_ab)},
    )


def make_dense_graph(seed: int = 0, n_a: int = 50, n_b: int = 200) -> FakeHeteroData:
    """A larger graph used to hit the threshold caps and the >K / =K / <K cases.

    Returns A-B with random edges and B-time uniform in [0, 100].
    """
    g = torch.Generator().manual_seed(seed)
    n_edges = 600
    src = torch.randint(0, n_a, (n_edges,), generator=g)
    dst = torch.randint(0, n_b, (n_edges,), generator=g)
    a = FakeNodeStore(num_nodes=n_a)
    b_time = torch.rand(n_b, generator=g) * 100.0
    b = FakeNodeStore(num_nodes=n_b, time=b_time)
    return FakeHeteroData(
        node_stores={"A": a, "B": b},
        edge_stores={("A", "to", "B"): FakeEdgeStore(torch.stack([src, dst]))},
    )
