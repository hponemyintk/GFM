"""CSR-backed adjacency cache.

Replaces the ``dict[node_type] -> list[set[(dst_type, dst_idx)]]`` adjacency
built by ``utils.build_adjacency_hetero`` with a CSR layout that uses
~12 bytes/edge instead of ~80 bytes/edge.

The public surface is intentionally small: callers ask for ``neighbors_set(...)``
which returns a Python ``set`` of ``(dst_type, dst_idx)`` pairs identical to the
set ``utils.build_adjacency_hetero`` would have produced. This preserves the
Python-set iteration order that dev-kyaw's sampler relies on for
``random.sample`` / ``random.choices`` reproducibility.

Dataset-name prefixing of node types (``"rel-f1::drivers"``) is supported so
multiple datasets can share the same encoder / collate / model without
namespace collisions across PR3+ multi-dataset training. In PR1 prefixes
are off by default so single-task runs match dev-kyaw exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


class CSRBlock:
    """CSR storage for the directed adjacency of a single source node type.

    Stores **all outgoing neighbors regardless of destination type** in a
    single packed array, matching the semantics of dev-kyaw's
    ``adjacency[src_type][s]`` which mixes destination types in one set.

    Attributes
    ----------
    indptr : int64 array of shape ``[N_src + 1]``
    nbr_type_id : int16 array of shape ``[E]`` -- destination type ids
    nbr_idx : int32 array of shape ``[E]`` -- destination row indices
    """

    __slots__ = ("indptr", "nbr_type_id", "nbr_idx")

    def __init__(
        self,
        indptr: np.ndarray,
        nbr_type_id: np.ndarray,
        nbr_idx: np.ndarray,
    ):
        assert indptr.dtype == np.int64
        assert nbr_type_id.dtype == np.int16
        assert nbr_idx.dtype == np.int32
        assert indptr.ndim == 1
        assert nbr_type_id.ndim == 1
        assert nbr_idx.ndim == 1
        assert nbr_type_id.shape[0] == nbr_idx.shape[0]
        assert indptr[-1] == nbr_type_id.shape[0]
        self.indptr = indptr
        self.nbr_type_id = nbr_type_id
        self.nbr_idx = nbr_idx

    @property
    def num_src(self) -> int:
        return int(self.indptr.shape[0]) - 1

    @property
    def num_edges(self) -> int:
        return int(self.indptr[-1])


class DatasetGraphCache:
    """In-memory CSR adjacency for a single dataset.

    Parameters
    ----------
    data : HeteroData
        The materialized RelBench graph (output of ``make_pkey_fkey_graph``).
    undirected : bool, default ``True``
        Whether to add the reverse direction for every edge. Matches
        ``utils.build_adjacency_hetero`` semantics.
    name_prefix : Optional[str], default ``None``
        If provided, all node-type names exposed by this cache are prefixed
        with ``f"{name_prefix}::"``. Off in PR1 (single-task) so dev-kyaw
        equivalence holds. Used in PR3 to namespace types across datasets.

    Notes
    -----
    PR1 builds the CSR fresh from ``data`` on construction. Disk caching to
    ``<cache>/<dataset>/adj.npz`` lands in PR2 alongside the TF memmap store.
    """

    def __init__(
        self,
        data: HeteroData,
        undirected: bool = True,
        name_prefix: Optional[str] = None,
    ):
        self.data = data
        self.undirected = undirected
        self.name_prefix = name_prefix or ""

        # Stable type ordering matches data.node_types (consistent with
        # main_node_ddp.py:228 and utils.RelGTTokens._create_global_mappings).
        raw_types: List[str] = list(data.node_types)
        self.node_types: List[str] = [self._with_prefix(t) for t in raw_types]
        self.raw_to_prefixed: Dict[str, str] = dict(zip(raw_types, self.node_types))
        self.prefixed_to_raw: Dict[str, str] = dict(zip(self.node_types, raw_types))
        self.node_type_to_index: Dict[str, int] = {
            t: i for i, t in enumerate(self.node_types)
        }
        self.index_to_node_type: Dict[int, str] = {
            i: t for i, t in enumerate(self.node_types)
        }

        # Per-source-type CSR blocks. Key is the **prefixed** type name.
        self.csr: Dict[str, CSRBlock] = self._build_csr(data, raw_types, undirected)

        # All-nodes fallback list (matches utils.local_nodes_hetero
        # ``GLOBAL_ALL_NODES``). Stored as a Python list of (prefixed_type, idx)
        # for direct use with random.sample / random.choices in the sampler.
        self.all_nodes: List[Tuple[str, int]] = []
        for nt in self.node_types:
            raw = self.prefixed_to_raw[nt]
            n = self._num_nodes_of(data, raw)
            self.all_nodes.extend((nt, i) for i in range(n))

    # ------------------------------------------------------------------ build
    def _with_prefix(self, t: str) -> str:
        return f"{self.name_prefix}::{t}" if self.name_prefix else t

    @staticmethod
    def _num_nodes_of(data: HeteroData, node_type: str) -> int:
        if "x" in data[node_type]:
            return int(data[node_type]["x"].size(0))
        return int(data[node_type].num_nodes)

    def _build_csr(
        self,
        data: HeteroData,
        raw_types: List[str],
        undirected: bool,
    ) -> Dict[str, CSRBlock]:
        # Per-(src_type) bucket: list of (src_idx, dst_type_id, dst_idx).
        raw_type_to_id = {t: i for i, t in enumerate(raw_types)}
        n_per_type = {t: self._num_nodes_of(data, t) for t in raw_types}

        # Count edges per source type so we can preallocate.
        per_type_edge_count: Dict[str, int] = {t: 0 for t in raw_types}
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            if "edge_index" not in data[edge_type]:
                continue
            ei = data[edge_type].edge_index
            n_e = int(ei.shape[1])
            per_type_edge_count[src_type] += n_e
            if undirected:
                per_type_edge_count[dst_type] += n_e

        # Allocate per-type "src_idx, dst_type_id, dst_idx" buffers.
        srcs: Dict[str, np.ndarray] = {}
        dst_types: Dict[str, np.ndarray] = {}
        dst_idxs: Dict[str, np.ndarray] = {}
        cursors: Dict[str, int] = {}
        for t in raw_types:
            e = per_type_edge_count[t]
            srcs[t] = np.empty(e, dtype=np.int64)
            dst_types[t] = np.empty(e, dtype=np.int16)
            dst_idxs[t] = np.empty(e, dtype=np.int32)
            cursors[t] = 0

        # Fill buffers.
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            if "edge_index" not in data[edge_type]:
                continue
            ei = data[edge_type].edge_index
            if isinstance(ei, torch.Tensor):
                ei = ei.cpu().numpy()
            s_arr = ei[0].astype(np.int64, copy=False)
            d_arr = ei[1].astype(np.int32, copy=False)
            ne = s_arr.shape[0]
            dst_id = np.int16(raw_type_to_id[dst_type])
            src_id = np.int16(raw_type_to_id[src_type])

            c = cursors[src_type]
            srcs[src_type][c : c + ne] = s_arr
            dst_types[src_type][c : c + ne] = dst_id
            dst_idxs[src_type][c : c + ne] = d_arr
            cursors[src_type] = c + ne

            if undirected:
                c = cursors[dst_type]
                srcs[dst_type][c : c + ne] = d_arr.astype(np.int64, copy=False)
                dst_types[dst_type][c : c + ne] = src_id
                dst_idxs[dst_type][c : c + ne] = s_arr.astype(np.int32, copy=False)
                cursors[dst_type] = c + ne

        # Now build CSR per type by sorting on src_idx.
        out: Dict[str, CSRBlock] = {}
        for t in raw_types:
            n = n_per_type[t]
            s = srcs[t]
            dt = dst_types[t]
            di = dst_idxs[t]

            if s.shape[0] == 0:
                indptr = np.zeros(n + 1, dtype=np.int64)
                out[self._with_prefix(t)] = CSRBlock(
                    indptr=indptr,
                    nbr_type_id=np.empty(0, dtype=np.int16),
                    nbr_idx=np.empty(0, dtype=np.int32),
                )
                continue

            order = np.argsort(s, kind="stable")
            s_sorted = s[order]
            dt_sorted = dt[order]
            di_sorted = di[order]

            # Build indptr via bincount over src ids.
            counts = np.bincount(s_sorted, minlength=n).astype(np.int64)
            indptr = np.empty(n + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            assert indptr[-1] == s_sorted.shape[0]

            out[self._with_prefix(t)] = CSRBlock(
                indptr=indptr,
                nbr_type_id=dt_sorted,
                nbr_idx=di_sorted,
            )
        return out

    # ----------------------------------------------------------------- query
    def neighbors_set(self, src_type: str, src_idx: int) -> Set[Tuple[str, int]]:
        """Return ``set`` of ``(dst_prefixed_type, dst_idx)`` neighbors.

        Equivalent to ``adjacency[src_type][src_idx]`` from
        ``utils.build_adjacency_hetero`` for the same ``undirected`` setting,
        with both keys living in the prefixed-type namespace.

        The returned set has the SAME elements as dev-kyaw's set; CPython's
        set iteration order is hash-determined and therefore identical for
        identical-element sets across the two implementations, which is what
        makes ``random.sample`` reproducible across the refactor.
        """
        block = self.csr[src_type]
        start = int(block.indptr[src_idx])
        end = int(block.indptr[src_idx + 1])
        if end == start:
            return set()
        nt_ids = block.nbr_type_id[start:end]
        nidx = block.nbr_idx[start:end]
        out: Set[Tuple[str, int]] = set()
        for k in range(end - start):
            out.add((self.index_to_node_type[int(nt_ids[k])], int(nidx[k])))
        return out

    # -------------------------------------------------------- diagnostic API
    def num_edges_total(self) -> int:
        return sum(b.num_edges for b in self.csr.values())

    def num_nodes_total(self) -> int:
        return sum(b.num_src for b in self.csr.values())
