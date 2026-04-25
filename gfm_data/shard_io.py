"""Memmap shard reader/writer for precomputed sample tokens.

Replaces the single-HDF5-per-task layout used by ``utils.RelGTTokens`` with
a directory of fixed-size memmap shards. Two reasons:

1. **Memory.** HDF5 chunked I/O ends up with the whole chunk in RAM during
   reads under DataLoader multi-worker; memmap slices are paged in by the
   kernel only as needed.
2. **Concurrency.** HDF5 has a process-wide GIL/lock for parallel reads;
   ``np.memmap`` does not, so DataLoader workers don't contend.

Layout under ``<root>/<split>/``:

    meta.json                         # {K, total_samples, shard_size, dtype map}
    shard_0000.types.i16              # [shard_size, K]
    shard_0000.indices.i32            # [shard_size, K]
    shard_0000.hops.i8                # [shard_size, K]
    shard_0000.times.f32              # [shard_size, K]
    shard_0000.edges.i16              # [2, total_edges_in_shard]
    shard_0000.edges_offsets.u64      # [shard_size + 1]
    shard_0001.*

The last shard may be partial (size <= shard_size). Edge arrays are
variable-length per sample, so the per-shard offsets array is needed.

Equivalence with the HDF5 layout is bit-exact for `types`, `indices`,
`hops`, `times`, and the concatenated edge stream — see test Sh1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


_DTYPES = {
    "types": np.int16,
    "indices": np.int32,
    "hops": np.int8,
    "times": np.float32,
    "edges": np.int16,
    "edges_offsets": np.uint64,
}


@dataclass
class ShardMeta:
    K: int
    total_samples: int
    shard_size: int
    num_shards: int

    def to_json(self) -> dict:
        return {
            "K": int(self.K),
            "total_samples": int(self.total_samples),
            "shard_size": int(self.shard_size),
            "num_shards": int(self.num_shards),
            "dtypes": {k: np.dtype(v).str for k, v in _DTYPES.items()},
            "layout_version": 1,
        }

    @staticmethod
    def from_json(d: dict) -> "ShardMeta":
        return ShardMeta(
            K=int(d["K"]),
            total_samples=int(d["total_samples"]),
            shard_size=int(d["shard_size"]),
            num_shards=int(d["num_shards"]),
        )


def _shard_path(root: str, shard_idx: int, name: str) -> str:
    return os.path.join(root, f"shard_{shard_idx:04d}.{name}.{np.dtype(_DTYPES[name]).str.lstrip('<>=|')}")


# ---------------------------------------------------------------- writer
class ShardWriter:
    """Write shards one at a time. Single-process, fail-fast on duplicate writes."""

    def __init__(self, root: str, K: int, total_samples: int, shard_size: int = 50_000):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.K = K
        self.total_samples = total_samples
        self.shard_size = shard_size
        self.num_shards = (total_samples + shard_size - 1) // shard_size
        self._written: List[bool] = [False] * self.num_shards

    def shard_range(self, shard_idx: int) -> Tuple[int, int]:
        start = shard_idx * self.shard_size
        end = min(start + self.shard_size, self.total_samples)
        return start, end

    def write_shard(
        self,
        shard_idx: int,
        types: np.ndarray,
        indices: np.ndarray,
        hops: np.ndarray,
        times: np.ndarray,
        edges_per_sample: List[np.ndarray],
    ):
        assert 0 <= shard_idx < self.num_shards
        assert not self._written[shard_idx], f"shard {shard_idx} already written"
        n_samples = types.shape[0]
        assert types.shape == (n_samples, self.K)
        assert indices.shape == (n_samples, self.K)
        assert hops.shape == (n_samples, self.K)
        assert times.shape == (n_samples, self.K)
        assert len(edges_per_sample) == n_samples

        types.astype(_DTYPES["types"], copy=False).tofile(_shard_path(self.root, shard_idx, "types"))
        indices.astype(_DTYPES["indices"], copy=False).tofile(_shard_path(self.root, shard_idx, "indices"))
        hops.astype(_DTYPES["hops"], copy=False).tofile(_shard_path(self.root, shard_idx, "hops"))
        times.astype(_DTYPES["times"], copy=False).tofile(_shard_path(self.root, shard_idx, "times"))

        offsets = np.zeros(n_samples + 1, dtype=_DTYPES["edges_offsets"])
        for i, e in enumerate(edges_per_sample):
            n_e = 0 if e is None or e.size == 0 else e.shape[1]
            offsets[i + 1] = offsets[i] + n_e
        total_edges = int(offsets[-1])

        if total_edges > 0:
            packed = np.empty((2, total_edges), dtype=_DTYPES["edges"])
            cur = 0
            for e in edges_per_sample:
                if e is None or e.size == 0:
                    continue
                n_e = e.shape[1]
                packed[:, cur : cur + n_e] = e.astype(_DTYPES["edges"], copy=False)
                cur += n_e
            packed.tofile(_shard_path(self.root, shard_idx, "edges"))
        else:
            np.empty((2, 0), dtype=_DTYPES["edges"]).tofile(
                _shard_path(self.root, shard_idx, "edges")
            )

        offsets.tofile(_shard_path(self.root, shard_idx, "edges_offsets"))
        self._written[shard_idx] = True

    def finalize(self):
        assert all(self._written), \
            f"missing shards: {[i for i, w in enumerate(self._written) if not w]}"
        meta = ShardMeta(
            K=self.K, total_samples=self.total_samples,
            shard_size=self.shard_size, num_shards=self.num_shards,
        )
        with open(os.path.join(self.root, "meta.json"), "w") as f:
            json.dump(meta.to_json(), f, indent=2)


# ---------------------------------------------------------------- reader
class ShardReader:
    """Memmap-based random-access reader.

    Memmaps are opened lazily per shard the first time any sample in that
    shard is requested. Closed memmaps stay alive for the lifetime of the
    reader; the OS handles paging.
    """

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "meta.json"), "r") as f:
            self.meta = ShardMeta.from_json(json.load(f))
        # Lazy memmap caches.
        self._mm: dict[Tuple[int, str], np.memmap] = {}

    def __len__(self) -> int:
        return self.meta.total_samples

    def _open(self, shard_idx: int, name: str) -> np.memmap:
        key = (shard_idx, name)
        if key in self._mm:
            return self._mm[key]
        path = _shard_path(self.root, shard_idx, name)
        # Compute shape based on shard size + K.
        K = self.meta.K
        n_in_shard = self._n_in_shard(shard_idx)
        if name == "types":
            shape = (n_in_shard, K); dtype = _DTYPES["types"]
        elif name == "indices":
            shape = (n_in_shard, K); dtype = _DTYPES["indices"]
        elif name == "hops":
            shape = (n_in_shard, K); dtype = _DTYPES["hops"]
        elif name == "times":
            shape = (n_in_shard, K); dtype = _DTYPES["times"]
        elif name == "edges_offsets":
            shape = (n_in_shard + 1,); dtype = _DTYPES["edges_offsets"]
        elif name == "edges":
            # Variable-size: read length from the corresponding offsets.
            off = self._open(shard_idx, "edges_offsets")
            total_edges = int(off[-1])
            shape = (2, total_edges) if total_edges > 0 else (2, 0)
            dtype = _DTYPES["edges"]
        else:
            raise KeyError(name)
        mm = np.memmap(path, mode="r", shape=shape, dtype=dtype) if (
            shape[-1] > 0 if len(shape) == 1 else (shape[0] > 0 if name == "edges" and shape[1] == 0 else True)
        ) else np.empty(shape, dtype=dtype)
        # The conditional above degenerates to: open memmap when there's any
        # bytes to read; for the edge file with zero edges, fall back to an
        # empty in-memory array.
        if name == "edges" and shape[1] == 0:
            mm = np.empty(shape, dtype=dtype)
        self._mm[key] = mm
        return mm

    def _n_in_shard(self, shard_idx: int) -> int:
        start = shard_idx * self.meta.shard_size
        end = min(start + self.meta.shard_size, self.meta.total_samples)
        return end - start

    def _idx_to_shard(self, idx: int) -> Tuple[int, int]:
        shard_idx = idx // self.meta.shard_size
        local_idx = idx - shard_idx * self.meta.shard_size
        return shard_idx, local_idx

    def read(self, idx: int) -> dict:
        """Return a dict with keys types/indices/hops/times/edge_index."""
        assert 0 <= idx < self.meta.total_samples, idx
        si, li = self._idx_to_shard(idx)
        out = {
            "types": np.array(self._open(si, "types")[li]),  # copy out of memmap
            "indices": np.array(self._open(si, "indices")[li]),
            "hops": np.array(self._open(si, "hops")[li]),
            "times": np.array(self._open(si, "times")[li]),
        }
        offsets = self._open(si, "edges_offsets")
        start = int(offsets[li]); end = int(offsets[li + 1])
        if end == start:
            out["edge_index"] = np.zeros((2, 0), dtype=_DTYPES["edges"])
        else:
            out["edge_index"] = np.array(self._open(si, "edges")[:, start:end])
        return out
