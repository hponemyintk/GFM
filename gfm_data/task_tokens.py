"""Per-(dataset, task, split) token dataset.

Replaces ``utils.RelGTTokens``. The main differences are:

* adjacency lookups are CSR-backed via ``DatasetGraphCache`` (passed in,
  not constructed here, so multiple tasks of the same dataset share one
  cache);
* sampling code lives in ``data.sampler`` rather than module-level
  globals + ``multiprocessing.Pool``;
* ``__getitem__`` adds two forward-compat keys ``task_id`` and
  ``task_type`` (single-valued in PR1, used by the multi-task path in PR3);
* HDF5 precompute is preserved bit-for-bit so existing ``--precompute``
  caches continue to work — the move to memmap shards is a PR2 change.

Single-task ``__getitem__`` output is otherwise identical to dev-kyaw.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from relbench.modeling.graph import get_node_train_table_input

from gfm_data.graph_cache import DatasetGraphCache
from gfm_data.sampler import sample_local_subgraph

# Type ids: 0 = regression, 1 = binary classification.
# Multi-label classification is out of scope for v0 multi-task; we map it to
# a sentinel here so the existing single-task multi-label path stays runnable.
TASK_TYPE_REGRESSION = 0
TASK_TYPE_BINARY = 1
TASK_TYPE_MULTILABEL = 2


class TaskTokens(Dataset):
    """Single-task token dataset reading from a shared ``DatasetGraphCache``.

    Parameters
    ----------
    cache : DatasetGraphCache
        Shared per-dataset CSR adjacency. Built once by the caller.
    task : relbench.base.EntityTask
    K : int
        Number of tokens per sample (seed + K-1 neighbors).
    split : {"train", "val", "test"}
    mode : {"hdf5", "streaming", "precomputed_shards"}
        Sample materialization strategy.

        * ``hdf5`` (default): legacy dev-kyaw layout, single HDF5 file per
          split. Built in-process if ``precompute`` is True.
        * ``streaming``: no precompute. ``__getitem__`` calls the sampler
          directly each time. Best for small datasets / iteration / laptop.
        * ``precomputed_shards``: read from memmap shards under
          ``shards_dir`` produced by ``tools/precompute_shards.py``. Best
          for production / 8xA100 box.
    precompute : bool
        Only relevant for ``mode="hdf5"``. Build the HDF5 if missing.
    precomputed_dir : str
        Cache directory for the HDF5 file (mode="hdf5").
    shards_dir : Optional[str]
        Directory of memmap shards (mode="precomputed_shards").
    train_stage : {"finetune"}
    task_id : int, default 0
        Forward-compat: which (dataset, task) this is in a multi-task run.
    task_type_id : Optional[int]
        Forward-compat: ``TASK_TYPE_*`` for the per-row dispatch in PR3.
        If ``None``, derived from ``task.task_type``.
    """

    def __init__(
        self,
        cache: DatasetGraphCache,
        task,
        K: int,
        split: str = "train",
        mode: str = "hdf5",
        precompute: bool = True,
        precomputed_dir: Optional[str] = None,
        shards_dir: Optional[str] = None,
        train_stage: str = "finetune",
        task_id: int = 0,
        task_type_id: Optional[int] = None,
    ):
        super().__init__()
        if mode not in ("hdf5", "streaming", "precomputed_shards"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.cache = cache
        self.data = cache.data  # exposed for back-compat with main_node_ddp.py
        self.task = task
        self.split = split
        self.K = K
        self.mode = mode
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir
        self.shards_dir = shards_dir
        self.train_stage = train_stage
        self.task_id = int(task_id)
        self.task_type_id = (
            int(task_type_id)
            if task_type_id is not None
            else _derive_task_type_id(task)
        )

        self.table = task.get_table(split=split)
        self.table_input = get_node_train_table_input(self.table, task)

        # The seed entity type, in raw (un-prefixed) form. Map to prefixed for
        # downstream use; cache.raw_to_prefixed is the source of truth.
        raw_seed_type, seed_idxs = self.table_input.nodes
        self.raw_node_type = raw_seed_type
        self.node_type = cache.raw_to_prefixed[raw_seed_type]
        self.node_idxs = seed_idxs
        self.target = self.table_input.target if self.table_input.target is not None else None
        self.time = getattr(self.table_input, "time", None)
        self.transform = getattr(self.table_input, "transform", None)

        # Inherit type tables from the cache (kept on the dataset object so
        # main_node_ddp.py:228-230 still works without a cache reference).
        self.node_types: List[str] = list(cache.node_types)
        self.node_type_to_index: Dict[str, int] = dict(cache.node_type_to_index)
        self.index_to_node_type: Dict[int, str] = dict(cache.index_to_node_type)
        self.max_neighbor_hop = 2 + 1  # 0,1,2 + fallback (3)

        self._create_global_mappings()

        # Per-mode setup.
        self.precomputed_path = None
        self._shard_reader = None
        if self.mode == "hdf5":
            self.precomputed_path = self._construct_precomputed_path() if precompute else None
            if self.precompute:
                if os.path.exists(self.precomputed_path):
                    print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
                else:
                    print(f"[{self.split}] Precomputing neighbor sampling (K={self.K})...")
                    self._precompute_sampling()
        elif self.mode == "precomputed_shards":
            from gfm_data.shard_io import ShardReader
            assert shards_dir is not None, \
                "mode='precomputed_shards' requires shards_dir"
            shard_split_dir = os.path.join(shards_dir, str(K), self.split)
            assert os.path.isdir(shard_split_dir), \
                f"shards not found at {shard_split_dir}; run tools/precompute_shards.py"
            self._shard_reader = ShardReader(shard_split_dir)
            assert self._shard_reader.meta.K == K
            assert len(self._shard_reader) == len(self.node_idxs), (
                f"shards has {len(self._shard_reader)} samples, "
                f"task split has {len(self.node_idxs)}"
            )
        # streaming: no setup needed; sampler runs in __getitem__

    # ------------------------------------------------------------ id maps
    def _create_global_mappings(self):
        """Stable ``(type_idx, local_idx) -> global_idx`` map.

        Used by main_node_ddp.py via ``data["train"].data.num_nodes`` and the
        seed-node global index lookup in collate. Identical to dev-kyaw's
        ``RelGTTokens._create_global_mappings``.
        """
        self.type_local_to_global: Dict[Tuple[int, int], int] = {}
        self.global_to_type_local: Dict[int, Tuple[int, int]] = {}
        g = 0
        for type_idx, prefixed_type in self.index_to_node_type.items():
            raw_type = self.cache.prefixed_to_raw[prefixed_type]
            n = self.cache._num_nodes_of(self.cache.data, raw_type)
            for local_idx in range(n):
                self.type_local_to_global[(type_idx, local_idx)] = g
                self.global_to_type_local[g] = (type_idx, local_idx)
                g += 1

    def get_global_index(self, type_idxs: List[int], local_idxs: List[int]) -> List[int]:
        return [self.type_local_to_global[(t, l)] for t, l in zip(type_idxs, local_idxs)]

    # --------------------------------------------------------- HDF5 cache
    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("must provide a 'precomputed_dir' to store expansions.")
        path = os.path.join(self.precomputed_dir, str(self.K), f"{self.split}.h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def __len__(self):
        return len(self.node_idxs)

    def _create_datasets(self, h5file: h5py.File, total_samples: int) -> dict:
        chunk = min(total_samples, 10000)
        return {
            "types": h5file.create_dataset(
                "types", shape=(total_samples, self.K), dtype="int16",
                chunks=(chunk, self.K),
            ),
            "indices": h5file.create_dataset(
                "indices", shape=(total_samples, self.K), dtype="int32",
                chunks=(chunk, self.K),
            ),
            "hops": h5file.create_dataset(
                "hops", shape=(total_samples, self.K), dtype="int8",
                chunks=(chunk, self.K),
            ),
            "times": h5file.create_dataset(
                "times", shape=(total_samples, self.K), dtype="float32",
                chunks=(chunk, self.K),
            ),
        }

    def _precompute_sampling(self):
        """Sequential CSR-based precompute. Mirrors dev-kyaw's HDF5 layout.

        We drop dev-kyaw's ``multiprocessing.Pool`` here because (a) it
        collides with DataLoader workers, and (b) PR2 moves precompute into
        an offline tool. The CSR sampler is fast enough that single-process
        is acceptable for the rel-f1 sizes we test on the laptop.
        """
        total = len(self.node_idxs)
        chunk_size = 10000

        with h5py.File(self.precomputed_path, "w") as hf:
            datasets = self._create_datasets(hf, total)
            adjacency_all: List[Optional[np.ndarray]] = [None] * total

            with tqdm(total=total, desc=f"Precomputing '{self.split}'") as pbar:
                for start_idx in range(0, total, chunk_size):
                    end_idx = min(start_idx + chunk_size, total)
                    size_chunk = end_idx - start_idx

                    chunk_node_idxs = self.node_idxs[start_idx:end_idx]
                    chunk_times = self.time[start_idx:end_idx] if self.time is not None else None

                    c_types = np.zeros((size_chunk, self.K), dtype=np.int16)
                    c_indices = np.zeros((size_chunk, self.K), dtype=np.int32)
                    c_hops = np.zeros((size_chunk, self.K), dtype=np.int8)
                    c_times = np.zeros((size_chunk, self.K), dtype=np.float32)

                    for i, node_idx_t in enumerate(chunk_node_idxs):
                        node_idx = int(node_idx_t.item() if isinstance(node_idx_t, Tensor) else node_idx_t)
                        seed_t = float(chunk_times[i].item()) if chunk_times is not None else 0.0
                        seed_val = hash((self.node_type, node_idx, seed_t, self.K)) & 0xFFFFFFFF

                        final_nodes, edge_index = sample_local_subgraph(
                            self.cache, self.K, self.node_type,
                            node_idx, seed_t, seed_val,
                        )
                        for j, (t_str, nbr_loc_idx, hop, t_val, _c1) in enumerate(final_nodes):
                            c_types[i, j] = self.node_type_to_index[t_str]
                            c_indices[i, j] = nbr_loc_idx
                            c_hops[i, j] = hop
                            c_times[i, j] = t_val
                        adjacency_all[start_idx + i] = edge_index

                    datasets["types"][start_idx:end_idx] = c_types
                    datasets["indices"][start_idx:end_idx] = c_indices
                    datasets["hops"][start_idx:end_idx] = c_hops
                    datasets["times"][start_idx:end_idx] = c_times
                    pbar.update(size_chunk)
                    gc.collect()

            offsets = np.zeros(total + 1, dtype=np.uint64)
            for i in range(total):
                e = adjacency_all[i].shape[1] if adjacency_all[i] is not None else 0
                offsets[i + 1] = offsets[i] + e
            total_edges = int(offsets[-1])
            edges_dset = hf.create_dataset("edges", shape=(2, total_edges), dtype="int16")
            for i in range(total):
                e_arr = adjacency_all[i]
                start = int(offsets[i])
                end_ = int(offsets[i + 1])
                if e_arr is not None and e_arr.size > 0:
                    edges_dset[:, start:end_] = e_arr
            hf.create_dataset("edges_offsets", data=offsets)

    # ----------------------------------------------------- per-mode samplers
    def _sample_from_hdf5(self, idx: int):
        with h5py.File(self.precomputed_path, "r") as hf:
            sample = {
                "types": torch.from_numpy(hf["types"][idx]).long(),
                "indices": torch.from_numpy(hf["indices"][idx]).long(),
                "hops": torch.from_numpy(hf["hops"][idx]).long(),
                "times": torch.from_numpy(hf["times"][idx]),
            }
            offsets = hf["edges_offsets"]
            edges_dset = hf["edges"]
            start = offsets[idx]
            end_ = offsets[idx + 1]
            if start == end_:
                eidx = torch.zeros((2, 0), dtype=torch.long)
            else:
                eidx = torch.from_numpy(edges_dset[:, start:end_]).long()
            sample["edge_index"] = eidx
        return sample

    def _sample_from_shards(self, idx: int):
        s = self._shard_reader.read(idx)
        return {
            "types": torch.from_numpy(s["types"].astype(np.int64, copy=False)),
            "indices": torch.from_numpy(s["indices"].astype(np.int64, copy=False)),
            "hops": torch.from_numpy(s["hops"].astype(np.int64, copy=False)),
            "times": torch.from_numpy(s["times"]),
            "edge_index": torch.from_numpy(s["edge_index"].astype(np.int64, copy=False)),
        }

    def _sample_streaming(self, idx: int):
        """Run the sampler on demand. No precompute, no caching."""
        from gfm_data.sampler import sample_local_subgraph
        node_idx_t = self.node_idxs[idx]
        node_idx = int(node_idx_t.item() if isinstance(node_idx_t, Tensor) else node_idx_t)
        seed_t = float(self.time[idx].item()) if self.time is not None else 0.0
        seed_val = hash((self.node_type, node_idx, seed_t, self.K)) & 0xFFFFFFFF
        final_nodes, edge_index = sample_local_subgraph(
            self.cache, self.K, self.node_type,
            node_idx, seed_t, seed_val,
        )
        K = self.K
        types_arr = np.zeros(K, dtype=np.int64)
        idx_arr = np.zeros(K, dtype=np.int64)
        hops_arr = np.zeros(K, dtype=np.int64)
        times_arr = np.zeros(K, dtype=np.float32)
        for j, (t_str, nbr_loc, hop, t_val, _c) in enumerate(final_nodes):
            types_arr[j] = self.node_type_to_index[t_str]
            idx_arr[j] = nbr_loc
            hops_arr[j] = hop
            times_arr[j] = t_val
        return {
            "types": torch.from_numpy(types_arr),
            "indices": torch.from_numpy(idx_arr),
            "hops": torch.from_numpy(hops_arr),
            "times": torch.from_numpy(times_arr),
            "edge_index": torch.from_numpy(edge_index.astype(np.int64, copy=False)),
        }

    # ------------------------------------------------------------- access
    def __getitem__(self, idx: int):
        if self.mode == "streaming":
            sample = self._sample_streaming(idx)
        elif self.mode == "precomputed_shards":
            sample = self._sample_from_shards(idx)
        else:  # hdf5
            sample = self._sample_from_hdf5(idx)

        label = self.target[idx] if self.target is not None else None

        sample["first_type"] = sample["types"][0].item()
        sample["first_index"] = sample["indices"][0].item()

        # Per-token TFs come from cache.tf_view, which dispatches to either
        # the in-RAM ``data[type].tf`` (default) or a memmap-backed
        # TFStoreReader (when ``tf_store_root`` was passed to the cache).
        sample["tfs"] = [
            self.cache.tf_view(
                self.cache.prefixed_to_raw[self.index_to_node_type[t.item()]],
                int(i.item()),
            )
            for t, i in zip(sample["types"], sample["indices"])
        ]
        sample["global_idx"] = idx

        # Forward-compat fields used by PR3's multi-task collate. Single value
        # here; in PR3 these may differ across rows of a mixed batch.
        sample["task_id"] = self.task_id
        sample["task_type_id"] = self.task_type_id
        return sample, label


def _derive_task_type_id(task) -> int:
    from relbench.base import TaskType
    if task.task_type == TaskType.REGRESSION:
        return TASK_TYPE_REGRESSION
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return TASK_TYPE_BINARY
    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return TASK_TYPE_MULTILABEL
    raise ValueError(f"Unsupported task.task_type: {task.task_type}")
