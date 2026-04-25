"""Disk-backed (memmap) TensorFrame columns.

Replaces the in-RAM ``data[node_type].tf`` dense tensors with per-column
memmap files. The OS pages in only the rows the BFS-reached samples
actually need, so resident memory is bounded by
``num_workers × batch_size × K × bytes_per_row``, not by the total table
size. This is what makes ``--max_rows_per_task`` actually bound RAM and
what unlocks training across rel-event (41M rows) on the laptop.

Layout, one directory per node-type table::

    <root>/<table>/
        meta.json                        # schema + dtypes + dims
        numerical.f32                    # [N, C_num]
        categorical.i64                  # [N, C_cat]
        timestamp.i64                    # [N, C_ts, 7]
        embedding.values.f32             # [N, total_emb_dim]
        embedding.offset.i64             # [C_emb + 1]

Multi-categorical (jagged per-row) is not exercised by rel-f1; it lands in
PR4 alongside rel-event. The builder raises ``NotImplementedError`` if it
encounters such a column so we fail fast rather than silently drop data.

The reader rebuilds a ``torch_frame.TensorFrame`` slice that is shape- and
dtype-equivalent to ``tf[idx]`` from the original in-RAM TF (test T1).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch_frame import TensorFrame, stype
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.multi_nested_tensor import MultiNestedTensor


# Per-stype memmap dtype.
_STYPE_DTYPE = {
    "numerical": np.float32,
    "categorical": np.int64,
    "timestamp": np.int64,  # always [N, C, 7]
    "embedding_values": np.float32,
    "embedding_offset": np.int64,
    "multicategorical_values": np.int64,
    "multicategorical_offset": np.int64,
}


def _stype_str(s: stype) -> str:
    return s.value  # "numerical" / "categorical" / ...


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------------------------------------------- builder
def build_tf_store(tf, root: str):
    """Materialize a single TensorFrame to disk under ``root``.

    Parameters
    ----------
    tf : torch_frame.TensorFrame
    root : str
        Output directory; created if missing. Existing contents may be
        overwritten.
    """
    _ensure_dir(root)
    meta: Dict[str, object] = {
        "num_rows": int(tf.num_rows),
        "col_names_dict": {_stype_str(s): list(cs) for s, cs in tf.col_names_dict.items()},
        "stypes": [_stype_str(s) for s in tf.feat_dict.keys()],
        "shapes": {},
        "embedding_offset": None,
        "multicategorical_num_cols": None,  # set if stype.multicategorical present
        "layout_version": 1,
    }

    for s, t in tf.feat_dict.items():
        s_name = _stype_str(s)
        if s_name == "embedding":
            # MultiEmbeddingTensor: values [N, total_dim] + offset [C+1]
            mt: MultiEmbeddingTensor = t
            values_np = mt.values.detach().cpu().numpy().astype(_STYPE_DTYPE["embedding_values"], copy=False)
            offset_np = mt.offset.detach().cpu().numpy().astype(_STYPE_DTYPE["embedding_offset"], copy=False)
            meta["shapes"]["embedding_values"] = list(values_np.shape)
            meta["shapes"]["embedding_offset"] = list(offset_np.shape)
            meta["embedding_offset"] = offset_np.tolist()
            values_np.tofile(os.path.join(root, "embedding.values.f32"))
            offset_np.tofile(os.path.join(root, "embedding.offset.i64"))
        elif s_name in ("numerical", "categorical", "timestamp"):
            arr = t.detach().cpu().numpy().astype(_STYPE_DTYPE[s_name], copy=False)
            meta["shapes"][s_name] = list(arr.shape)
            arr.tofile(os.path.join(root, f"{s_name}.{ _STYPE_DTYPE[s_name].__name__}"))
        elif s_name == "multicategorical":
            # MultiNestedTensor: flat ``values`` + flat ``offset`` of length
            # num_rows*num_cols + 1 (one offset per (row, col) cell).
            mnt: MultiNestedTensor = t
            values_np = mnt.values.detach().cpu().numpy().astype(
                _STYPE_DTYPE["multicategorical_values"], copy=False
            )
            offset_np = mnt.offset.detach().cpu().numpy().astype(
                _STYPE_DTYPE["multicategorical_offset"], copy=False
            )
            meta["shapes"]["multicategorical_values"] = list(values_np.shape)
            meta["shapes"]["multicategorical_offset"] = list(offset_np.shape)
            meta["multicategorical_num_cols"] = int(mnt.num_cols)
            values_np.tofile(os.path.join(root, "multicategorical.values.i64"))
            offset_np.tofile(os.path.join(root, "multicategorical.offset.i64"))
        else:
            raise NotImplementedError(f"unsupported stype: {s_name}")

    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_dataset_tf_store(data, root: str):
    """Build per-table TF stores for all node types of a HeteroData."""
    for nt in data.node_types:
        tf = getattr(data[nt], "tf", None)
        if tf is None:
            continue
        build_tf_store(tf, os.path.join(root, nt))


# ------------------------------------------------------------------ reader
@dataclass
class _TableMeta:
    num_rows: int
    col_names_dict: Dict[stype, List[str]]
    stypes: List[str]
    shapes: Dict[str, List[int]]
    embedding_offset: Optional[List[int]]
    multicategorical_num_cols: Optional[int]


class TFStoreReader:
    """Memmap reader for one table. ``view(row_idx)`` returns a TensorFrame."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "meta.json"), "r") as f:
            raw = json.load(f)
        # Re-hydrate stype keys.
        self.meta = _TableMeta(
            num_rows=int(raw["num_rows"]),
            col_names_dict={
                stype(s): list(cs) for s, cs in raw["col_names_dict"].items()
            },
            stypes=list(raw["stypes"]),
            shapes={k: list(v) for k, v in raw["shapes"].items()},
            embedding_offset=raw.get("embedding_offset"),
            multicategorical_num_cols=raw.get("multicategorical_num_cols"),
        )
        # Lazy memmaps.
        self._mm: Dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return self.meta.num_rows

    # ------------------- low-level memmap access
    def _open(self, name: str) -> np.memmap:
        if name in self._mm:
            return self._mm[name]
        if name == "numerical":
            shape = tuple(self.meta.shapes["numerical"])
            dtype = _STYPE_DTYPE["numerical"]
            path = os.path.join(self.root, f"numerical.{dtype.__name__}")
        elif name == "categorical":
            shape = tuple(self.meta.shapes["categorical"])
            dtype = _STYPE_DTYPE["categorical"]
            path = os.path.join(self.root, f"categorical.{dtype.__name__}")
        elif name == "timestamp":
            shape = tuple(self.meta.shapes["timestamp"])
            dtype = _STYPE_DTYPE["timestamp"]
            path = os.path.join(self.root, f"timestamp.{dtype.__name__}")
        elif name == "embedding_values":
            shape = tuple(self.meta.shapes["embedding_values"])
            dtype = _STYPE_DTYPE["embedding_values"]
            path = os.path.join(self.root, "embedding.values.f32")
        elif name == "embedding_offset":
            shape = tuple(self.meta.shapes["embedding_offset"])
            dtype = _STYPE_DTYPE["embedding_offset"]
            path = os.path.join(self.root, "embedding.offset.i64")
        elif name == "multicategorical_values":
            shape = tuple(self.meta.shapes["multicategorical_values"])
            dtype = _STYPE_DTYPE["multicategorical_values"]
            path = os.path.join(self.root, "multicategorical.values.i64")
        elif name == "multicategorical_offset":
            shape = tuple(self.meta.shapes["multicategorical_offset"])
            dtype = _STYPE_DTYPE["multicategorical_offset"]
            path = os.path.join(self.root, "multicategorical.offset.i64")
        else:
            raise KeyError(name)
        mm = np.memmap(path, mode="r", shape=shape, dtype=dtype)
        self._mm[name] = mm
        return mm

    # ------------------- TF reconstruction
    def view(self, row_idx) -> TensorFrame:
        """Return a ``TensorFrame`` slice for the given row indices.

        Parameters
        ----------
        row_idx : int, sequence[int], np.ndarray, or torch.Tensor
        """
        if isinstance(row_idx, torch.Tensor):
            idx_np = row_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        elif isinstance(row_idx, (list, tuple)):
            idx_np = np.asarray(row_idx, dtype=np.int64)
        elif isinstance(row_idx, np.ndarray):
            idx_np = row_idx.astype(np.int64, copy=False)
        elif isinstance(row_idx, int):
            idx_np = np.asarray([row_idx], dtype=np.int64)
        else:
            raise TypeError(f"unsupported row_idx type: {type(row_idx)}")

        feat_dict = {}
        for s_name in self.meta.stypes:
            s_enum = stype(s_name)
            if s_name == "numerical":
                arr = np.array(self._open("numerical")[idx_np])  # copy
                feat_dict[s_enum] = torch.from_numpy(arr)
            elif s_name == "categorical":
                arr = np.array(self._open("categorical")[idx_np])
                feat_dict[s_enum] = torch.from_numpy(arr)
            elif s_name == "timestamp":
                arr = np.array(self._open("timestamp")[idx_np])
                feat_dict[s_enum] = torch.from_numpy(arr)
            elif s_name == "embedding":
                values_arr = np.array(self._open("embedding_values")[idx_np])
                offset_arr = np.array(self._open("embedding_offset"))  # small, copy whole
                num_rows = values_arr.shape[0]
                num_cols = offset_arr.shape[0] - 1
                feat_dict[s_enum] = MultiEmbeddingTensor(
                    num_rows=num_rows,
                    num_cols=num_cols,
                    values=torch.from_numpy(values_arr),
                    offset=torch.from_numpy(offset_arr),
                )
            elif s_name == "multicategorical":
                feat_dict[s_enum] = self._view_multicategorical(idx_np)
            else:
                raise NotImplementedError(s_name)

        return TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=self.meta.col_names_dict,
            num_rows=int(idx_np.shape[0]),
        )

    def _view_multicategorical(self, idx_np: np.ndarray) -> "MultiNestedTensor":
        """Build a MultiNestedTensor slice from the memmaps.

        The on-disk ``offset`` is layout-major (one entry per (row, col)
        cell + 1), so cell ``(i, j)`` lives at
        ``values[offset[i*num_cols + j] : offset[i*num_cols + (j+1)]]``.
        We copy the cell slices for each requested row into a new flat
        buffer and produce a fresh offset of length
        ``len(idx_np) * num_cols + 1``.
        """
        num_cols = self.meta.multicategorical_num_cols
        assert num_cols is not None
        full_values = self._open("multicategorical_values")
        full_offset = self._open("multicategorical_offset")
        n_out = idx_np.shape[0]

        # First pass: compute per-cell lengths -> new offset.
        new_offset = np.zeros(n_out * num_cols + 1, dtype=np.int64)
        cur = 0
        per_cell = []  # list[(start, end)] in old values
        for i in range(n_out):
            row = int(idx_np[i])
            base = row * num_cols
            for j in range(num_cols):
                start = int(full_offset[base + j])
                end = int(full_offset[base + j + 1])
                per_cell.append((start, end))
                cur += (end - start)
                new_offset[i * num_cols + j + 1] = cur

        # Second pass: copy values.
        new_values = np.empty(cur, dtype=full_values.dtype)
        write = 0
        for (s, e) in per_cell:
            n = e - s
            if n > 0:
                new_values[write:write + n] = full_values[s:e]
                write += n
        return MultiNestedTensor(
            num_rows=n_out,
            num_cols=num_cols,
            values=torch.from_numpy(new_values),
            offset=torch.from_numpy(new_offset),
        )
