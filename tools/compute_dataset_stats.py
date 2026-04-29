"""Compute per-column stats for a TF store.

Walks ``<tf_store_root>/<table>/`` (the layout produced by
``tools/build_tf_store.py`` / ``gfm_data.tf_store.build_tf_store``) and
emits a ``col_stats_dict`` consumable by
``NeighborTfsEncoder.register_dataset(...)``.

Output format::

    {
        "<prefixed_table_name>": {
            "<col_name>": {StatType.MEAN: <float>, StatType.STD: <float>},
            "<cat_col>":  {StatType.COUNT: <int>},
            "<emb_col>":  {StatType.EMB_DIM: <int>},
            ...
        },
        ...
    }

Stats per stype:

  * **numerical**  -> ``StatType.MEAN``, ``StatType.STD`` (NaN-aware via
    ``np.nanmean`` / ``np.nanstd``; std clamped to a 1e-8 floor so the
    encoder's downstream ``(x - mean) / (std + 1e-8)`` stays bounded
    on degenerate constant columns).
  * **categorical** -> ``StatType.COUNT`` = number of distinct integer
    levels observed across the column. The model's
    ``SharedCategoricalEncoder`` uses a hash-bucket and doesn't read
    this stat at runtime, but the field is populated for parity with
    relbench's ``make_pkey_fkey_graph`` output and downstream
    inspection.
  * **multicategorical** -> ``StatType.COUNT`` = distinct values across
    the flattened jagged column.
  * **embedding** -> ``StatType.EMB_DIM`` = ``offset[i+1] - offset[i]``.
  * **timestamp** -> not emitted (no per-column stat used by the
    encoder).

Phase-4 use case: at adoption time on a held-out dataset, run
``tools/build_tf_store.py`` to materialize the TFs, then this tool to
compute stats, then call
``backbone.tfs_encoder.register_dataset(node_type_map, col_names_dict,
col_stats_dict)`` before extraction. Cheap (a few-second walk over
memmap files) and reproducible -- no relbench import required.

Usage::

    python -m tools.compute_dataset_stats \\
        --tf_store_dir ~/.cache/relbench_examples/tf_store/rel-f1 \\
        --out         ~/.cache/relbench_examples/tf_store/rel-f1.col_stats.pt \\
        [--name_prefix rel-f1::]

``--name_prefix`` (optional) prepends to each table-directory name in
the output dict so the keys match the prefixed-type convention used
by ``DatasetGraphCache`` / multi-task ``unified_type_map``. Without
it, output keys are bare directory names (matches single-dataset
``main_node_ddp`` usage where no prefix is applied).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_frame.data.stats import StatType


# Floor for std: matches the encoder's runtime ``(x - mean) / (std +
# 1e-8)``. A constant column has std=0; without the floor the
# downstream divide produces inf.
_STD_FLOOR = 1e-8


def _read_meta(table_dir: Path) -> dict:
    with open(table_dir / "meta.json") as f:
        return json.load(f)


def _open_memmap(path: Path, dtype, shape) -> np.memmap:
    return np.memmap(path, mode="r", shape=tuple(shape), dtype=dtype)


def compute_table_stats(table_dir: Path) -> Dict[str, Dict[StatType, object]]:
    """Compute per-column stats for a single TF-store table.

    Returns a dict keyed by column name; each value is a per-stat
    sub-dict using ``StatType`` keys.
    """
    if not (table_dir / "meta.json").exists():
        raise FileNotFoundError(
            f"meta.json not found in {table_dir} -- not a TF-store table"
        )
    meta = _read_meta(table_dir)
    col_names = meta["col_names_dict"]      # {stype_str: [names]}
    shapes = meta.get("shapes", {})

    out: Dict[str, Dict[StatType, object]] = {}

    # --- numerical: per-column mean + std, NaN-aware ---
    if "numerical" in col_names and col_names["numerical"]:
        cols = col_names["numerical"]
        if "numerical" not in shapes:
            raise ValueError(
                f"{table_dir}/meta.json declares numerical cols "
                f"but no 'numerical' shape; corrupt store?"
            )
        arr = _open_memmap(
            table_dir / "numerical.float32",
            dtype=np.float32, shape=shapes["numerical"],
        )  # [N, C_num]
        # nanmean/nanstd handle all-NaN columns by returning NaN; we
        # convert those to mean=0, std=1 so register_dataset gets a
        # safe identity transform on degenerate columns.
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        stds = np.nan_to_num(stds, nan=1.0)
        for i, col in enumerate(cols):
            out[col] = {
                StatType.MEAN: float(means[i]),
                StatType.STD: float(max(stds[i], _STD_FLOOR)),
            }

    # --- categorical: per-column distinct level count ---
    if "categorical" in col_names and col_names["categorical"]:
        cols = col_names["categorical"]
        if "categorical" not in shapes:
            raise ValueError(
                f"{table_dir}/meta.json declares categorical cols "
                f"but no 'categorical' shape; corrupt store?"
            )
        arr = _open_memmap(
            table_dir / "categorical.int64",
            dtype=np.int64, shape=shapes["categorical"],
        )  # [N, C_cat]
        # Per-column np.unique. Slower than columnar batch ops but
        # rel-f1 / rel-hm scale (< 1e7 rows) makes it instant.
        for i, col in enumerate(cols):
            unique_count = int(np.unique(arr[:, i]).shape[0])
            out[col] = {StatType.COUNT: unique_count}

    # --- multicategorical: distinct values across the flat values ---
    if "multicategorical" in col_names and col_names["multicategorical"]:
        cols = col_names["multicategorical"]
        v_shape = shapes.get("multicategorical_values")
        o_shape = shapes.get("multicategorical_offset")
        ncols = meta.get("multicategorical_num_cols")
        if v_shape and o_shape and ncols:
            values = _open_memmap(
                table_dir / "multicategorical.values.int64",
                dtype=np.int64, shape=v_shape,
            )
            offset = _open_memmap(
                table_dir / "multicategorical.offset.int64",
                dtype=np.int64, shape=o_shape,
            )
            n_rows = (offset.shape[0] - 1) // ncols
            for i, col in enumerate(cols):
                # Cells (r, i) live at offset[r*ncols + i] : offset[r*ncols + i + 1].
                col_start = offset[i :: ncols][:-1] if False else None  # placeholder
                # Walk per-row to gather this column's values.
                vals = []
                for r in range(n_rows):
                    s = int(offset[r * ncols + i])
                    e = int(offset[r * ncols + i + 1])
                    if e > s:
                        vals.append(values[s:e])
                if vals:
                    flat = np.concatenate(vals)
                    out[col] = {StatType.COUNT: int(np.unique(flat).shape[0])}
                else:
                    out[col] = {StatType.COUNT: 0}
        else:
            # Layout fields missing -- skip rather than crash.
            for col in cols:
                out[col] = {StatType.COUNT: 0}

    # --- embedding: per-column emb_dim from offset ---
    if "embedding" in col_names and col_names["embedding"]:
        cols = col_names["embedding"]
        # The meta carries the offset directly (small int list).
        offset = meta.get("embedding_offset")
        if offset is None:
            # Fall back to memmap.
            o_shape = shapes.get("embedding_offset")
            if o_shape is None:
                raise ValueError(
                    f"{table_dir}/meta.json declares embedding cols "
                    f"but no embedding_offset; corrupt store?"
                )
            offset = _open_memmap(
                table_dir / "embedding.offset.i64",
                dtype=np.int64, shape=o_shape,
            ).tolist()
        for i, col in enumerate(cols):
            emb_dim = int(offset[i + 1] - offset[i])
            out[col] = {StatType.EMB_DIM: emb_dim}

    # --- timestamp: no per-column stat needed by register_dataset ---
    # (encoder's SharedTimestampEncoder is stateless / per-batch).
    return out


def compute_dataset_stats(
    tf_store_root: str,
    name_prefix: str = "",
) -> Dict[str, Dict[str, Dict[StatType, object]]]:
    """Walk a TF-store root and compute stats for every table.

    Parameters
    ----------
    tf_store_root : str
        Directory under which each subdir is a TF-store table (i.e.,
        contains a ``meta.json``).
    name_prefix : str, optional
        Prepended to each table's directory name in the output dict
        keys. Use ``"<dataset>::"`` to match the prefixed-type
        convention used by ``DatasetGraphCache``; leave empty (default)
        for single-dataset bare-name keys.

    Returns
    -------
    dict
        ``{<prefixed_table>: {<col>: {<StatType>: <value>}}}``, ready
        for ``NeighborTfsEncoder.register_dataset``.
    """
    root = Path(tf_store_root)
    if not root.exists():
        raise FileNotFoundError(f"TF-store root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"TF-store root is not a directory: {root}")

    out: Dict[str, Dict[str, Dict[StatType, object]]] = {}
    for table_dir in sorted(root.iterdir()):
        if not table_dir.is_dir():
            continue
        if not (table_dir / "meta.json").exists():
            continue
        key = f"{name_prefix}{table_dir.name}"
        out[key] = compute_table_stats(table_dir)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tf_store_dir", required=True, type=str,
        help="TF-store root, e.g. ~/.cache/relbench_examples/tf_store/rel-f1",
    )
    p.add_argument(
        "--out", required=True, type=str,
        help="Output .pt path. Loadable via torch.load(out).",
    )
    p.add_argument(
        "--name_prefix", default="", type=str,
        help="Prefix for table-name keys in the output dict (e.g. "
             "'rel-f1::' for multi-task convention).",
    )
    args = p.parse_args(argv)

    stats = compute_dataset_stats(
        os.path.expanduser(args.tf_store_dir),
        name_prefix=args.name_prefix,
    )
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(stats, out_path)
    print(f"Wrote col_stats_dict for {len(stats)} tables -> {out_path}")
    for tn, cs in stats.items():
        print(f"  {tn}: {len(cs)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
