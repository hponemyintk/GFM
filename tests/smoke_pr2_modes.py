"""End-to-end smoke for PR2: streaming and precomputed_shards modes
produce equivalent batches; memmap-TF and in-RAM TF produce equivalent
batches. This stands in for tests E1 + E3 from the plan.

Run as: ``python tests/smoke_pr2_modes.py``.

This is gated by the rel-f1 cache being present (``~/.cache/relbench_examples/rel-f1/``)
and is intentionally not under pytest -- it loads relbench, builds shards
and TF memmaps in /tmp, and verifies tensors are identical across modes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task

from gfm_data import DatasetGraphCache, TaskTokens, collate_single_task
from gfm_data.tf_store import build_dataset_tf_store
from utils import GloveTextEmbedding


def load_data():
    cache_dir = os.path.expanduser("~/.cache/relbench_examples")
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True)
    stypes_path = Path(cache_dir) / "rel-f1" / "stypes.json"
    with open(stypes_path) as f:
        cs = json.load(f)
    for tab, c2s in cs.items():
        for col, st in c2s.items():
            c2s[col] = stype(st)
    data, _ = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device="cpu"),
            batch_size=256,
        ),
        cache_dir=f"{cache_dir}/rel-f1/materialized",
    )
    return data, task, cache_dir


def _tensor_eq_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Tensor equality that treats NaN==NaN (needed for missing-value cells)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(torch.allclose(a, b, equal_nan=True, rtol=0.0, atol=0.0))
    return bool(torch.equal(a, b))


def _eq(a, b, name=""):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{name}: tensor mismatch"
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), f"{name}: ndarray mismatch"
    elif isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), f"{name}: dict keys differ"
        for k in a:
            _eq(a[k], b[k], f"{name}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{name}: list len differs"
        for i, (ai, bi) in enumerate(zip(a, b)):
            _eq(ai, bi, f"{name}[{i}]")
    else:
        assert a == b, f"{name}: scalar mismatch ({a!r} vs {b!r})"


def main():
    K = 32
    N_BATCH = 16  # how many samples per batch we compare

    data, task, cache_dir = load_data()
    print("Building CSR cache (in-RAM TFs) ...")
    cache_inram = DatasetGraphCache(data=data, undirected=True, name_prefix=None)

    # ---- (1) STREAMING mode reference ----
    print("\n[1/4] Building TaskTokens in streaming mode ...")
    ds_stream = TaskTokens(
        cache=cache_inram, task=task, K=K, split="val", mode="streaming",
    )
    N_total = len(ds_stream.node_idxs)
    print(f"  val split has {N_total} seeds; comparing first {N_BATCH} as a batch")
    batch_stream = collate_single_task(ds_stream, [ds_stream[i] for i in range(N_BATCH)])
    print(f"  streaming batch keys: {sorted(batch_stream.keys())}")
    print(f"  neighbor_types shape: {batch_stream['neighbor_types'].shape}")

    # ---- (2) PRECOMPUTED_SHARDS mode ----
    print("\n[2/4] Building shards via tools/precompute_shards.py logic ...")
    with tempfile.TemporaryDirectory() as tmp:
        from gfm_data.shard_io import ShardWriter
        from gfm_data.sampler import sample_local_subgraph
        from relbench.modeling.graph import get_node_train_table_input

        shards_root = os.path.join(tmp, "shards")
        # Replicate tools/precompute_shards.py inline to avoid a subprocess.
        table_input = get_node_train_table_input(task.get_table("val"), task)
        raw_seed_type, seed_idxs = table_input.nodes
        seed_times = getattr(table_input, "time", None)
        type_to_id = cache_inram.node_type_to_index
        seed_node_type_prefixed = cache_inram.raw_to_prefixed[raw_seed_type]

        split_root = os.path.join(shards_root, str(K), "val")
        os.makedirs(split_root, exist_ok=True)
        N_shards = N_total
        writer = ShardWriter(split_root, K=K, total_samples=N_shards, shard_size=128)
        for s_idx in range(writer.num_shards):
            lo, hi = writer.shard_range(s_idx)
            size = hi - lo
            types = np.zeros((size, K), dtype=np.int16)
            indices = np.zeros((size, K), dtype=np.int32)
            hops = np.zeros((size, K), dtype=np.int8)
            times = np.zeros((size, K), dtype=np.float32)
            edges = []
            for k in range(size):
                gk = lo + k
                ni = int(seed_idxs[gk].item() if hasattr(seed_idxs[gk], "item") else seed_idxs[gk])
                st = float(seed_times[gk].item()) if seed_times is not None else 0.0
                sv = hash((seed_node_type_prefixed, ni, st, K)) & 0xFFFFFFFF
                fn, ei = sample_local_subgraph(
                    cache_inram, K, seed_node_type_prefixed, ni, st, sv,
                )
                for j, (t_str, nbr_loc, hop, t_val, _c) in enumerate(fn):
                    types[k, j] = type_to_id[t_str]
                    indices[k, j] = nbr_loc
                    hops[k, j] = hop
                    times[k, j] = t_val
                edges.append(ei)
            writer.write_shard(s_idx, types, indices, hops, times, edges)
        writer.finalize()

        ds_shards = TaskTokens(
            cache=cache_inram, task=task, K=K, split="val",
            mode="precomputed_shards", shards_dir=shards_root,
        )
        batch_shards = collate_single_task(
            ds_shards, [ds_shards[i] for i in range(N_BATCH)]
        )

        # ---- E1: streaming batch == precomputed_shards batch (mod TF identity) ----
        print("\n[3/4] E1: comparing streaming vs precomputed_shards batches ...")
        for k in ("neighbor_types", "neighbor_indices", "neighbor_hops",
                  "neighbor_times", "edge_index", "batch", "task_id",
                  "task_type_id", "labels", "node_indices", "global_idx"):
            _eq(batch_stream[k], batch_shards[k], k)
        print("  streaming == precomputed_shards on all subgraph tensors")

        # ---- (3) MEMMAP-TF mode ----
        print("\n[4/4] E3: comparing memmap-TF vs in-RAM TF batches ...")
        tf_root = os.path.join(tmp, "tfs")
        build_dataset_tf_store(data, tf_root)
        cache_memmap = DatasetGraphCache(
            data=data, undirected=True, name_prefix=None, tf_store_root=tf_root,
        )
        ds_stream_mm = TaskTokens(
            cache=cache_memmap, task=task, K=K, split="val", mode="streaming",
        )
        batch_mm = collate_single_task(
            ds_stream_mm, [ds_stream_mm[i] for i in range(N_BATCH)]
        )

        # Subgraph tensors must match.
        for k in ("neighbor_types", "neighbor_indices", "neighbor_hops",
                  "neighbor_times", "edge_index", "batch"):
            _eq(batch_stream[k], batch_mm[k], k)

        # grouped_tfs are TensorFrame objects -- compare their feat_dict tensors.
        for tid in batch_stream["grouped_tfs"]:
            assert tid in batch_mm["grouped_tfs"], tid
            tfa = batch_stream["grouped_tfs"][tid]
            tfb = batch_mm["grouped_tfs"][tid]
            assert sorted(tfa.feat_dict.keys(), key=lambda s: s.value) == \
                   sorted(tfb.feat_dict.keys(), key=lambda s: s.value)
            from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
            for s in tfa.feat_dict:
                fa, fb = tfa.feat_dict[s], tfb.feat_dict[s]
                if isinstance(fa, MultiEmbeddingTensor):
                    # NaN-tolerant: missing values are stored as NaN in
                    # numerical/embedding columns; in-RAM and memmap paths
                    # both preserve NaN positions, but torch.equal is strict.
                    assert _tensor_eq_nan(fa.values, fb.values), \
                        f"emb values mismatch for type {tid} stype {s}"
                    assert torch.equal(fa.offset, fb.offset), \
                        f"emb offset mismatch for type {tid} stype {s}"
                else:
                    assert _tensor_eq_nan(fa, fb), \
                        f"feat mismatch for type {tid} stype {s}"
        print("  memmap-TF == in-RAM TF on all features")

    print("\nALL PR2 SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
