"""Real-data dev-kyaw equivalence test for the CSR sampler (S19).

Distinct from tests/test_csr_sampler.py because that suite's conftest.py
mocks out relbench / torch_geometric / sentence_transformers to keep the
fast unit suite from JIT-compiling C++ extensions on M1. This test needs
the real modules to materialize rel-f1 and run the production sampling
code path on actual relbench data.

Coverage: 200 train seeds from rel-f1.driver-top3 x K in {16, 64, 300}
= 600 head-to-head sample_local_subgraph vs dev-kyaw _process_one_seed
comparisons. Catches divergences the synthetic-graph S16/S17 cases miss
(real-degree distributions hitting the 5000 cap, time-tensor dtype edge
cases, multi-edge fanout patterns).

Opt-in (does ~30s of materialization + needs the relbench cache populated):

    PYTEST_REAL_DATA=1 pytest tests/test_csr_sampler_real_data.py -v
"""

from __future__ import annotations

import sys

# tests/conftest.py installs MagicMock-based stubs for torch_geometric,
# relbench, sentence_transformers, h5py to keep the fast unit suite light.
# We need the REAL modules here -- pop the stubs out of sys.modules BEFORE
# the real imports run so subsequent ``import relbench`` etc. resolve to
# the genuine packages.
for _m in [
    "torch_geometric", "torch_geometric.data", "torch_geometric.nn",
    "torch_geometric.transforms",
    "sentence_transformers",
    "relbench", "relbench.base",
    "relbench.datasets", "relbench.tasks",
    "relbench.modeling", "relbench.modeling.graph", "relbench.modeling.utils",
    "h5py",
]:
    sys.modules.pop(_m, None)

import gc as _gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_REAL_DATA = os.environ.get("PYTEST_REAL_DATA", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _REAL_DATA,
    reason="Real-data test; set PYTEST_REAL_DATA=1 to run.",
)


def _multiset(tokens):
    """Comparable multiset of (type, idx, hop) for token equivalence."""
    return sorted((t, i, h) for (t, i, h, _t, _c) in tokens)


def _dev_kyaw_process_one_seed(adjacency, all_nodes, data, K, seed_type,
                               seed_idx, seed_t, seed_val):
    """Run dev-kyaw's _process_one_seed without going through Pool.

    utils.py uses module-level globals normally set by the worker
    initializer; we set them explicitly for direct invocation.
    """
    import utils
    utils.GLOBAL_ADJ = adjacency
    utils.GLOBAL_ALL_NODES = all_nodes
    args = (data, K, seed_type, seed_idx, seed_t, seed_val)
    _, _, final_tokens, edge_index = utils._process_one_seed(args)
    return final_tokens, edge_index


@pytest.fixture(scope="module")
def rel_f1_data_and_task():
    """Materialize rel-f1 once and reuse across all S19 cases.

    Uses the same call site as tools/precompute_shards.py:load_data so any
    behavior difference between the test and the production builder is
    immediately visible.
    """
    pytest.importorskip("relbench.datasets")
    pytest.importorskip("relbench.tasks")
    pytest.importorskip("relbench.modeling.graph")
    pytest.importorskip("torch_frame")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from gfm_data.stypes import filter_to_db_columns, load_or_generate_stypes
    from utils import GloveTextEmbedding

    cache_dir = os.path.expanduser("~/.cache/relbench_examples")
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True)

    stypes_path = Path(cache_dir) / "rel-f1" / "stypes.json"
    cs = load_or_generate_stypes(stypes_path, dataset, upto_test_timestamp=True)
    db = dataset.get_db(upto_test_timestamp=True)
    cs = filter_to_db_columns(cs, db)

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=cs,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=embed_device),
            batch_size=512,
        ),
        cache_dir=f"{cache_dir}/rel-f1/materialized",
    )

    # The sampler reads adjacency + per-type time tensors; TF columns are
    # never touched. Drop them so downstream calls (and the dev-kyaw side)
    # don't accidentally materialize a multi-GB tensor.
    for nt in list(data.node_types):
        store = data[nt]
        if hasattr(store, "tf"):
            try:
                n = store.num_nodes
                if n is not None:
                    store.num_nodes = int(n)
            except Exception:
                pass
            try:
                del store["tf"]
            except Exception:
                try:
                    delattr(store, "tf")
                except Exception:
                    pass
    _gc.collect()
    return data, task


def test_S19_dev_kyaw_equivalent_on_rel_f1(rel_f1_data_and_task):
    """End-to-end equivalence vs dev-kyaw on real rel-f1 driver-top3.

    200 train seeds x K in {16, 64, 300} = 600 head-to-head comparisons.
    Each comparison asserts:
      * Token multiset (the (type, idx, hop) bag) matches dev-kyaw exactly.
      * edge_index matches dev-kyaw bit-for-bit.
    """
    from gfm_data.graph_cache import DatasetGraphCache
    from gfm_data.sampler import sample_local_subgraph
    from utils import build_adjacency_hetero
    from relbench.modeling.graph import get_node_train_table_input

    data, task = rel_f1_data_and_task

    # Build BOTH adjacencies from the same materialization.
    cache = DatasetGraphCache(data=data, undirected=True)
    dev_adj = build_adjacency_hetero(data, undirected=True)
    dev_all = []
    for nt in data.node_types:
        n = (
            int(data[nt]["x"].size(0)) if "x" in data[nt]
            else int(data[nt].num_nodes)
        )
        for i in range(n):
            dev_all.append((nt, i))

    # Pull 200 train seeds from rel-f1.driver-top3.
    table = task.get_table("train")
    table_input = get_node_train_table_input(table, task)
    raw_seed_type, seed_idxs_t = table_input.nodes
    seed_times_t = table_input.time

    n_seeds = min(200, len(seed_idxs_t))
    seed_idxs = [int(seed_idxs_t[i].item()) for i in range(n_seeds)]
    seed_times = [float(seed_times_t[i].item()) for i in range(n_seeds)]

    for K in (16, 64, 300):
        for i in range(n_seeds):
            seed_val = hash(
                (raw_seed_type, seed_idxs[i], seed_times[i], K)
            ) & 0xFFFFFFFF

            f_new, e_new = sample_local_subgraph(
                cache, K=K, seed_node_type=raw_seed_type,
                seed_node_idx=seed_idxs[i], seed_time=seed_times[i],
                seed_val=seed_val,
            )
            f_dev, e_dev = _dev_kyaw_process_one_seed(
                dev_adj, dev_all, data, K, raw_seed_type,
                seed_idxs[i], seed_times[i], seed_val,
            )

            assert _multiset(f_new) == _multiset(f_dev), (
                f"token multiset diverges at seed {i} K={K} "
                f"(type={raw_seed_type}, idx={seed_idxs[i]}, "
                f"t={seed_times[i]})"
            )
            assert np.array_equal(e_new, e_dev), (
                f"edge_index diverges at seed {i} K={K} "
                f"(type={raw_seed_type}, idx={seed_idxs[i]}, "
                f"t={seed_times[i]})"
            )
