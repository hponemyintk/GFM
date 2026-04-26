"""Regression test for the phase-3 OOM fix: ``_release_cache_data``.

After all DataLoader / model construction is complete and we're about
to fork worker processes, ``cache.data`` and ``TaskTokens.data`` must
be nulled in shards+tf_store mode so that:

  * ``HeteroData[type].time`` tensors (multi-GiB on rel-event) are freed.
  * Worker forks don't COW-duplicate the HeteroData reference graph.

The release function must:
  1. Be a no-op in streaming mode (sampler reads cache.data per batch).
  2. Be a no-op when tf_store_dir is None (cache.tf_view falls back to
     data[type].tf).
  3. Null cache.data AND every TaskTokens.data when conditions are met.

We re-import ``_release_cache_data`` via a textual extraction rather
than ``from train_multi_task import ...`` because train_multi_task
pulls in heavy modules (relbench, torch_geometric, sentence_transformers)
that conftest mocks at the top level only -- nested imports (e.g.,
``model.py`` -> ``torch_geometric.nn.dense.linear``) escape the mock.
"""

from __future__ import annotations

import os
import sys
import textwrap
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_release_fn():
    """Pull just the function source from train_multi_task.py and exec it.

    Keeps the test isolated from heavy imports while still exercising
    the actual code path that ships in the runtime.
    """
    src_path = Path(__file__).resolve().parents[1] / "train_multi_task.py"
    text = src_path.read_text()

    start_marker = "def _release_cache_data("
    start = text.index(start_marker)
    # Find the next top-level def/class after this function.
    after = text[start:]
    end_rel = -1
    for i, line in enumerate(after.splitlines(keepends=True)):
        if i == 0:
            continue
        if line.startswith("def ") or line.startswith("class "):
            # back off to byte offset
            end_rel = sum(
                len(l) for l in after.splitlines(keepends=True)[:i]
            )
            break
    body = after if end_rel < 0 else after[:end_rel]
    body = textwrap.dedent(body)
    # Stub helpers the function references.
    class _FakeDist:
        @staticmethod
        def is_initialized() -> bool:  # tests run without a process group
            return False

        @staticmethod
        def barrier() -> None:
            pass

    ns: dict = {"_rss_gb": lambda: 0.0, "dist": _FakeDist}
    exec(body, ns)
    return ns["_release_cache_data"]


_release_cache_data = _load_release_fn()


def _fake_caches_and_tokens(n_caches: int = 2, n_tokens_per_split: int = 3):
    caches = {}
    for i in range(n_caches):
        c = SimpleNamespace()
        c.data = SimpleNamespace(node_types=["a", "b"])
        caches[f"ds{i}"] = c
    task_tokens = {"train": [], "val": [], "test": []}
    for split in task_tokens:
        for j in range(n_tokens_per_split):
            tok = SimpleNamespace()
            tok.data = SimpleNamespace(num_nodes=42)
            task_tokens[split].append(tok)
    return caches, task_tokens


def test_release_drops_cache_and_tokens_in_shards_tf_store_mode():
    args = SimpleNamespace(mode="precomputed_shards", tf_store_dir="/tmp/tf")
    caches, task_tokens = _fake_caches_and_tokens()
    _release_cache_data(args, caches, task_tokens, local_rank=0)
    for c in caches.values():
        assert c.data is None
    for split_toks in task_tokens.values():
        for tok in split_toks:
            assert tok.data is None


def test_release_drops_in_hdf5_tf_store_mode_too():
    args = SimpleNamespace(mode="hdf5", tf_store_dir="/tmp/tf")
    caches, task_tokens = _fake_caches_and_tokens()
    _release_cache_data(args, caches, task_tokens, local_rank=0)
    for c in caches.values():
        assert c.data is None


def test_release_is_noop_in_streaming_mode():
    """Streaming sampler reads cache.data inside the per-batch loop."""
    args = SimpleNamespace(mode="streaming", tf_store_dir="/tmp/tf")
    caches, task_tokens = _fake_caches_and_tokens()
    _release_cache_data(args, caches, task_tokens, local_rank=0)
    for c in caches.values():
        assert c.data is not None
    for split_toks in task_tokens.values():
        for tok in split_toks:
            assert tok.data is not None


def test_release_is_noop_when_tf_store_dir_none():
    """Without tf_store_dir, cache.tf_view falls back to data[type].tf."""
    args = SimpleNamespace(mode="precomputed_shards", tf_store_dir=None)
    caches, task_tokens = _fake_caches_and_tokens()
    _release_cache_data(args, caches, task_tokens, local_rank=0)
    for c in caches.values():
        assert c.data is not None
    for split_toks in task_tokens.values():
        for tok in split_toks:
            assert tok.data is not None
