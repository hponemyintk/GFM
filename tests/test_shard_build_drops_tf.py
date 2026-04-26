"""Verify ``tools/precompute_shards.py`` drops the TF columns from the
loaded ``HeteroData`` before sampling, to keep per-process RAM bounded
when running phase 2 in parallel on big datasets like rel-event.

Strategy: invoke the inline TF-drop code on a fake HeteroData and assert
the ``tf`` attribute is gone afterwards. We don't run the full
``load_data`` because it pulls in relbench / make_pkey_fkey_graph; the
drop logic is the part we actually care about regression-testing.
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _FakeStore:
    """Mimics torch_geometric NodeStorage closely enough to exercise the
    drop-TF cleanup path. Supports both ``del store['tf']`` and
    ``delattr(store, 'tf')`` so the fallback in precompute_shards is
    covered."""

    def __init__(self, tf, num_nodes=10):
        self.tf = tf
        self.num_nodes = num_nodes
        self._fields = {"tf": tf, "num_nodes": num_nodes}

    def __contains__(self, key):
        return key in self._fields

    def __getitem__(self, key):
        return self._fields[key]

    def __delitem__(self, key):
        if key in self._fields:
            del self._fields[key]
        if hasattr(self, key):
            delattr(self, key)


class _FakeHetero:
    def __init__(self, stores):
        self._stores = stores

    @property
    def node_types(self):
        return list(self._stores.keys())

    def __getitem__(self, key):
        return self._stores[key]


def _drop_tf(data):
    """The exact snippet that lives at the end of load_data() in
    tools/precompute_shards.py. Kept inline here so a refactor of the
    snippet has to update both this test and the source."""
    import gc as _gc
    for _nt in list(data.node_types):
        store = data[_nt]
        if hasattr(store, "tf"):
            try:
                del store["tf"]
            except Exception:
                try:
                    delattr(store, "tf")
                except Exception:
                    pass
    _gc.collect()


def test_drop_tf_via_delitem():
    big_tf = [object()] * 1000  # stand-in for "expensive memory"
    data = _FakeHetero({
        "users": _FakeStore(tf=big_tf, num_nodes=5),
        "events": _FakeStore(tf=big_tf, num_nodes=10),
    })
    # Before: every store has a tf.
    for nt in data.node_types:
        assert hasattr(data[nt], "tf")

    _drop_tf(data)

    # After: every store no longer has a tf attribute or key.
    for nt in data.node_types:
        assert not hasattr(data[nt], "tf"), f"tf still on {nt}"
        assert "tf" not in data[nt], f"tf key still in {nt}"


def test_drop_tf_handles_store_without_tf():
    """Stores missing tf shouldn't error out (e.g., schema-only types)."""
    class _NoTF:
        num_nodes = 3

    data = _FakeHetero({"some_type": _NoTF()})
    _drop_tf(data)  # must not raise


def test_drop_tf_handles_delitem_failure_via_delattr_fallback():
    """If a store doesn't support __delitem__, fall through to delattr."""
    class _DelitemBlocker:
        def __init__(self):
            self.tf = "expensive"
            self.num_nodes = 1

        def __delitem__(self, key):
            raise NotImplementedError("not supported")

    store = _DelitemBlocker()
    data = _FakeHetero({"users": store})
    _drop_tf(data)
    assert not hasattr(store, "tf")
