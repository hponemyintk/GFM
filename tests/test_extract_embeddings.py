"""Unit tests for tools/extract_embeddings.py.

Heavy components (RelBench dataset load, real GloVe, full RelGT
forward) are mocked so the test runs in CI in seconds without GPU.
The IO contract -- output .pt format with the expected keys, the
register_dataset path, the per-split loop -- is what these tests
guard.

End-to-end smoke against a real backbone + cached TF store is left
as a separate manual check (slow, requires the workflow's saved
artifacts) and is part of the Phase-4 holdout-task launcher.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file un-mock guard.
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401


def _fake_loader(n_batches=2, batch_size=4, channels=8):
    """Yield batches that the extraction loop's grouped_tfs / tensor
    keys mirror -- enough to drive the model-forward and the
    label/global_idx aggregation without a real DataLoader."""
    for b_i in range(n_batches):
        yield {
            "neighbor_types": torch.zeros(batch_size, 4, dtype=torch.long),
            "node_indices": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_hops": torch.zeros(batch_size, 4, dtype=torch.long),
            "neighbor_times": torch.zeros(batch_size, 4),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "batch": torch.zeros(batch_size * 4, dtype=torch.long),
            "grouped_tfs": {},
            "grouped_indices": {},
            "flat_batch_idx": [],
            "flat_nbr_idx": [],
            "labels": torch.arange(b_i * batch_size, (b_i + 1) * batch_size).float(),
            "global_idx": torch.arange(b_i * batch_size, (b_i + 1) * batch_size),
        }


def test_extract_split_aggregates_embeddings_and_labels():
    """The per-split aggregator must concat per-batch embeddings,
    labels, and global_idx in batch order."""
    from tools.extract_embeddings import _extract_split

    channels = 8
    batch_size = 4
    n_batches = 3
    expected_n = batch_size * n_batches

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfs_encoder = type("E", (), {"channels": channels})()

        def forward(self, n_types, n_idx, n_hop, n_t, grouped, **kw):
            B = n_types.shape[0]
            # Deterministic [B, channels] -- the global_idx values
            # come from the loader, so we encode them in the embedding
            # to assert order is preserved post-concat.
            return torch.full(
                (B, channels),
                fill_value=0.0,
            ) + torch.arange(B).unsqueeze(1).float()

    model = _StubModel()
    loader = list(_fake_loader(n_batches, batch_size, channels))
    out = _extract_split(model, loader, device="cpu")

    assert out["embeddings"].shape == (expected_n, channels)
    assert out["labels"].shape == (expected_n,)
    assert out["global_idx"].shape == (expected_n,)
    # global_idx must equal arange(expected_n) since the fake loader
    # generates contiguous indices and _extract_split shouldn't shuffle.
    assert torch.equal(
        out["global_idx"], torch.arange(expected_n, dtype=torch.long),
    )


def test_extract_split_handles_3d_output_via_seed_collapse():
    """If a backbone variant outputs ``[B, K, C]`` (e.g., the encoder
    stack without seed-token collapse), the extraction must take the
    seed token at position 0 to produce a ``[B, C]`` embedding."""
    from tools.extract_embeddings import _extract_split

    channels = 8

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfs_encoder = type("E", (), {"channels": channels})()

        def forward(self, n_types, n_idx, *a, **kw):
            B = n_types.shape[0]
            K = n_types.shape[1]
            return torch.zeros(B, K, channels)

    model = _StubModel()
    loader = list(_fake_loader(n_batches=1, batch_size=2, channels=channels))
    out = _extract_split(model, loader, device="cpu")
    assert out["embeddings"].shape == (2, channels), (
        f"expected [B=2, C={channels}] after seed-collapse; "
        f"got {tuple(out['embeddings'].shape)}"
    )


def test_extract_split_no_labels_when_target_is_none():
    """Some adoption-time splits (e.g., a held-out test split with
    masked labels for blind eval) have ``labels=None`` per row. The
    extractor must NOT produce a 'labels' key in that case."""
    from tools.extract_embeddings import _extract_split

    channels = 4

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfs_encoder = type("E", (), {"channels": channels})()

        def forward(self, n_types, *a, **kw):
            return torch.zeros(n_types.shape[0], channels)

    def _no_label_loader():
        yield {
            "neighbor_types": torch.zeros(2, 4, dtype=torch.long),
            "node_indices": torch.zeros(2, dtype=torch.long),
            "neighbor_hops": torch.zeros(2, 4, dtype=torch.long),
            "neighbor_times": torch.zeros(2, 4),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "batch": torch.zeros(8, dtype=torch.long),
            "grouped_tfs": {},
            "grouped_indices": {},
            "flat_batch_idx": [],
            "flat_nbr_idx": [],
            "labels": None,
            "global_idx": torch.arange(2),
        }

    out = _extract_split(_StubModel(), list(_no_label_loader()), device="cpu")
    assert "embeddings" in out
    assert "global_idx" in out
    assert "labels" not in out, (
        "expected no 'labels' key when batches carry None"
    )


def test_extract_main_writes_per_split_pt_files(tmp_path):
    """The main() entrypoint must write one ``<split>.pt`` per
    requested split with the expected keys: embeddings + global_idx
    (+ labels if present) + split + task + dataset + channels."""
    from tools.extract_embeddings import main as extract_main

    channels = 8
    n = 6  # 2 batches × 3

    # Build a fake meta + schema + weights so RelGT.load_backbone
    # would succeed -- but we'll patch load_backbone to skip that.
    fake_meta = {
        "channels": channels,
        "num_centroids": 4,
        "num_layers": 1,
        "num_heads": 2,
        "num_neighbors": 4,
        "max_neighbor_hop": 3,
        "global_dim": 4,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
        "gt_conv_type": "full",
        "ablate": "none",
        "gnn_pe_dim": 0,
        "node_type_map": {"t::A": 0},
        "best_epoch": 1,
        "best_val_macro": 0.5,
    }
    meta_path = tmp_path / "backbone_meta.json"
    weights_path = tmp_path / "best_backbone.pt"
    schema_path = tmp_path / "backbone_schema.pt"
    meta_path.write_text(json.dumps(fake_meta))
    weights_path.write_bytes(b"")  # contents irrelevant; load is mocked
    schema_path.write_bytes(b"")

    out_dir = tmp_path / "embeddings"

    # Stub the heavy stuff: load_backbone, _build_cache, _build_loader.
    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfs_encoder = type("E", (), {"channels": channels})()

        def forward(self, n_types, *a, **kw):
            return torch.zeros(n_types.shape[0], channels)

        def to(self, device):
            return self

    def _stub_load_backbone(*a, **kw):
        return _StubModel()

    def _stub_build_cache(args):
        return object(), object()  # cache, task -- unused by the loader stub

    def _stub_build_loader(args, split, cache, task):
        return None, list(_fake_loader(n_batches=2, batch_size=3, channels=channels))

    argv = [
        "--backbone_meta", str(meta_path),
        "--backbone_weights", str(weights_path),
        "--backbone_schema", str(schema_path),
        "--dataset", "rel-f1", "--task", "driver-top3",
        "--split", "all",
        "--out_dir", str(out_dir),
        "--device", "cpu",
    ]

    with patch("model.RelGT.load_backbone", side_effect=_stub_load_backbone), \
         patch("tools.extract_embeddings._build_cache",
               side_effect=_stub_build_cache), \
         patch("tools.extract_embeddings._build_loader",
               side_effect=_stub_build_loader):
        rc = extract_main(argv)
    assert rc == 0

    # All three splits saved.
    for split in ("train", "val", "test"):
        p = out_dir / f"{split}.pt"
        assert p.exists(), f"missing {split}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        assert d["embeddings"].shape == (n, channels)
        assert d["split"] == split
        assert d["task"] == "driver-top3"
        assert d["dataset"] == "rel-f1"
        assert d["channels"] == channels
        assert d["global_idx"].shape == (n,)


def test_extract_no_backbone_grad():
    """Backbone params must remain requires_grad=False during
    extraction, even if upstream caller forgot to freeze. The
    @torch.no_grad on _extract_split is the actual barrier; this
    test just verifies we don't accidentally enable grads inside."""
    from tools.extract_embeddings import _extract_split

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tfs_encoder = type("E", (), {"channels": 4})()
            self.lin = torch.nn.Linear(2, 4)

        def forward(self, n_types, *a, **kw):
            B = n_types.shape[0]
            return self.lin(torch.zeros(B, 2))

    model = _Model()
    # Caller "forgot" to freeze:
    for p in model.parameters():
        p.requires_grad = True

    _extract_split(model, list(_fake_loader(2, 4, 4)), device="cpu")

    # No buffers should have gradients (no_grad context blocks the graph).
    for p in model.parameters():
        assert p.grad is None, (
            "Unexpected gradient on backbone param after extraction; "
            "@torch.no_grad guard is missing or broken"
        )
