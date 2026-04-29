"""Regression tests for PR 2.1 -- best-val checkpoint to disk.

PR 2.1 replaces the in-memory ``best_state =
copy.deepcopy(model.module.state_dict())`` at train_multi_task.py
with on-disk saves of four files per best-val event:

  * best_full.pt            (full MultiTaskRelGT state_dict)
  * best_backbone.pt        (backbone-only state_dict for adoption)
  * backbone_meta.json      (architectural constants + node_type_map +
                             best_epoch + best_val_macro)
  * backbone_schema.pt      (col_names_dict + col_stats_dict sidecar
                             -- StatType keys / NaN floats not JSON-
                             friendly)

Tests below validate the helper's IO surface without needing to spin
up a real training loop. Heavier integration is covered by the
parity sweep.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file unmock guard for torch_geometric.
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401


def _make_fake_model(channels=8):
    """A tiny stand-in MultiTaskRelGT-like wrapper. The save helper only
    touches ``model.module.state_dict()`` and
    ``model.module.backbone.state_dict()`` so the real RelGT isn't
    needed."""
    class _Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, channels)
    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(channels, 1)
    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()
            self.head = _Head()
    class _DDPMimic:
        def __init__(self, m): self.module = m
    return _DDPMimic(_Wrapper())


def _meta_dict(channels=8):
    """Mirrors what train_multi_task._build_model returns."""
    return {
        "channels": channels,
        "num_centroids": 64,
        "num_layers": 1,
        "num_heads": 2,
        "num_neighbors": 16,
        "max_neighbor_hop": 3,
        "global_dim": channels // 2,
        "ff_dropout": 0.1,
        "attn_dropout": 0.1,
        "gt_conv_type": "full",
        "ablate": "none",
        "gnn_pe_dim": 0,
        "node_type_map": {"t::a": 0, "t::b": 1},
        "num_tasks": 1,
        "task_type_ids": [0],
        "_col_names_dict": {"t::a": {}, "t::b": {}},
        "_col_stats_dict": {"t::a": {}, "t::b": {}},
    }


def test_save_best_checkpoint_writes_all_four_files(tmp_path):
    from train_multi_task import _save_best_checkpoint
    model = _make_fake_model()
    _save_best_checkpoint(
        model, str(tmp_path), _meta_dict(),
        best_epoch=3, best_metric=0.5,
    )
    for name in ("best_full.pt", "best_backbone.pt",
                 "backbone_meta.json", "backbone_schema.pt"):
        p = tmp_path / name
        assert p.exists(), f"missing {name}"
        assert p.stat().st_size > 0, f"{name} is empty"


def test_meta_json_schema_complete(tmp_path):
    """Meta JSON must contain every architectural-constant key the
    Phase-2 load_backbone helper reads. Missing keys would silently
    become defaults at load time and drift the loaded backbone."""
    from train_multi_task import _save_best_checkpoint
    model = _make_fake_model()
    _save_best_checkpoint(
        model, str(tmp_path), _meta_dict(),
        best_epoch=7, best_metric=0.812,
    )
    with open(tmp_path / "backbone_meta.json") as f:
        meta = json.load(f)
    required = {
        "channels", "num_centroids", "num_layers", "num_heads",
        "num_neighbors", "max_neighbor_hop", "global_dim",
        "ff_dropout", "attn_dropout", "gt_conv_type", "ablate",
        "gnn_pe_dim", "node_type_map", "best_epoch", "best_val_macro",
    }
    missing = required - set(meta.keys())
    assert missing == set(), f"meta JSON missing keys: {missing}"
    # Recorded best_epoch / best_val_macro must round-trip exactly.
    assert meta["best_epoch"] == 7
    assert meta["best_val_macro"] == pytest.approx(0.812)
    # Private (underscore-prefixed) keys must NOT leak into JSON --
    # those are the schema dicts that ride in the .pt sidecar.
    private_in_json = [k for k in meta if k.startswith("_")]
    assert private_in_json == [], (
        f"private keys leaked into JSON: {private_in_json}"
    )


def test_schema_pt_roundtrip_with_stype_keys(tmp_path):
    """col_stats_dict uses StatType enum keys and may carry NaN/inf
    floats from degenerate columns. Validates the .pt sidecar
    preserves both."""
    from torch_frame.data.stats import StatType
    from train_multi_task import _save_best_checkpoint
    meta = _meta_dict()
    meta["_col_stats_dict"] = {
        "t::a": {
            "x": {StatType.MEAN: 1.0, StatType.STD: 2.0},
            "y": {StatType.MEAN: float("nan"), StatType.STD: 1.0},
        },
        "t::b": {
            "z": {StatType.COUNT: 5},
            "e": {StatType.EMB_DIM: 64},
        },
    }
    _save_best_checkpoint(
        _make_fake_model(), str(tmp_path), meta,
        best_epoch=0, best_metric=0.0,
    )
    schema = torch.load(
        tmp_path / "backbone_schema.pt", weights_only=False,
    )
    cs = schema["col_stats_dict"]
    assert StatType.MEAN in cs["t::a"]["x"]
    assert cs["t::a"]["x"][StatType.MEAN] == 1.0
    # NaN comparison: assert the field IS NaN.
    assert cs["t::a"]["y"][StatType.MEAN] != cs["t::a"]["y"][StatType.MEAN]
    assert cs["t::b"]["z"][StatType.COUNT] == 5
    assert cs["t::b"]["e"][StatType.EMB_DIM] == 64


def test_overwrite_on_new_best(tmp_path):
    """Each best-val event overwrites the prior files (no numbered
    history). Otherwise disk would balloon over a long training run
    and the load path would have to pick a 'latest'."""
    from train_multi_task import _save_best_checkpoint
    model = _make_fake_model()
    _save_best_checkpoint(
        model, str(tmp_path), _meta_dict(),
        best_epoch=1, best_metric=0.4,
    )
    first_meta = json.loads((tmp_path / "backbone_meta.json").read_text())
    assert first_meta["best_epoch"] == 1

    _save_best_checkpoint(
        model, str(tmp_path), _meta_dict(),
        best_epoch=5, best_metric=0.55,
    )
    second_meta = json.loads((tmp_path / "backbone_meta.json").read_text())
    assert second_meta["best_epoch"] == 5
    assert second_meta["best_val_macro"] == pytest.approx(0.55)
    # Single canonical name; no numbered variants in the directory.
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "backbone_meta.json", "backbone_schema.pt",
        "best_backbone.pt", "best_full.pt",
    ]


def test_no_in_memory_deepcopy_in_train_multi_task():
    """Catch a future revert: the old ``best_state =
    copy.deepcopy(model.module.state_dict())`` pattern should be gone
    from train_multi_task.py. Any matching string would mean someone
    re-introduced the in-memory copy."""
    src = (
        Path(__file__).resolve().parent.parent / "train_multi_task.py"
    ).read_text()
    # Match: best_state ASSIGNED to copy.deepcopy(...). Excludes the
    # post-load-from-disk usage at the end of the loop where
    # ``best_state = torch.load(...)`` is intentional.
    pattern = re.compile(
        r"best_state\s*=\s*copy\.deepcopy\(",
    )
    assert pattern.search(src) is None, (
        "train_multi_task.py still contains the in-memory "
        "best_state = copy.deepcopy(...) pattern"
    )


def test_save_then_load_roundtrip_state_dict(tmp_path):
    """Round-trip integrity check on the full state_dict: save +
    load returns bit-equal tensors. Catches any FP-precision or
    pickle-serialization bug introduced by the save/load split."""
    from train_multi_task import _save_best_checkpoint
    model = _make_fake_model()
    sd_before = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
    _save_best_checkpoint(
        model, str(tmp_path), _meta_dict(),
        best_epoch=0, best_metric=0.0,
    )
    sd_after = torch.load(
        tmp_path / "best_full.pt", map_location="cpu", weights_only=False,
    )
    assert set(sd_before.keys()) == set(sd_after.keys())
    for k in sd_before:
        assert torch.equal(sd_before[k], sd_after[k]), f"mismatch on {k}"
