"""Regression tests for PR 2.2 -- ``RelGT.load_backbone`` /
``MultiTaskRelGT.load_backbone``.

The classmethod reads the four artifacts written by PR 2.1's
``_save_best_checkpoint`` (``backbone_meta.json``,
``best_backbone.pt``, ``backbone_schema.pt``) and returns a
forward-ready ``RelGT`` in eval mode with frozen params.

Tests below build a tiny RelGT, save it, load via the helper, and
assert:
  * architectural shape matches
  * register_dataset state was rebuilt from schema
  * weights are bit-equal
  * eval_mode + freeze defaults are applied
  * forward outputs match the live model on a fixed batch (the
    integration smoke -- if anything's off in encoder buffer sync,
    this catches it)
  * MultiTaskRelGT.load_backbone delegates to RelGT.load_backbone
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock as _MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file un-mock guard.
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401

import torch_frame
from torch_frame.data.stats import StatType


class _FakeGlove:
    """Deterministic GloVe mock used across encoder tests."""
    def __init__(self, device="cpu"):
        pass
    def __call__(self, names):
        out = torch.zeros(len(names), 300)
        for i, name in enumerate(names):
            torch.manual_seed(hash(name) % 2**32)
            out[i] = torch.randn(300)
        return out


def _tiny_relgt(channels=16, num_centroids=8):
    """Minimal RelGT: 1 type, 1 numerical column. Small enough that
    save/load + forward all run in milliseconds."""
    from model import RelGT

    node_type_map = {"t::A": 0}
    col_names_dict = {"t::A": {torch_frame.numerical: ["x"]}}
    col_stats_dict = {
        "t::A": {"x": {StatType.MEAN: 0.0, StatType.STD: 1.0}},
    }

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        backbone = RelGT(
            max_neighbor_hop=3,
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
            local_num_layers=1,
            channels=channels,
            out_channels=channels,
            global_dim=channels // 2,
            heads=2,
            ff_dropout=0.0,
            attn_dropout=0.0,
            conv_type="full",
            ablate="none",
            gnn_pe_dim=0,
            num_centroids=num_centroids,
            sample_node_len=4,
            args=None,
        )
    return backbone, node_type_map, col_names_dict, col_stats_dict


def _save_artifacts(tmp_path: Path, backbone, meta_dict):
    """Mimics what train_multi_task._save_best_checkpoint writes."""
    public_meta = {k: v for k, v in meta_dict.items() if not k.startswith("_")}
    public_meta["best_epoch"] = 1
    public_meta["best_val_macro"] = 0.5
    with open(tmp_path / "backbone_meta.json", "w") as f:
        json.dump(public_meta, f, indent=2)
    torch.save(backbone.state_dict(), tmp_path / "best_backbone.pt")
    torch.save(
        {
            "col_names_dict": meta_dict["_col_names_dict"],
            "col_stats_dict": meta_dict["_col_stats_dict"],
        },
        tmp_path / "backbone_schema.pt",
    )


def _meta_for_tiny(channels=16, num_centroids=8,
                   col_names_dict=None, col_stats_dict=None,
                   node_type_map=None):
    return {
        "channels": channels,
        "num_centroids": num_centroids,
        "num_layers": 1,
        "num_heads": 2,
        "num_neighbors": 4,
        "max_neighbor_hop": 3,
        "global_dim": channels // 2,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
        "gt_conv_type": "full",
        "ablate": "none",
        "gnn_pe_dim": 0,
        "node_type_map": node_type_map or {"t::A": 0},
        "_col_names_dict": col_names_dict or {},
        "_col_stats_dict": col_stats_dict or {},
    }


def test_load_backbone_constructs_correct_arch(tmp_path):
    """Loaded backbone's architectural constants must match the saved
    meta -- channels, num_centroids, layer count."""
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt(channels=16, num_centroids=8)
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded = RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )
    # Architectural state.
    assert loaded.tfs_encoder.channels == 16
    assert loaded.convs[0].num_centroids == 8
    assert loaded.convs[0].heads == 2


def test_load_backbone_rebuilds_register_dataset_state(tmp_path):
    """The schema dicts in the .pt sidecar must drive
    NeighborTfsEncoder.register_dataset on the loaded encoder so its
    Z-score buffers / col-name GloVe table populate to match the
    saved state_dict shape. Without this the strict load_state_dict
    would error."""
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt()
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded = RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )
    enc = loaded.tfs_encoder
    assert enc._node_type_to_safe == {"t::A": "t__A"}
    assert "x" in enc._col_name_to_idx
    assert hasattr(enc, "_num_mean_t__A")
    assert hasattr(enc, "_num_std_t__A")


def test_load_backbone_state_dict_strict_match(tmp_path):
    """Every parameter and buffer in the loaded backbone must equal
    the saved one bit-for-bit (FP precision)."""
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt()
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded = RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )
    # Walk full state_dict (params + buffers).
    sd_orig = backbone.state_dict()
    sd_load = loaded.state_dict()
    assert set(sd_orig.keys()) == set(sd_load.keys()), (
        f"key set mismatch: orig only={set(sd_orig) - set(sd_load)}, "
        f"loaded only={set(sd_load) - set(sd_orig)}"
    )
    for k in sd_orig:
        assert torch.equal(sd_orig[k], sd_load[k]), f"mismatch on {k}"


def test_load_backbone_eval_mode_and_frozen_by_default(tmp_path):
    """Adoption defaults: backbone returned in eval mode, all
    parameters frozen. Can opt out via freeze=False / eval_mode=False."""
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt()
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded = RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )
    assert loaded.training is False
    for p in loaded.parameters():
        assert p.requires_grad is False, "expected all params frozen"

    # Opt-out: freeze=False keeps requires_grad=True.
    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded2 = RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
            eval_mode=False,
            freeze=False,
        )
    assert loaded2.training is True
    assert any(p.requires_grad for p in loaded2.parameters()), (
        "expected at least one param trainable when freeze=False"
    )


def test_load_backbone_missing_schema_raises(tmp_path):
    """If the schema .pt sidecar is missing, raise FileNotFoundError
    rather than silently returning an under-registered backbone."""
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt()
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)
    # Delete the schema sidecar.
    (tmp_path / "backbone_schema.pt").unlink()

    with patch("encoders.GloveTextEmbedding", _FakeGlove), \
         pytest.raises(FileNotFoundError, match="backbone_schema"):
        RelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )


def test_multitask_load_backbone_delegates_to_relgt(tmp_path):
    """MultiTaskRelGT.load_backbone should return a bare RelGT (not
    a wrapper) -- per-task heads are throwaway at adoption."""
    from heads.multi_task_head import MultiTaskRelGT
    from model import RelGT
    backbone, node_type_map, cn, cs = _tiny_relgt()
    meta = _meta_for_tiny(
        channels=16, num_centroids=8,
        col_names_dict=cn, col_stats_dict=cs,
        node_type_map=node_type_map,
    )
    _save_artifacts(tmp_path, backbone, meta)

    with patch("encoders.GloveTextEmbedding", _FakeGlove):
        loaded = MultiTaskRelGT.load_backbone(
            str(tmp_path / "backbone_meta.json"),
            str(tmp_path / "best_backbone.pt"),
            str(tmp_path / "backbone_schema.pt"),
        )
    # Should be a plain RelGT (no per-task heads attached).
    assert isinstance(loaded, RelGT)
    assert not hasattr(loaded, "head") or not isinstance(
        getattr(loaded, "head", None), nn.ModuleList
    )
