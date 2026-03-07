"""Tests for emb_dims discovery plumbing in NeighborTfsEncoder."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch_frame
from torch_frame.data.stats import StatType
from unittest.mock import patch, MagicMock

from encoders import NeighborTfsEncoder


def _make_encoder(node_type_map, col_names_dict, col_stats_dict, channels=32):
    """Create NeighborTfsEncoder with mocked GloVe."""
    with patch("encoders.GloveTextEmbedding") as MockGlove:
        mock_instance = MagicMock()
        mock_instance.__call__ = lambda names: torch.randn(len(names), 300)
        MockGlove.return_value = mock_instance

        enc = NeighborTfsEncoder(
            channels=channels,
            node_type_map=node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
    return enc


def _get_emb_projectors(enc):
    """Get the embedding projector keys from the encoder."""
    emb_enc = enc.table_agnostic_encoder.encoders[str(torch_frame.embedding)]
    return set(emb_enc.projectors.keys())


class TestEmbDimsDiscovery:

    def test_discovers_single_dim(self):
        enc = _make_encoder(
            node_type_map={"articles": 0},
            col_names_dict={"articles": {
                torch_frame.embedding: ["title_emb"],
            }},
            col_stats_dict={"articles": {
                "title_emb": {StatType.EMB_DIM: 300},
            }},
        )
        assert _get_emb_projectors(enc) == {"300"}

    def test_discovers_multiple_dims_across_tables(self):
        enc = _make_encoder(
            node_type_map={"articles": 0, "users": 1},
            col_names_dict={
                "articles": {torch_frame.embedding: ["title_emb"]},
                "users": {torch_frame.embedding: ["bio_emb"]},
            },
            col_stats_dict={
                "articles": {"title_emb": {StatType.EMB_DIM: 300}},
                "users": {"bio_emb": {StatType.EMB_DIM: 768}},
            },
        )
        assert _get_emb_projectors(enc) == {"300", "768"}

    def test_duplicate_dims_collapsed(self):
        """Two tables with same emb dim -> one projector."""
        enc = _make_encoder(
            node_type_map={"articles": 0, "products": 1},
            col_names_dict={
                "articles": {torch_frame.embedding: ["title_emb"]},
                "products": {torch_frame.embedding: ["desc_emb"]},
            },
            col_stats_dict={
                "articles": {"title_emb": {StatType.EMB_DIM: 300}},
                "products": {"desc_emb": {StatType.EMB_DIM: 300}},
            },
        )
        assert _get_emb_projectors(enc) == {"300"}

    def test_no_embedding_columns(self):
        enc = _make_encoder(
            node_type_map={"drivers": 0},
            col_names_dict={"drivers": {
                torch_frame.numerical: ["speed"],
            }},
            col_stats_dict={"drivers": {
                "speed": {StatType.MEAN: 0.0, StatType.STD: 1.0},
            }},
        )
        assert _get_emb_projectors(enc) == set()

    def test_none_col_dicts(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict=None,
            col_stats_dict=None,
        )
        assert _get_emb_projectors(enc) == set()

    def test_missing_emb_dim_stat(self):
        """Column exists as embedding but StatType.EMB_DIM is missing."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.embedding: ["emb_col"]}},
            col_stats_dict={"t": {"emb_col": {}}},  # no EMB_DIM
        )
        assert _get_emb_projectors(enc) == set()

    def test_zero_emb_dim_skipped(self):
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {torch_frame.embedding: ["emb_col"]}},
            col_stats_dict={"t": {"emb_col": {StatType.EMB_DIM: 0}}},
        )
        assert _get_emb_projectors(enc) == set()

    def test_multiple_emb_columns_same_table(self):
        """Two embedding columns in one table with same dim."""
        enc = _make_encoder(
            node_type_map={"articles": 0},
            col_names_dict={"articles": {
                torch_frame.embedding: ["title_emb", "body_emb"],
            }},
            col_stats_dict={"articles": {
                "title_emb": {StatType.EMB_DIM: 300},
                "body_emb": {StatType.EMB_DIM: 300},
            }},
        )
        assert _get_emb_projectors(enc) == {"300"}

    def test_mixed_stypes_only_discovers_embedding(self):
        """Non-embedding stypes should not affect emb_dims."""
        enc = _make_encoder(
            node_type_map={"t": 0},
            col_names_dict={"t": {
                torch_frame.numerical: ["x"],
                torch_frame.categorical: ["c"],
                torch_frame.embedding: ["e"],
            }},
            col_stats_dict={"t": {
                "x": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                "c": {},
                "e": {StatType.EMB_DIM: 768},
            }},
        )
        assert _get_emb_projectors(enc) == {"768"}
