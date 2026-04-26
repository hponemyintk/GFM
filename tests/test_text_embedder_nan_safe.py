"""Test that ``utils.GloveTextEmbedding`` coerces NaN/None inputs to ''.

RelBench v2 tables include NaN cells in text columns; the underlying
sentence_transformers tokenizer crashes with ``'float' object has no
attribute 'split'`` if those reach it. The wrapper must coerce.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _build_embedder_with_mock_model():
    """Construct GloveTextEmbedding without loading the real model."""
    # conftest already mocked sentence_transformers; this test relies
    # on that mock being in place. Skip the heavy SentenceTransformer
    # __init__ by replacing self.model post-construction.
    from utils import GloveTextEmbedding
    e = GloveTextEmbedding.__new__(GloveTextEmbedding)
    captured = {"sentences": None}

    class _MockModel:
        def encode(self, sentences, show_progress_bar=False):
            captured["sentences"] = list(sentences)
            return np.zeros((len(sentences), 8), dtype=np.float32)

    e.model = _MockModel()
    return e, captured


def test_nan_float_coerced_to_empty_string():
    e, captured = _build_embedder_with_mock_model()
    e([float("nan"), "hello", float("nan")])
    assert captured["sentences"] == ["", "hello", ""]


def test_none_coerced_to_empty_string():
    e, captured = _build_embedder_with_mock_model()
    e([None, "world"])
    assert captured["sentences"] == ["", "world"]


def test_finite_float_coerced_to_str_not_dropped():
    e, captured = _build_embedder_with_mock_model()
    e([3.14, "hello"])
    assert captured["sentences"] == ["3.14", "hello"]


def test_returns_torch_tensor_with_correct_shape():
    e, _ = _build_embedder_with_mock_model()
    out = e([float("nan"), "x", "y"])
    import torch
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 8)


def test_progress_bar_only_for_large_inputs():
    """Big batches show progress; small ones don't (else CI logs are noisy)."""
    from utils import GloveTextEmbedding
    e = GloveTextEmbedding.__new__(GloveTextEmbedding)
    captured = {"show": None}

    class _MockModel:
        def encode(self, sentences, show_progress_bar=False):
            captured["show"] = show_progress_bar
            return np.zeros((len(sentences), 4), dtype=np.float32)
    e.model = _MockModel()

    e(["a", "b"])  # tiny
    assert captured["show"] is False

    e(["x"] * 10_000)  # at threshold
    assert captured["show"] is True
