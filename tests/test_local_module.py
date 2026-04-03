"""
Tests for local_module.py: CrossAttentionLayer, EncoderLayer, LocalModule.

Tests cover:
  1. CrossAttentionLayer — shape correctness, gradient flow, mismatched Q/KV lengths
  2. EncoderLayer — shape correctness (unchanged behavior)
  3. LocalModule (cross-attention mode) — full forward pass, decoder readout shape,
     reset_parameters, pretrain_token mode
  4. LocalModule (self-attention mode) — backward compat, same behavior as before
  5. FeedForwardNetwork — shape correctness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn

from local_module import (
    CrossAttentionLayer,
    EncoderLayer,
    FeedForwardNetwork,
    LocalModule,
)


# ──────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────
B = 4           # batch size
K_INPUT = 20    # input sequence length (num_neighbors)
K_LATENT = 8    # number of latent tokens
D = 32          # hidden dimension
FFN_DIM = 64    # ffn dimension (2 * D)
NUM_HEADS = 4
DROPOUT = 0.0   # deterministic for testing
INPUT_DIM = 48  # raw input dim (before projection)


# ──────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cross_attn_layer():
    return CrossAttentionLayer(D, FFN_DIM, DROPOUT, DROPOUT, NUM_HEADS)


@pytest.fixture
def encoder_layer():
    return EncoderLayer(D, FFN_DIM, DROPOUT, DROPOUT, NUM_HEADS)


@pytest.fixture
def local_module_cross():
    return LocalModule(
        seq_len=K_INPUT,
        input_dim=INPUT_DIM,
        hidden_dim=D,
        num_heads=NUM_HEADS,
        n_layers=2,
        dropout_rate=DROPOUT,
        attention_dropout_rate=DROPOUT,
        local_attn_type="cross",
        num_latent_tokens=K_LATENT,
    )


@pytest.fixture
def local_module_self():
    return LocalModule(
        seq_len=K_INPUT,
        input_dim=INPUT_DIM,
        hidden_dim=D,
        num_heads=NUM_HEADS,
        n_layers=2,
        dropout_rate=DROPOUT,
        attention_dropout_rate=DROPOUT,
        local_attn_type="self",
    )


# ──────────────────────────────────────────────────────────────────
#  1. CrossAttentionLayer tests
# ──────────────────────────────────────────────────────────────────

class TestCrossAttentionLayer:
    def test_output_shape(self, cross_attn_layer):
        query = torch.randn(B, K_LATENT, D)
        context = torch.randn(B, K_INPUT, D)
        out = cross_attn_layer(query, context)
        assert out.shape == (B, K_LATENT, D)

    def test_single_token_query(self, cross_attn_layer):
        """Decoder readout uses a single-token query."""
        query = torch.randn(B, 1, D)
        context = torch.randn(B, K_LATENT, D)
        out = cross_attn_layer(query, context)
        assert out.shape == (B, 1, D)

    def test_gradient_flows(self, cross_attn_layer):
        query = torch.randn(B, K_LATENT, D, requires_grad=True)
        context = torch.randn(B, K_INPUT, D, requires_grad=True)
        out = cross_attn_layer(query, context)
        loss = out.sum()
        loss.backward()
        assert query.grad is not None
        assert context.grad is not None
        assert query.grad.shape == query.shape
        assert context.grad.shape == context.shape

    def test_reset_parameters(self, cross_attn_layer):
        """reset_parameters should not raise."""
        cross_attn_layer.reset_parameters()
        query = torch.randn(B, K_LATENT, D)
        context = torch.randn(B, K_INPUT, D)
        out = cross_attn_layer(query, context)
        assert out.shape == (B, K_LATENT, D)

    def test_equal_length_qkv(self, cross_attn_layer):
        """Works when query and context have the same length."""
        x = torch.randn(B, K_INPUT, D)
        out = cross_attn_layer(x, x)
        assert out.shape == (B, K_INPUT, D)


# ──────────────────────────────────────────────────────────────────
#  2. EncoderLayer tests
# ──────────────────────────────────────────────────────────────────

class TestEncoderLayer:
    def test_output_shape(self, encoder_layer):
        x = torch.randn(B, K_INPUT, D)
        out = encoder_layer(x)
        assert out.shape == (B, K_INPUT, D)

    def test_output_shape_latent_length(self, encoder_layer):
        """EncoderLayer should work on shorter (latent) sequences too."""
        x = torch.randn(B, K_LATENT, D)
        out = encoder_layer(x)
        assert out.shape == (B, K_LATENT, D)

    def test_gradient_flows(self, encoder_layer):
        x = torch.randn(B, K_INPUT, D, requires_grad=True)
        out = encoder_layer(x)
        out.sum().backward()
        assert x.grad is not None


# ──────────────────────────────────────────────────────────────────
#  3. LocalModule — cross-attention mode
# ──────────────────────────────────────────────────────────────────

class TestLocalModuleCross:
    def test_output_shape(self, local_module_cross):
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_cross.eval()
        out = local_module_cross(x)
        assert out.shape == (B, D)

    def test_pretrain_token_mode(self, local_module_cross):
        """pretrain_token=True returns full latent sequence."""
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_cross.eval()
        out = local_module_cross(x, pretrain_token=True)
        assert out.shape == (B, K_LATENT, D)

    def test_gradient_flows(self, local_module_cross):
        x = torch.randn(B, K_INPUT, INPUT_DIM, requires_grad=True)
        local_module_cross.train()
        out = local_module_cross(x)
        out.sum().backward()
        assert x.grad is not None
        # Latent tokens should also get gradients
        assert local_module_cross.latent_tokens.grad is not None

    def test_has_cross_attn_layers(self, local_module_cross):
        assert hasattr(local_module_cross, 'encoder_cross_attn')
        assert hasattr(local_module_cross, 'decoder_cross_attn')
        assert hasattr(local_module_cross, 'latent_tokens')
        assert not hasattr(local_module_cross, 'attn_layer')

    def test_latent_tokens_shape(self, local_module_cross):
        assert local_module_cross.latent_tokens.shape == (1, K_LATENT, D)

    def test_reset_parameters(self, local_module_cross):
        local_module_cross.reset_parameters()
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_cross.eval()
        out = local_module_cross(x)
        assert out.shape == (B, D)

    def test_batch_size_one(self, local_module_cross):
        x = torch.randn(1, K_INPUT, INPUT_DIM)
        local_module_cross.eval()
        out = local_module_cross(x)
        assert out.shape == (1, D)

    def test_deterministic_eval(self, local_module_cross):
        """Same input should produce same output in eval mode."""
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_cross.eval()
        out1 = local_module_cross(x)
        out2 = local_module_cross(x)
        assert torch.allclose(out1, out2, atol=1e-6)


# ──────────────────────────────────────────────────────────────────
#  4. LocalModule — self-attention mode (backward compat)
# ──────────────────────────────────────────────────────────────────

class TestLocalModuleSelf:
    def test_output_shape(self, local_module_self):
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_self.eval()
        out = local_module_self(x)
        assert out.shape == (B, D)

    def test_pretrain_token_mode(self, local_module_self):
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_self.eval()
        out = local_module_self(x, pretrain_token=True)
        assert out.shape == (B, K_INPUT, D)

    def test_has_self_attn_layers(self, local_module_self):
        assert hasattr(local_module_self, 'attn_layer')
        assert not hasattr(local_module_self, 'encoder_cross_attn')
        assert not hasattr(local_module_self, 'decoder_cross_attn')
        assert not hasattr(local_module_self, 'latent_tokens')

    def test_gradient_flows(self, local_module_self):
        x = torch.randn(B, K_INPUT, INPUT_DIM, requires_grad=True)
        local_module_self.train()
        out = local_module_self(x)
        out.sum().backward()
        assert x.grad is not None

    def test_reset_parameters(self, local_module_self):
        local_module_self.reset_parameters()
        x = torch.randn(B, K_INPUT, INPUT_DIM)
        local_module_self.eval()
        out = local_module_self(x)
        assert out.shape == (B, D)


# ──────────────────────────────────────────────────────────────────
#  5. FeedForwardNetwork tests
# ──────────────────────────────────────────────────────────────────

class TestFeedForwardNetwork:
    def test_output_shape(self):
        ffn = FeedForwardNetwork(D, FFN_DIM, DROPOUT)
        x = torch.randn(B, K_INPUT, D)
        out = ffn(x)
        assert out.shape == (B, K_INPUT, D)

    def test_output_shape_latent_length(self):
        """FFN should work on shorter (latent) sequences."""
        ffn = FeedForwardNetwork(D, FFN_DIM, DROPOUT)
        x = torch.randn(B, K_LATENT, D)
        out = ffn(x)
        assert out.shape == (B, K_LATENT, D)
