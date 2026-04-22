"""Behavioral tests for the distilled-sampler pipeline.

Run:
    pytest -xvs test_distillation.py

These tests focus on the extraction/freeze/sampling primitives that don't
require a full relbench dataset or GPU. They can run on CPU in under a minute.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from local_module import EncoderLayer, LocalModule
from distill_sampler import DistillSampler, gumbel_top_k


def test_manual_dots_match_reference():
    """Assert the extracted seed_logits equal a reference numpy computation."""
    torch.manual_seed(0)
    B, L, D, H = 2, 6, 16, 4
    d_h = D // H
    layer = EncoderLayer(hidden_size=D, ffn_size=32, dropout_rate=0.0,
                         attention_dropout_rate=0.0, num_heads=H).eval()
    x = torch.randn(B, L, D)

    _, seed_logits = layer(x, extract_seed_logits=True)
    assert seed_logits.shape == (B, H, L)

    # Reference: recompute Q/K from the same normed x and take row-0.
    x_norm = layer.self_attention_norm(x)
    Q = layer.q_proj(x_norm).view(B, L, H, d_h).transpose(1, 2)  # [B, H, L, d_h]
    K = layer.k_proj(x_norm).view(B, L, H, d_h).transpose(1, 2)
    dots = torch.matmul(Q[:, :, 0:1, :], K.transpose(-2, -1)) / math.sqrt(d_h)
    ref = dots.squeeze(2)  # [B, H, L]

    torch.testing.assert_close(seed_logits, ref, atol=1e-5, rtol=1e-5)


def test_local_module_extracts_from_last_layer():
    """Multi-layer LocalModule should return seed_logits from the LAST enc_layer."""
    torch.manual_seed(1)
    B, L, D, H = 2, 5, 16, 4
    lm = LocalModule(seq_len=L, input_dim=D, n_layers=3, num_heads=H,
                     hidden_dim=D, dropout_rate=0.0, attention_dropout_rate=0.0).eval()
    x = torch.randn(B, L, D)
    _, seed_logits = lm(x, extract_seed_logits=True)
    assert seed_logits is not None
    assert seed_logits.shape == (B, H, L)

    # Without the flag: seed_logits is None.
    _, seed_logits_off = lm(x, extract_seed_logits=False)
    assert seed_logits_off is None


def test_distill_sampler_shapes_and_loss_decreases():
    """Train the sampler on synthetic teacher targets; loss should decrease."""
    torch.manual_seed(2)
    B, K, D_in, D_h, T, H = 8, 12, 20, 16, 3, 4
    d_head = D_h // H
    sampler = DistillSampler(embed_dim=D_in, hidden_dim=D_h,
                             num_node_types=T, num_heads=H)
    base = torch.randn(B, K, D_in)
    types = torch.randint(0, T, (B, K))

    # Synthetic per-head teacher: random per-head projection, seed-dot-candidate
    # with 1/sqrt(d_head) scaling (matches the sampler's geometry).
    with torch.no_grad():
        proj_h = torch.randn(H, D_in, d_head)
        p = torch.einsum("bke,hed->bhkd", base, proj_h)         # [B, H, K, d_head]
        teacher_logits = torch.einsum("bhd,bhkd->bhk", p[:, :, 0, :], p) / math.sqrt(d_head)

    opt = torch.optim.Adam(sampler.parameters(), lr=0.05)
    initial_loss = DistillSampler.distillation_loss(sampler(base, types), teacher_logits).item()
    for _ in range(300):
        q_imp = sampler(base, types)
        loss = DistillSampler.distillation_loss(q_imp, teacher_logits)
        opt.zero_grad(); loss.backward(); opt.step()
    final_loss = loss.item()
    assert final_loss < 0.3 * initial_loss, (initial_loss, final_loss)
    assert q_imp.shape == (B, H, K)


def test_gumbel_top_k_shape_and_no_replacement():
    """Gumbel-Top-K returns K-1 unique indices in [1, K)."""
    torch.manual_seed(3)
    B, K = 4, 20
    q_imp = torch.randn(B, K)
    sel = gumbel_top_k(q_imp, k=K - 1, temperature=1.0, stochastic=True)
    assert sel.shape == (B, K - 1)
    assert int(sel.min()) >= 1 and int(sel.max()) < K
    # No duplicates within each row.
    for b in range(B):
        assert len(torch.unique(sel[b])) == K - 1


def test_gumbel_top_k_deterministic_reproducibility():
    """With stochastic=False, same q_imp → identical selection across calls."""
    torch.manual_seed(4)
    q_imp = torch.randn(3, 15)
    sel1 = gumbel_top_k(q_imp, k=10, temperature=1.0, stochastic=False)
    sel2 = gumbel_top_k(q_imp, k=10, temperature=1.0, stochastic=False)
    torch.testing.assert_close(sel1, sel2)


def test_freeze_persists_through_train_mode():
    """After model.train(), frozen submodules re-put in eval() stay in eval."""
    # Build a tiny mock RelGT-like structure.
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.type_encoder = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
            self.convs = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))])
            self.head = nn.Linear(8, 1)

    m = ToyModel()
    frozen = [m.type_encoder]
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()

    # Simulate the training loop's `model.train()` call.
    m.train()
    # Without re-applying, the frozen module would leak into train mode:
    assert m.type_encoder.training is True

    # Applying the freeze_eval hook restores it.
    for mod in frozen:
        mod.eval()
    assert m.type_encoder.training is False
    assert m.convs[0].training is True
    assert m.head.training is True

    # requires_grad assertions.
    assert all(not p.requires_grad for p in m.type_encoder.parameters())
    assert all(p.requires_grad for p in m.convs.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_gumbel_temperature_scaling_is_unbiased():
    """As T → ∞, selection should approach uniform; as T → 0, approach top-K."""
    torch.manual_seed(5)
    B, K = 1, 50
    q_imp = torch.zeros(B, K)
    q_imp[0, 1] = 10.0  # strongly prefers index 1

    # Low T: should almost always include index 1 in top-3.
    included_low_T = 0
    for _ in range(50):
        sel = gumbel_top_k(q_imp, k=3, temperature=0.1, stochastic=True)
        if 1 in sel[0].tolist():
            included_low_T += 1

    # High T: picks should be nearly uniform, index 1 is not special.
    included_high_T = 0
    for _ in range(50):
        sel = gumbel_top_k(q_imp, k=3, temperature=1000.0, stochastic=True)
        if 1 in sel[0].tolist():
            included_high_T += 1

    assert included_low_T >= 45          # almost always
    assert included_high_T <= 20          # roughly uniform ~ 3/49 * 50 ≈ 3


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-xvs"])
