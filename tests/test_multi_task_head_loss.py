"""Tests H1-H7 + M1-M4 for heads/multi_task_head.py + losses/multi_task_loss.py."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfm_data.task_tokens import TASK_TYPE_BINARY, TASK_TYPE_REGRESSION  # noqa: E402
from heads.multi_task_head import MultiTaskHead  # noqa: E402
from losses.multi_task_loss import MultiTaskLoss  # noqa: E402


# =============================================================== H1
def test_H1_numeric_head_huber_matches_torch_reference():
    torch.manual_seed(0)
    B, C = 8, 16
    h = torch.randn(B, C)
    head = MultiTaskHead(channels=C, num_tasks=1,
                         task_type_ids=[TASK_TYPE_REGRESSION])
    # All rows belong to the single regression task.
    task_id = torch.zeros(B, dtype=torch.long)
    pred = head(h, task_id)
    target = torch.randn(B)
    loss = F.huber_loss(pred, target, reduction="mean", delta=1.0)
    # Reference: same loss when computed manually via the per-task head.
    ref = F.huber_loss(head.task_heads[0](h).squeeze(-1), target,
                       reduction="mean", delta=1.0)
    assert torch.allclose(loss, ref)


# =============================================================== H2
def test_H2_boolean_head_BCE_matches_reference():
    torch.manual_seed(0)
    B, C = 8, 16
    h = torch.randn(B, C)
    head = MultiTaskHead(channels=C, num_tasks=1,
                         task_type_ids=[TASK_TYPE_BINARY])
    task_id = torch.zeros(B, dtype=torch.long)
    pred = head(h, task_id)
    target = torch.randint(0, 2, (B,)).float()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    ref = F.binary_cross_entropy_with_logits(
        head.task_heads[0](h).squeeze(-1), target, reduction="mean")
    assert torch.allclose(loss, ref)


# =============================================================== H3
def test_H3_per_task_batches_compose_to_RT_formula():
    """With per-task heads + single-task-per-batch, the RT formula
    (mean over per-row Huber+BCE) is realized by running two
    forward passes and averaging."""
    torch.manual_seed(1)
    B, C = 6, 8
    head = MultiTaskHead(channels=C, num_tasks=2,
                         task_type_ids=[TASK_TYPE_REGRESSION, TASK_TYPE_BINARY])
    h_reg = torch.randn(B, C, requires_grad=True)
    h_bin = torch.randn(B, C, requires_grad=True)
    pred_reg = head(h_reg, torch.zeros(B, dtype=torch.long))
    pred_bin = head(h_bin, torch.ones(B, dtype=torch.long))
    target_reg = torch.randn(B)
    target_bin = torch.randint(0, 2, (B,)).float()

    huber = F.huber_loss(pred_reg, target_reg, reduction="mean", delta=1.0)
    bce = F.binary_cross_entropy_with_logits(
        pred_bin, target_bin, reduction="mean")
    # Reference using direct per-task heads.
    ref_huber = F.huber_loss(
        head.task_heads[0](h_reg).squeeze(-1), target_reg,
        reduction="mean", delta=1.0,
    )
    ref_bce = F.binary_cross_entropy_with_logits(
        head.task_heads[1](h_bin).squeeze(-1), target_bin,
        reduction="mean",
    )
    assert torch.allclose(huber, ref_huber, atol=1e-6)
    assert torch.allclose(bce, ref_bce, atol=1e-6)


# =============================================================== H4
def test_H4_only_active_task_head_receives_gradient():
    """A single-task batch must produce nonzero grad on that task's
    head AND zero grad on every other task's head."""
    torch.manual_seed(2)
    B, C = 4, 8
    head = MultiTaskHead(
        channels=C, num_tasks=3,
        task_type_ids=[TASK_TYPE_REGRESSION, TASK_TYPE_BINARY,
                       TASK_TYPE_REGRESSION],
    )
    h = torch.randn(B, C, requires_grad=True)
    pred = head(h, torch.zeros(B, dtype=torch.long))  # task 0 only
    pred.sum().backward()
    assert head.task_heads[0].weight.grad is not None
    assert head.task_heads[0].weight.grad.abs().sum() > 0
    # Tasks 1 and 2 must have either no grad or all-zero grad.
    for ti in (1, 2):
        g = head.task_heads[ti].weight.grad
        assert g is None or g.abs().sum() == 0, (
            f"task {ti} head got grad on a task-0-only batch: {g}"
        )


# =============================================================== H5
def test_H5_backbone_grad_flows_from_each_task_call():
    """Two separate single-task forward+backward calls (one per task)
    must each contribute a gradient to their input embedding."""
    torch.manual_seed(3)
    B, C = 6, 8
    head = MultiTaskHead(
        channels=C, num_tasks=2,
        task_type_ids=[TASK_TYPE_REGRESSION, TASK_TYPE_BINARY],
    )
    h_reg = torch.randn(B, C, requires_grad=True)
    h_bin = torch.randn(B, C, requires_grad=True)
    pred_reg = head(h_reg, torch.zeros(B, dtype=torch.long))
    pred_reg.sum().backward()
    assert (h_reg.grad.norm(dim=1) > 0).all()
    pred_bin = head(h_bin, torch.ones(B, dtype=torch.long))
    pred_bin.sum().backward()
    assert (h_bin.grad.norm(dim=1) > 0).all()


# =============================================================== H8
def test_H8_embedding_extraction_path_returns_backbone_output():
    """``MultiTaskRelGT.forward(task_id=None)`` returns the raw
    ``[B, channels]`` embedding -- the downstream / TabPFN path."""
    from heads.multi_task_head import MultiTaskRelGT
    torch.manual_seed(4)
    B, C = 5, 7

    class _StubBackbone(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.proj = nn.Linear(ch + 1, ch)

        def forward(self, *a, **kw):
            # Ignore the args; return a fixed [B, C] for the test.
            return self.proj(torch.zeros(B, C + 1))

    wrapper = MultiTaskRelGT(
        backbone=_StubBackbone(C), channels=C,
        num_tasks=2, task_type_ids=[TASK_TYPE_REGRESSION, TASK_TYPE_BINARY],
    )
    out = wrapper(None, None, None, None, None, task_id=None)
    assert out.shape == (B, C), f"expected [B={B}, C={C}], got {tuple(out.shape)}"
    # Sanity: with task_id provided we get [B] predictions, not [B, C].
    out2 = wrapper(None, None, None, None, None,
                   task_id=torch.zeros(B, dtype=torch.long))
    assert out2.shape == (B,), f"expected [B], got {tuple(out2.shape)}"


# =============================================================== H6
def test_H6_target_normalize_denormalize_roundtrip():
    """TaskTokens.normalize_target/denormalize_pred are inverses."""
    from gfm_data.task_tokens import TaskTokens
    # Skip TF/cache wiring; just exercise the helpers directly.
    tt_obj = TaskTokens.__new__(TaskTokens)
    tt_obj.task_type_id = TASK_TYPE_REGRESSION
    tt_obj.target_mean = 5.0
    tt_obj.target_std = 2.0
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z = tt_obj.normalize_target(y)
    yh = tt_obj.denormalize_pred(z)
    assert torch.allclose(y, yh, atol=1e-6)


# =============================================================== H7
def test_H7_target_stats_fitted_only_on_train_split():
    """Sibling adopt_target_stats(train) on val should not refit on val."""
    from gfm_data.task_tokens import TaskTokens
    val = TaskTokens.__new__(TaskTokens)
    val.task_type_id = TASK_TYPE_REGRESSION
    val.target_mean = None
    val.target_std = None
    val.adopt_target_stats(mean=0.0, std=10.0)
    assert val.target_mean == 0.0 and val.target_std == 10.0
    y = torch.tensor([100.0, -100.0])
    # Normalized using train stats, NOT val stats.
    z = val.normalize_target(y)
    assert torch.allclose(z, torch.tensor([10.0, -10.0]))


# =============================================================== M1, M3, M4
def test_M1_per_task_metric_split():
    """MultiTaskLoss.info reports per-task means; aggregation modes work."""
    pred = torch.tensor([0.1, 0.2, 1.0, 1.1, 0.5, 0.6])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.6])
    tt = torch.tensor([TASK_TYPE_REGRESSION] * 4 + [TASK_TYPE_REGRESSION] * 2,
                      dtype=torch.long)
    task_id = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)

    loss_fn = MultiTaskLoss(num_tasks=2, aggregation="per_task_mean")
    loss, info = loss_fn(pred, target, task_id, tt)

    # Manual per-task means then average over 2 tasks.
    huber_all = F.huber_loss(pred, target, reduction="none", delta=1.0)
    t0 = huber_all[task_id == 0].mean()
    t1 = huber_all[task_id == 1].mean()
    expected = (t0 + t1) / 2
    assert torch.allclose(loss, expected, atol=1e-6)
    # info preserves the per-task numbers.
    assert torch.allclose(info["task_0_loss"], t0, atol=1e-6)
    assert torch.allclose(info["task_1_loss"], t1, atol=1e-6)
    assert int(info["task_0_count"].item()) == 4
    assert int(info["task_1_count"].item()) == 2


def test_M2_uncertainty_weighting_has_learnable_log_sigma():
    loss_fn = MultiTaskLoss(num_tasks=2, aggregation="uncertainty")
    assert hasattr(loss_fn, "log_sigma2")
    assert loss_fn.log_sigma2.requires_grad
    assert loss_fn.log_sigma2.shape == (2,)
    pred = torch.randn(8)
    target = torch.randn(8)
    tt = torch.full((8,), TASK_TYPE_REGRESSION, dtype=torch.long)
    task_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    loss, _ = loss_fn(pred, target, task_id, tt)
    # Differentiable wrt log_sigma2.
    grad = torch.autograd.grad(loss, loss_fn.log_sigma2, retain_graph=True)[0]
    assert grad is not None and grad.shape == (2,)


def test_M3_fixed_weighted_aggregation():
    loss_fn = MultiTaskLoss(num_tasks=2, aggregation="fixed:1.0,3.0")
    pred = torch.tensor([0.1, 0.1, 0.5, 0.5])
    target = torch.tensor([0.0, 0.0, 0.0, 0.0])
    tt = torch.full((4,), TASK_TYPE_REGRESSION, dtype=torch.long)
    task_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss, _ = loss_fn(pred, target, task_id, tt)
    huber = F.huber_loss(pred, target, reduction="none", delta=1.0)
    t0 = huber[:2].mean(); t1 = huber[2:].mean()
    expected = (1.0 * t0 + 3.0 * t1) / (1.0 + 3.0)
    assert torch.allclose(loss, expected, atol=1e-6)
