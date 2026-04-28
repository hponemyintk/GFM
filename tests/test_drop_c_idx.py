"""Regression tests for PR 1.1 -- drop the ``c_idx`` buffer and source
the global-attention popularity bias from the VQ's own
``_ema_cluster_size`` instead.

Background: ``c_idx`` was a ``[num_nodes_total]`` int64 buffer that
recorded each training-set node's most-recent codebook assignment, then
exposed the histogram-over-the-buffer to ``RelGTLayer.global_forward``
as a popularity bias on attention logits. The buffer's size pinned
``num_nodes`` into the model architecture (blocking adoption on a new
dataset with new node ids) and its labels went stale relative to the
drifting codebook (each node-id slot was last written at some past
step, but the bias was evaluated against the *current* codebook). The
VQ already maintains a drift-aware, DDP-synced, Laplace-smoothed EMA
over per-batch cluster sizes (``codebook.py:_ema_cluster_size``); the
fix is to read the bias from there and delete ``c_idx``.

Tests in this module:

  * ``test_no_c_idx_attribute_after_refactor`` -- buffer must not
    appear anywhere in the model module tree.
  * ``test_relgt_init_signature_drops_num_nodes`` -- ``RelGT`` and
    ``RelGTLayer`` constructors must no longer accept ``num_nodes``.
  * ``test_train_multi_task_drops_num_nodes_total`` -- the multi-task
    builder must not recompute ``num_nodes_total`` (an ex-side-effect
    of the buffer that's now meaningless).
  * ``test_global_forward_uses_ema_cluster_size`` -- changing
    ``_ema_cluster_size`` must change the layer's output (proves the
    bias actually flows through; an unbiased layer would be
    insensitive to it).
"""

from __future__ import annotations

import inspect
import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tests/conftest.py mocks torch_geometric for M1-Mac compatibility
# (the JIT-compiled C++ extensions can hang there). On Linux the real
# package imports cleanly, and we need the real ``Linear`` /
# ``MLP`` /``GINConv`` /``PositionalEncoding`` symbols to construct a
# functional RelGTLayer. Pop the mocks so subsequent imports resolve
# to the real package -- but only if it's still the MagicMock. If a
# sibling test file (e.g. test_register_dataset.py) already swapped
# in the real torch_geometric, popping again would trigger a DataPipe
# re-registration which torch refuses ("batch_graphs already taken").
from unittest.mock import MagicMock as _MagicMock
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401  -- re-import the real one


def _build_layer(num_centroids: int = 8, channels: int = 16):
    """Construct a minimal ``RelGTLayer`` with global attention enabled.

    Test-only sizes (channels=16, heads=2, num_centroids=8) -- no real
    workload, just enough to exercise the global-forward path.
    """
    from model import RelGTLayer
    return RelGTLayer(
        in_channels=channels,
        out_channels=channels,
        local_num_layers=1,
        global_dim=channels // 2,
        heads=2,
        ff_dropout=0.0,
        attn_dropout=0.0,
        conv_type="global",  # forces VQ + global-attention path
        num_centroids=num_centroids,
        sample_node_len=8,
    )


def test_no_c_idx_attribute_after_refactor():
    """Walk every submodule + state_dict key; ``c_idx`` must not exist
    anywhere. Catches any leftover buffer registration we missed."""
    layer = _build_layer()
    for name, mod in layer.named_modules():
        assert not hasattr(mod, "c_idx") or not torch.is_tensor(
            getattr(mod, "c_idx", None)
        ), f"residual c_idx tensor on submodule {name!r}"
    sd_keys = list(layer.state_dict().keys())
    matching = [k for k in sd_keys if "c_idx" in k]
    assert matching == [], f"c_idx still in state_dict: {matching}"


def test_relgt_init_signature_drops_num_nodes():
    """``num_nodes`` must not be a parameter of either constructor.
    Pre-PR-1.1 both ``RelGTLayer`` and ``RelGT`` took it; post-PR-1.1
    neither does. Reflective check so a future re-introduction trips
    this even if no caller wires it up yet."""
    from model import RelGT, RelGTLayer
    layer_params = inspect.signature(RelGTLayer.__init__).parameters
    relgt_params = inspect.signature(RelGT.__init__).parameters
    assert "num_nodes" not in layer_params, (
        f"RelGTLayer.__init__ still has num_nodes param: "
        f"{list(layer_params)}"
    )
    assert "num_nodes" not in relgt_params, (
        f"RelGT.__init__ still has num_nodes param: "
        f"{list(relgt_params)}"
    )


def test_train_multi_task_drops_num_nodes_total():
    """``num_nodes_total`` was the input that fed ``num_nodes`` into
    ``RelGT``. With the param gone, the corresponding sum-over-caches
    and the ``_build_model`` arg must also be gone -- otherwise we
    silently keep a multi-second reduction that nobody reads."""
    train_path = os.path.join(
        os.path.dirname(__file__), "..", "train_multi_task.py"
    )
    with open(train_path) as f:
        src = f.read()
    # The exact token must not appear in the active source. Comments
    # are fine in principle, but the codebase doesn't currently hold a
    # reference one, so any hit is a real call site we missed.
    assert "num_nodes_total" not in src, (
        "train_multi_task.py still references num_nodes_total; "
        "drop the computation, the _build_model param, and the call "
        "argument."
    )
    # And no stray ``num_nodes=`` kwarg into RelGT(...) either.
    assert not re.search(r"RelGT\([^)]*num_nodes\s*=", src, flags=re.S), (
        "train_multi_task.py still passes num_nodes= into RelGT(...)"
    )


def test_global_forward_uses_ema_cluster_size():
    """Setting ``vq._ema_cluster_size`` to two different distributions
    must produce two different layer outputs -- i.e., the bias term
    actually flows through global_forward.

    We bypass ``vq.update`` (which would mutate ``_ema_cluster_size``
    each forward) by running the layer in eval mode. The popularity
    bias is read whether or not we're training, so eval is the cleaner
    probe.
    """
    torch.manual_seed(0)
    layer = _build_layer(num_centroids=8, channels=16).eval()

    B, K = 4, 1
    x_set = torch.randn(B, K, layer.in_channels)
    x = x_set[:, 0, :]
    node_indices = torch.zeros(B, dtype=torch.long)  # unused post-refactor

    # Run #1: very unbalanced -- centroid 0 should dominate the bias.
    with torch.no_grad():
        layer.vq._ema_cluster_size.copy_(
            torch.tensor([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )
        out_unbalanced = layer.global_forward(x, node_indices).clone()

    # Run #2: uniform -- bias term is constant across centroids and
    # cancels in softmax.
    with torch.no_grad():
        layer.vq._ema_cluster_size.copy_(torch.ones(8))
        out_uniform = layer.global_forward(x, node_indices).clone()

    delta = (out_unbalanced - out_uniform).abs().max().item()
    assert delta > 1e-4, (
        f"layer output insensitive to _ema_cluster_size "
        f"(max abs delta {delta:.2e}); the popularity bias is not "
        f"flowing through global_forward"
    )


def test_global_forward_bias_matches_log_ema_cluster_size():
    """Stronger correctness check: the actual bias-on-dots must equal
    ``log(ema_cluster_size.clamp(min=1))``. We probe by comparing two
    runs that differ ONLY by whose ``_ema_cluster_size`` is in place;
    the pre-softmax ``dots`` delta should be exactly the log-ratio of
    the two cluster-size vectors broadcast over (heads, B).

    Implementation: we patch ``layer.attn_fn`` to capture the dots
    tensor before the softmax is applied -- the only place the bias
    addition is observable from outside the function.
    """
    torch.manual_seed(0)
    layer = _build_layer(num_centroids=4, channels=8).eval()

    captured: dict = {}

    def _capture_softmax(t, dim=-1):
        captured.setdefault("dots", []).append(t.detach().clone())
        return torch.nn.functional.softmax(t, dim=dim)

    layer.attn_fn = _capture_softmax

    B = 2
    x = torch.randn(B, layer.in_channels)
    node_indices = torch.zeros(B, dtype=torch.long)

    sizes_a = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sizes_b = torch.tensor([10.0, 1.0, 5.0, 2.0])

    with torch.no_grad():
        layer.vq._ema_cluster_size.copy_(sizes_a)
        layer.global_forward(x, node_indices)
        layer.vq._ema_cluster_size.copy_(sizes_b)
        layer.global_forward(x, node_indices)

    dots_a, dots_b = captured["dots"]  # both [H, B, num_centroids]
    delta = (dots_b - dots_a)  # bias-only (q,k,v are deterministic on x)
    expected = (
        torch.log(sizes_b.clamp(min=1)) - torch.log(sizes_a.clamp(min=1))
    ).view(1, 1, -1)
    # ``expected`` broadcasts over (heads, batch). The actual delta
    # should be constant along those axes since the bias is added the
    # same way to every (head, query) row.
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-6), (
        f"dots delta does not match log-cluster-size delta; "
        f"got max abs err "
        f"{(delta - expected.expand_as(delta)).abs().max().item():.2e}"
    )
