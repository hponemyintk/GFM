"""Tests for resume-from-checkpoint (DDP multi-task pretraining).

Covers the helpers added to ``train_multi_task.py``:

  * ``_resume_checkpoint_path`` -- 'auto' / explicit-path / None resolution
  * ``_save_resume_checkpoint`` / ``_load_resume_checkpoint`` -- model +
    optimizer + loss_fn + per-rank RNG round-trip, atomic write
  * ``_rng_state_dict`` / ``_load_rng_state_dict`` -- RNG snapshot/restore
  * ``_prune_resume_checkpoints`` -- keep-newest-N (with RNG sidecars)

Unit tests run CPU-only with ``dist`` not initialized (the save helper
falls back to ``world_size = 1`` then). An opt-in ``@pytest.mark.slow``
2-process gloo smoke exercises the real DDP path: train 2 "epochs",
resume, train to 4, assert no crash + ``per_epoch_macro`` carries through.
"""

from __future__ import annotations

import os
import socket
import sys
from unittest.mock import MagicMock as _MagicMock

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cross-file unmock guard for torch_geometric (mirrors test_checkpoint_to_disk).
if isinstance(sys.modules.get("torch_geometric"), _MagicMock):
    for _k in [k for k in sys.modules if k.startswith("torch_geometric")]:
        del sys.modules[_k]
    import torch_geometric  # noqa: F401


# ----------------------------------------------------------------- fixtures
class _DDPMimic:
    """Stand-in for the DDP wrapper: exposes ``.module``."""
    def __init__(self, m):
        self.module = m

    def parameters(self):
        return self.module.parameters()


def _make_model_optim_loss(aggregation="uncertainty"):
    """Tiny model + Adam (over model + loss params) + MultiTaskLoss."""
    from losses.multi_task_loss import MultiTaskLoss
    torch.manual_seed(0)
    inner = torch.nn.Sequential(
        torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2),
    )
    model = _DDPMimic(inner)
    loss_fn = MultiTaskLoss(num_tasks=2, aggregation=aggregation)
    optim = torch.optim.Adam(
        list(model.module.parameters()) + list(loss_fn.parameters()), lr=1e-2,
    )
    return model, optim, loss_fn


def _step_a_few(model, optim, loss_fn, n=3):
    """Run a few optimizer steps so Adam moments + log_sigma2 are non-trivial."""
    for _ in range(n):
        optim.zero_grad()
        x = torch.randn(5, 4)
        out = model.module(x).sum()
        # touch loss_fn params too (only present in 'uncertainty' mode)
        for p in loss_fn.parameters():
            out = out + p.sum()
        out.backward()
        optim.step()


# ------------------------------------------------------- _resume_checkpoint_path
def test_resume_checkpoint_path_none(tmp_path):
    from train_multi_task import _resume_checkpoint_path
    assert _resume_checkpoint_path(str(tmp_path), None) is None


def test_resume_checkpoint_path_auto_empty(tmp_path):
    from train_multi_task import _resume_checkpoint_path
    # No checkpoints yet -> auto returns None (fresh start, not an error).
    assert _resume_checkpoint_path(str(tmp_path), "auto") is None


def test_resume_checkpoint_path_auto_picks_newest(tmp_path):
    from train_multi_task import _resume_checkpoint_path
    for e in (1, 2, 10):
        (tmp_path / f"checkpoint_epoch_{e:04d}.pt").write_bytes(b"x")
    got = _resume_checkpoint_path(str(tmp_path), "auto")
    assert got == str(tmp_path / "checkpoint_epoch_0010.pt"), (
        "zero-padded epoch => lexicographic max must be the highest epoch"
    )


def test_resume_checkpoint_path_explicit(tmp_path):
    from train_multi_task import _resume_checkpoint_path
    p = tmp_path / "checkpoint_epoch_0007.pt"
    p.write_bytes(b"x")
    assert _resume_checkpoint_path(str(tmp_path), str(p)) == str(p)
    with pytest.raises(FileNotFoundError):
        _resume_checkpoint_path(str(tmp_path), str(tmp_path / "nope.pt"))


# ------------------------------------------------------------ RNG round-trip
def test_rng_state_roundtrip():
    import numpy as np
    import random as _random
    from train_multi_task import _rng_state_dict, _load_rng_state_dict
    snap = _rng_state_dict()
    a = (torch.randn(4).tolist(), np.random.rand(4).tolist(), [_random.random() for _ in range(4)])
    _load_rng_state_dict(snap)
    b = (torch.randn(4).tolist(), np.random.rand(4).tolist(), [_random.random() for _ in range(4)])
    assert a == b
    if torch.cuda.is_available():
        snap2 = _rng_state_dict()
        ca = torch.randn(4, device="cuda").tolist()
        _load_rng_state_dict(snap2)
        cb = torch.randn(4, device="cuda").tolist()
        assert ca == cb


# ----------------------------------------------- save / load full round-trip
def test_save_load_roundtrip(tmp_path):
    from train_multi_task import _save_resume_checkpoint, _load_resume_checkpoint

    model, optim, loss_fn = _make_model_optim_loss(aggregation="uncertainty")
    _step_a_few(model, optim, loss_fn, n=3)

    model_sd = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
    opt_sd = optim.state_dict()
    # snapshot a couple of Adam moment tensors for comparison
    first_pid = next(iter(opt_sd["state"]))
    exp_avg_before = opt_sd["state"][first_pid]["exp_avg"].clone()
    log_sigma2_before = loss_fn.log_sigma2.detach().clone()

    per_epoch = {1: 0.40, 2: 0.71, 3: 0.55}
    _save_resume_checkpoint(
        model, optim, loss_fn, str(tmp_path),
        epoch=3, global_step=42, best_macro=0.71, best_epoch=2,
        best_ckpt_written=True, per_epoch_macro=per_epoch, rank=0,
    )
    ckpt_file = tmp_path / "checkpoint_epoch_0003.pt"
    assert ckpt_file.exists() and ckpt_file.stat().st_size > 0
    assert (tmp_path / "rng_state_epoch0003_rank0.pt").exists()
    # atomic write left no temp file behind
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    # Fresh objects, then restore.
    model2, optim2, loss_fn2 = _make_model_optim_loss(aggregation="uncertainty")
    # perturb so the load is actually doing something
    with torch.no_grad():
        for p in model2.module.parameters():
            p.add_(1.0)
    meta = _load_resume_checkpoint(str(ckpt_file), model2, optim2, loss_fn2,
                                   str(tmp_path), rank=0)
    assert meta == {
        "epoch": 3, "global_step": 42, "best_macro": 0.71, "best_epoch": 2,
        "best_ckpt_written": True, "per_epoch_macro": {1: 0.40, 2: 0.71, 3: 0.55},
    }
    for k, v in model2.module.state_dict().items():
        assert torch.equal(v, model_sd[k]), f"model param {k} did not round-trip"
    assert torch.equal(loss_fn2.log_sigma2.detach(), log_sigma2_before)
    # optimizer Adam state round-tripped
    opt2_sd = optim2.state_dict()
    assert torch.equal(opt2_sd["state"][first_pid]["exp_avg"], exp_avg_before)


def test_save_load_no_uncertainty_params(tmp_path):
    """aggregation='none' -> loss_fn has empty state_dict; round-trip still works."""
    from train_multi_task import _save_resume_checkpoint, _load_resume_checkpoint
    model, optim, loss_fn = _make_model_optim_loss(aggregation="none")
    assert list(loss_fn.parameters()) == []
    _step_a_few(model, optim, loss_fn, n=2)
    _save_resume_checkpoint(
        model, optim, loss_fn, str(tmp_path),
        epoch=1, global_step=2, best_macro=float("-inf"), best_epoch=0,
        best_ckpt_written=False, per_epoch_macro={1: 0.1}, rank=0,
    )
    model2, optim2, loss_fn2 = _make_model_optim_loss(aggregation="none")
    meta = _load_resume_checkpoint(str(tmp_path / "checkpoint_epoch_0001.pt"),
                                   model2, optim2, loss_fn2, str(tmp_path), rank=0)
    assert meta["epoch"] == 1 and meta["best_ckpt_written"] is False


def test_load_rejects_bad_format_version(tmp_path):
    from train_multi_task import _load_resume_checkpoint
    model, optim, loss_fn = _make_model_optim_loss(aggregation="none")
    bad = tmp_path / "checkpoint_epoch_0001.pt"
    torch.save({"format_version": 999, "epoch": 1}, bad)
    with pytest.raises(ValueError):
        _load_resume_checkpoint(str(bad), model, optim, loss_fn, str(tmp_path), rank=0)


# --------------------------------------------------------------- pruning
def test_prune_resume_checkpoints(tmp_path):
    from train_multi_task import _prune_resume_checkpoints
    for e in range(1, 6):
        (tmp_path / f"checkpoint_epoch_{e:04d}.pt").write_bytes(b"x")
        (tmp_path / f"rng_state_epoch{e:04d}_rank0.pt").write_bytes(b"x")
        (tmp_path / f"rng_state_epoch{e:04d}_rank1.pt").write_bytes(b"x")
    _prune_resume_checkpoints(str(tmp_path), keep_n=3)
    remaining_ckpts = sorted(p.name for p in tmp_path.glob("checkpoint_epoch_*.pt"))
    assert remaining_ckpts == [
        "checkpoint_epoch_0003.pt", "checkpoint_epoch_0004.pt", "checkpoint_epoch_0005.pt",
    ]
    remaining_rng = sorted(p.name for p in tmp_path.glob("rng_state_epoch*_rank*.pt"))
    assert remaining_rng == [
        "rng_state_epoch0003_rank0.pt", "rng_state_epoch0003_rank1.pt",
        "rng_state_epoch0004_rank0.pt", "rng_state_epoch0004_rank1.pt",
        "rng_state_epoch0005_rank0.pt", "rng_state_epoch0005_rank1.pt",
    ]
    # keep_n=0 and keep_n>=count are no-ops
    _prune_resume_checkpoints(str(tmp_path), keep_n=0)
    assert len(list(tmp_path.glob("checkpoint_epoch_*.pt"))) == 3
    _prune_resume_checkpoints(str(tmp_path), keep_n=99)
    assert len(list(tmp_path.glob("checkpoint_epoch_*.pt"))) == 3


# ---------------------------------------------- opt-in DDP integration smoke
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ddp_worker(rank, world_size, port, out_dir, epochs, resume, result_path):
    import torch.distributed as dist
    import torch.nn as nn
    from losses.multi_task_loss import MultiTaskLoss
    from train_multi_task import (
        _resume_checkpoint_path, _save_resume_checkpoint,
        _load_resume_checkpoint, _prune_resume_checkpoints,
    )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(42 + rank)
        model = nn.parallel.DistributedDataParallel(nn.Linear(4, 2))
        loss_fn = MultiTaskLoss(num_tasks=2, aggregation="none")
        optim = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3,
        )
        os.makedirs(out_dir, exist_ok=True)
        g_rank = dist.get_rank()

        resume_path = _resume_checkpoint_path(out_dir, resume)
        start_epoch, global_step = 1, 0
        best_macro, best_epoch, best_written = float("-inf"), 0, False
        per_epoch_macro = {}
        if resume_path is not None:
            m = _load_resume_checkpoint(resume_path, model, optim, loss_fn, out_dir, g_rank)
            start_epoch = m["epoch"] + 1
            global_step = m["global_step"]
            best_macro, best_epoch, best_written = m["best_macro"], m["best_epoch"], m["best_ckpt_written"]
            per_epoch_macro = dict(m["per_epoch_macro"])
            dist.barrier()

        for epoch in range(start_epoch, epochs + 1):
            for _ in range(2):
                optim.zero_grad()
                model(torch.randn(3, 4)).sum().backward()
                optim.step()
                global_step += 1
            per_epoch_macro[epoch] = float(epoch)  # stub macro
            dist.barrier()
            _save_resume_checkpoint(
                model, optim, loss_fn, out_dir,
                epoch=epoch, global_step=global_step, best_macro=best_macro,
                best_epoch=best_epoch, best_ckpt_written=best_written,
                per_epoch_macro=per_epoch_macro, rank=g_rank,
            )
            if g_rank == 0:
                _prune_resume_checkpoints(out_dir, 3)
            dist.barrier()

        if g_rank == 0:
            import json
            with open(result_path, "w") as f:
                json.dump({"epochs": sorted(per_epoch_macro),
                           "global_step": global_step}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_ddp_resume_smoke(tmp_path):
    import json
    import torch.multiprocessing as mp
    out_dir = str(tmp_path / "multi_task")
    result_path = str(tmp_path / "result.json")
    ctx = mp.get_context("spawn")

    def _run(epochs, resume):
        port = _free_port()
        procs = [ctx.Process(target=_ddp_worker,
                             args=(r, 2, port, out_dir, epochs, resume, result_path))
                 for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        for p in procs:
            assert p.exitcode == 0, f"ddp worker exited {p.exitcode}"
        with open(result_path) as f:
            return json.load(f)

    r1 = _run(epochs=2, resume=None)
    assert r1["epochs"] == [1, 2]
    assert {f.name for f in (tmp_path / "multi_task").glob("checkpoint_epoch_*.pt")} == {
        "checkpoint_epoch_0001.pt", "checkpoint_epoch_0002.pt",
    }

    r2 = _run(epochs=4, resume="auto")
    assert r2["epochs"] == [1, 2, 3, 4], "per_epoch_macro must carry through resume"
    assert r2["global_step"] > r1["global_step"]
    # keep_checkpoints=3 default: only epochs 2,3,4 remain
    assert {f.name for f in (tmp_path / "multi_task").glob("checkpoint_epoch_*.pt")} == {
        "checkpoint_epoch_0002.pt", "checkpoint_epoch_0003.pt", "checkpoint_epoch_0004.pt",
    }
