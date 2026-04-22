import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DistillSampler(nn.Module):
    """Predicts RelGT's last-layer per-head pre-softmax attention logits from
    the concatenated [type, hop, time, tfs] base embeddings.

    Per node-type projection `Ws[t]: embed_dim -> hidden_dim`, split into
    `num_heads` heads of `hidden_dim/num_heads`. Score is a per-head dot
    product with `1/sqrt(d_head)` scaling, matching the teacher's attention
    convention. Returns `[B, H, K]` per-head logits.

    Signal-weighted distillation: `head_weights[h] ∝ Var(teacher head h)`
    downweights dead/collapsed heads during training MSE and in the curate-
    time reduction, so the sampler's capacity and the top-K selection aren't
    polluted by heads that converged to near-uniform attention. Weights are
    set by `set_head_weights` after a calibration pass on the teacher; the
    default is uniform (falls back to unweighted mean).
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_node_types: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must divide num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.Ws = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim, bias=False)
            for _ in range(num_node_types)
        ])
        # Normalized to mean 1; uniform by default so loss is equivalent to
        # unweighted MSE until set_head_weights() is called.
        self.register_buffer("head_weights", torch.ones(num_heads))

    def reset_parameters(self):
        for w in self.Ws:
            nn.init.xavier_uniform_(w.weight)
        self.head_weights.fill_(1.0)

    def set_head_weights(self, per_head_std: Tensor):
        """Set head weights from measured teacher per-head std (shape [H]).
        Weights proportional to std² (variance), normalized to mean 1.
        """
        assert per_head_std.shape == (self.num_heads,)
        var = per_head_std.detach().to(self.head_weights.dtype).pow(2)
        w = var / var.mean().clamp_min(1e-8)
        self.head_weights.copy_(w.to(self.head_weights.device))

    def forward(self, base_concat: Tensor, types: Tensor) -> Tensor:
        # base_concat: [B, K, embed_dim]. Seed at index 0, candidates at 1..K-1.
        # types:       [B, K] long. types[:, 0] is the seed's node-type.
        # returns:     [B, H, K] per-head pre-softmax logits.
        B, K, E = base_concat.shape
        flat_x = base_concat.reshape(B * K, E)
        flat_t = types.reshape(B * K)
        proj = torch.empty(B * K, self.hidden_dim,
                           device=flat_x.device, dtype=flat_x.dtype)
        for t_id, W in enumerate(self.Ws):
            m = flat_t == t_id
            if m.any():
                proj[m] = W(flat_x[m])
        # [B, K, H, d_h]
        proj = proj.view(B, K, self.num_heads, self.head_dim)
        seed = proj[:, 0:1, :, :]                                      # [B, 1, H, d_h]
        # Per-head seed·cand dot with sqrt(d_h) scaling (mirrors teacher).
        # [B, H, 1, d_h] x [B, H, d_h, K] -> [B, H, 1, K] -> [B, H, K]
        seed_t = seed.transpose(1, 2)                                  # [B, H, 1, d_h]
        cand_t = proj.transpose(1, 2).transpose(-2, -1)                # [B, H, d_h, K]
        q_imp = torch.matmul(seed_t, cand_t).squeeze(2) / math.sqrt(self.head_dim)
        return q_imp                                                   # [B, H, K]

    def reduce_heads(self, q_imp: Tensor) -> Tensor:
        """Signal-weighted mean across heads: `score_k = mean_h(w_h * q_h,k)`.
        Input `[B, H, K]`, returns `[B, K]`.
        """
        w = self.head_weights.to(q_imp.dtype).view(1, -1, 1)
        return (q_imp * w).mean(dim=1)

    def distillation_loss(self, q_imp: Tensor, teacher_logits: Tensor) -> Tensor:
        # Per-head MSE (drop self-entry col 0), weighted by self.head_weights.
        err = (q_imp[:, :, 1:] - teacher_logits[:, :, 1:]).pow(2)        # [B, H, K-1]
        per_head_mse = err.mean(dim=(0, 2))                              # [H]
        w = self.head_weights.to(per_head_mse.dtype)
        return (w * per_head_mse).mean()


def gumbel_top_k(
    q_imp: Tensor,
    k: int,
    temperature: float = 1.0,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select k candidate indices from q_imp's columns [1, K), excluding the seed.

    Input is `[B, K]` — a single scalar score per candidate. Callers with a
    multi-head sampler should reduce `[B, H, K] -> [B, K]` (e.g. via
    `DistillSampler.reduce_heads`) before calling. Returns indices in
    [1, K) of shape [B, k].
    """
    B, K = q_imp.shape
    candidates = q_imp[:, 1:]                                      # [B, K-1]
    if stochastic:
        if generator is not None:
            u = torch.rand(candidates.shape, device=candidates.device, generator=generator)
        else:
            u = torch.rand_like(candidates)
        gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        scores = candidates / temperature + gumbel
    else:
        scores = candidates
    _, sel = torch.topk(scores, k=k, dim=1)
    sel = sel + 1                                                  # shift past seed
    return sel
