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
    convention. Returns `[B, H, K]` per-head logits; at inference, reduce
    across heads (sum/mean) before Gumbel-Top-K.
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

    def reset_parameters(self):
        for w in self.Ws:
            nn.init.xavier_uniform_(w.weight)

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

    @staticmethod
    def distillation_loss(q_imp: Tensor, teacher_logits: Tensor) -> Tensor:
        # Drop self-entry at column 0 on both sides (across all heads).
        return F.mse_loss(q_imp[:, :, 1:], teacher_logits[:, :, 1:])


def gumbel_top_k(
    q_imp: Tensor,
    k: int,
    temperature: float = 1.0,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select k candidate indices from q_imp's columns [1, K), excluding the seed.

    Input is `[B, K]` — a single scalar score per candidate. Callers with a
    multi-head sampler should reduce `[B, H, K] -> [B, K]` (e.g. mean over
    heads) before calling. Returns indices in [1, K) of shape [B, k].
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
