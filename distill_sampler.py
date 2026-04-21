import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DistillSampler(nn.Module):
    """Predicts RelGT's last-layer seed-to-candidate pre-softmax attention logits
    from the concatenated [type, hop, time, tfs] base embeddings.

    Single projection `Ws` shared between seed and candidates; score is a dot
    product in the projected space. At inference, scores feed Gumbel-Top-K
    (without replacement) with temperature `sample_temp`.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.Ws = nn.Linear(embed_dim, hidden_dim, bias=False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Ws.weight)

    def forward(self, base_concat: Tensor) -> Tensor:
        # base_concat: [B, K, embed_dim]. Seed at index 0, candidates at 1..K-1.
        proj = self.Ws(base_concat)                                # [B, K, hidden_dim]
        seed = proj[:, 0:1, :]                                     # [B, 1, hidden_dim]
        q_imp = torch.bmm(seed, proj.transpose(1, 2)).squeeze(1)   # [B, K]
        return q_imp

    @staticmethod
    def distillation_loss(q_imp: Tensor, teacher_logits: Tensor) -> Tensor:
        # Drop self-entry at column 0 on both sides — self-attention is spuriously high
        # and not informative for sampling.
        return F.mse_loss(q_imp[:, 1:], teacher_logits[:, 1:])


def gumbel_top_k(
    q_imp: Tensor,
    k: int,
    temperature: float = 1.0,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select k candidate indices from q_imp's columns [1, K), excluding the seed.

    Returns indices in [1, K) of shape [B, k]. The seed at column 0 is implicit
    and must be prepended by the caller when materializing the curated subgraph.
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
