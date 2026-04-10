"""
PASS-GNN heterogeneous sampler for RelGT.

Implements the Performance-Adaptive Sampling Strategy (PASS) from:
  "Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks"
  (Yoon et al., KDD 2021, LinkedIn)

Paper formula (Equations 4-7):
    q_imp(j|i) = (Ws · h_i) · (Ws · h_j)           # single Ws, dot product
    q_rand(j|i) = 1/N(i)                             # uniform
    q̃(j|i) = as · [q_imp(j|i), q_rand(j|i)]        # as ∈ R^{1×2}
    q(j|i) = q̃(j|i) / Σ_k q̃(k|i)                 # normalize

Parameters:
    Ws ∈ R^{D(s)×D(l)}  — single projection matrix
    as ∈ R^{1×2}         — learnable 2-element attention vector

Adapted for heterogeneous heterophilic graphs by:
  - Reusing RelGT's per-type TorchFrame encoders as the per-type projection (Ws)
  - Adding learnable type embeddings for cross-type attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PASSHeteroSampler(nn.Module):
    """
    PASS learned neighbor sampler adapted for heterogeneous graphs.

    Sampling policy (from paper Eq. 4-7):
        q_imp  = dot(Ws @ h_i, Ws @ h_j)      — importance head (single Ws)
        q_rand = 1/N(i)                         — uniform head
        q̃     = as[0]*q_imp + as[1]*q_rand    — as is 2-element learnable vector
        q      = normalize(q̃)                  — valid probability distribution

    Heterogeneous adaptation:
        - Per-type projection via shared RelGT tfs_encoder (one encoder per node type)
        - Learnable type embeddings added to projected features for cross-type attention
    """

    def __init__(
        self,
        tfs_encoder: nn.Module,
        num_types: int,
        embed_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.tfs_encoder = tfs_encoder  # shared reference to RelGT's tfs_encoder
        self.num_types = num_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Type embeddings for cross-type interaction modeling
        self.type_embeddings = nn.Embedding(num_types, embed_dim)

        # Ws: single projection matrix (paper Eq. 4)
        # Maps embed_dim → hidden_dim before dot-product attention
        self.Ws = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)

        # as: 2-element learnable attention vector (paper Eq. 6)
        # as[0] weights q_imp, as[1] weights q_rand
        self.as_ = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

        # State saved for REINFORCE
        self.batch_selected = None
        self.batch_dist = None
        self.selected_embeds = None

    def own_parameters(self):
        """Return only the sampler's own parameters (not the shared tfs_encoder)."""
        return [self.Ws, self.as_, *self.type_embeddings.parameters()]

    def encode_candidates(self, scope_batch, data, device):
        """
        Encode scope candidates through the shared tfs_encoder + type_embeddings.

        Args:
            scope_batch: dict with scope_types [B, S], scope_indices [B, S]
            data: HeteroData with .tf per node type
            device: target device

        Returns:
            candidate_embeds: [B, S, embed_dim]
        """
        scope_types = scope_batch["scope_types"]      # [B, S]
        scope_indices = scope_batch["scope_indices"]  # [B, S]
        B, S = scope_types.shape

        encoded_flat = torch.zeros(B * S, self.embed_dim, device=device)

        for t_int, encoder in self.tfs_encoder.encoders.items():
            t_idx = self.tfs_encoder.node_type_map[t_int]
            mask = (scope_types == t_idx)  # [B, S]
            if not mask.any():
                continue

            local_idxs = scope_indices[mask]
            tf = data[t_int].tf[local_idxs].to(device=device)

            for st, tensor in tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tf.feat_dict[st] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            out = encoder(tf)
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)

            flat_positions = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
            encoded_flat[flat_positions] = out

        candidate_embeds = encoded_flat.reshape(B, S, self.embed_dim)
        candidate_embeds = candidate_embeds + self.type_embeddings(scope_types.to(device))
        return candidate_embeds

    def encode_seeds(self, scope_batch, data, device):
        """
        Encode seed nodes through the shared tfs_encoder + type_embeddings.

        Args:
            scope_batch: dict with seed_type [B], seed_index [B]
            data: HeteroData
            device: target device

        Returns:
            seed_embeds: [B, embed_dim]
        """
        seed_types = scope_batch["seed_type"]    # [B]
        seed_indices = scope_batch["seed_index"]  # [B]
        B = seed_types.shape[0]

        seed_embeds = torch.zeros(B, self.embed_dim, device=device)

        for t_int, encoder in self.tfs_encoder.encoders.items():
            t_idx = self.tfs_encoder.node_type_map[t_int]
            mask = (seed_types == t_idx)
            if not mask.any():
                continue

            local_idxs = seed_indices[mask]
            tf = data[t_int].tf[local_idxs].to(device=device)

            for st, tensor in tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tf.feat_dict[st] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            out = encoder(tf)
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)

            seed_embeds[mask] = out

        seed_embeds = seed_embeds + self.type_embeddings(seed_types.to(device))
        return seed_embeds

    def forward(self, seed_embeds, candidate_embeds, scope_counts, K):
        """
        PASS sampling policy (paper Eq. 4-7).

        Args:
            seed_embeds:      [B, embed_dim]   — encoded seed nodes
            candidate_embeds: [B, S, embed_dim] — encoded candidates
            scope_counts:     [B]               — valid candidate count per seed
            K:                int               — select K-1 neighbors

        Returns:
            selected: [B, K-1] — indices into scope dimension
            dist:     Categorical distribution (saved for REINFORCE)
        """
        B, S, D = candidate_embeds.shape

        # Project via Ws: h → Ws · h  (paper Eq. 4)
        # source: [B*S, embed_dim] → [B*S, hidden_dim]
        source = seed_embeds.unsqueeze(1).expand(B, S, D).reshape(B * S, D)
        target = candidate_embeds.reshape(B * S, D)

        ss = torch.mm(source, self.Ws)  # [B*S, hidden_dim]
        tt = torch.mm(target, self.Ws)  # [B*S, hidden_dim]

        # q_imp = (Ws · h_i) · (Ws · h_j)  — dot product  (paper Eq. 4)
        q_imp = torch.bmm(ss.unsqueeze(1), tt.unsqueeze(2)).squeeze(2)  # [B*S, 1]
        q_imp = q_imp.reshape(B, S)  # [B, S]

        # q_rand = 1/N(i)  (paper Eq. 5)
        q_rand = 1.0 / scope_counts.unsqueeze(1).clamp(min=1).float()  # [B, 1]
        q_rand = q_rand.expand(B, S)  # [B, S]

        # q̃ = as[0]*q_imp + as[1]*q_rand  (paper Eq. 6)
        # Use softmax on as so weights are positive and interpretable
        as_w = F.softmax(self.as_, dim=0)  # [2], sums to 1
        q_tilde = as_w[0] * q_imp + as_w[1] * q_rand  # [B, S]

        # Mask padding positions before normalization
        pad_mask = (
            torch.arange(S, device=q_tilde.device).unsqueeze(0).expand(B, S)
            >= scope_counts.unsqueeze(1)
        )
        q_tilde.masked_fill_(pad_mask, 0.0)

        # Clamp to non-negative for valid probability distribution
        q_tilde = q_tilde.clamp(min=0.0) + 1e-9

        # q = q̃ / Σ_k q̃(k|i)  (paper Eq. 7) — Categorical normalizes internally
        dist = torch.distributions.Categorical(probs=q_tilde)
        selected = dist.sample((K - 1,)).T  # [B, K-1]

        # Save for REINFORCE
        self.batch_selected = selected
        self.batch_dist = dist
        self.selected_embeds = torch.gather(
            candidate_embeds, 1,
            selected.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K-1, embed_dim]

        return selected, dist

    def reinforce_loss(self, loss_up):
        """
        PASS REINFORCE gradient (paper Theorem 4.1).

        ∇θ L = (dL/dh) · E[ ∇θ log q(j|i) · h_j ]

        Args:
            loss_up: [B, embed_dim] — dL/dh (gradient of loss w.r.t.
                     intermediate representation, i.e. x_set.grad)

        Returns:
            scalar REINFORCE loss
        """
        # log π(action) for the K-1 sampled neighbors
        logp = self.batch_dist.log_prob(self.batch_selected.T).T  # [B, K-1]

        sel_embeds = self.selected_embeds  # [B, K-1, embed_dim]

        # X = log_prob * h_j, averaged over sampled neighbors  (paper Theorem 4.1)
        X = logp.unsqueeze(2) * sel_embeds  # [B, K-1, embed_dim]
        X = X.mean(dim=1)                   # [B, embed_dim]

        # Dot product with upstream gradient as reward signal
        batch_loss = torch.bmm(loss_up.unsqueeze(1), X.unsqueeze(2))  # [B, 1, 1]
        return batch_loss.mean()
