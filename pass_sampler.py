"""
PASS-GNN heterogeneous sampler for RelGT.

Implements the Performance-Adaptive Sampling Strategy (PASS) from:
  "Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks"
  (Yoon et al., KDD 2021, LinkedIn)

Adapted for heterogeneous heterophilic graphs by:
  - Reusing RelGT's per-type TorchFrame encoders as projection matrices
  - Adding learnable type embeddings for cross-type attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any


class PASSHeteroSampler(nn.Module):
    """
    PASS-style learned neighbor sampler for heterogeneous graphs.

    Uses the exact PASS 3-head attention formula:
        att1 = dot(source @ W1, target @ W1)
        att2 = dot(source @ W2, target @ W2)
        att3 = uniform
        att  = relu( [att1, att2, att3] @ softmax(sample_a) ) + eps
        dist = Categorical(probs=att)

    Heterogeneous adaptation:
        - Per-type feature projection via shared RelGT tfs_encoder
        - Learnable type embeddings added to projected features
    """

    def __init__(
        self,
        tfs_encoder: nn.Module,
        num_types: int,
        embed_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.tfs_encoder = tfs_encoder  # shared reference, not a copy
        self.num_types = num_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Type embeddings for cross-type interaction modeling
        self.type_embeddings = nn.Embedding(num_types, embed_dim)

        # PASS attention parameters — exact same init as PASS-GNN
        self.sample_W = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        nn.init.xavier_uniform_(self.sample_W.data, gain=1.414)

        self.sample_W2 = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        nn.init.xavier_uniform_(self.sample_W2.data, gain=1.414)

        # Attention-of-attentions: shape [3, 1], same init as PASS
        self.sample_a = nn.Parameter(
            torch.FloatTensor([[1e-2], [1e-2], [1e-1]])
        )
        self.softmax_a = nn.Softmax(dim=0)

        # State saved for REINFORCE
        self.batch_selected = None
        self.batch_dist = None
        self.selected_embeds = None

    def own_parameters(self):
        """Return only the sampler's own parameters (not the shared tfs_encoder)."""
        return [
            self.sample_W, self.sample_W2, self.sample_a,
            *self.type_embeddings.parameters(),
        ]

    def _node_attention(self, source, target, weight):
        """Exact PASS _node_attention: bilinear dot-product attention."""
        # source: [N, embed_dim], target: [N, embed_dim], weight: [embed_dim, hidden_dim]
        ss = torch.mm(source, weight)   # [N, hidden_dim]
        tt = torch.mm(target, weight)   # [N, hidden_dim]
        att = torch.bmm(ss.unsqueeze(1), tt.unsqueeze(2)).squeeze(2)  # [N, 1]
        return att

    def encode_candidates(self, scope_batch, data, device):
        """
        Encode scope candidates through the shared tfs_encoder + type_embeddings.

        Args:
            scope_batch: dict with scope_types [B, S], scope_indices [B, S], scope_count [B]
            data: HeteroData with .tf per node type
            device: target device

        Returns:
            candidate_embeds: [B, S, embed_dim]
        """
        scope_types = scope_batch["scope_types"]    # [B, S] int
        scope_indices = scope_batch["scope_indices"]  # [B, S] int
        B, S = scope_types.shape

        inv_node_type_map = self.tfs_encoder.inv_node_type_map

        # Group candidates by type (same pattern as RelGTTokens.collate)
        encoded_flat = torch.zeros(B * S, self.embed_dim, device=device)

        for t_int, encoder in self.tfs_encoder.encoders.items():
            # t_int is a string key in ModuleDict
            t_idx = self.tfs_encoder.node_type_map[t_int]
            mask = (scope_types == t_idx)  # [B, S] bool
            if not mask.any():
                continue

            local_idxs = scope_indices[mask]  # 1D
            tf = data[t_int].tf[local_idxs].to(device=device)

            # Clean NaN/Inf
            for st, tensor in tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tf.feat_dict[st] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            out = encoder(tf)  # [N_group, channels] or [N_group, 1, channels]
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)

            # Scatter back to flat positions
            flat_positions = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
            encoded_flat[flat_positions] = out

        # Reshape to [B, S, embed_dim]
        candidate_embeds = encoded_flat.reshape(B, S, self.embed_dim)

        # Add type embeddings
        type_emb = self.type_embeddings(scope_types.to(device))  # [B, S, embed_dim]
        candidate_embeds = candidate_embeds + type_emb

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
        seed_types = scope_batch["seed_type"]    # [B] int
        seed_indices = scope_batch["seed_index"]  # [B] int
        B = seed_types.shape[0]

        inv_node_type_map = self.tfs_encoder.inv_node_type_map
        seed_embeds = torch.zeros(B, self.embed_dim, device=device)

        for t_int, encoder in self.tfs_encoder.encoders.items():
            t_idx = self.tfs_encoder.node_type_map[t_int]
            mask = (seed_types == t_idx)  # [B] bool
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

        # Add type embeddings
        type_emb = self.type_embeddings(seed_types.to(device))  # [B, embed_dim]
        seed_embeds = seed_embeds + type_emb

        return seed_embeds

    def forward(self, seed_embeds, candidate_embeds, scope_counts, K):
        """
        PASS attention-based sampling. Exact PASS formula.

        Args:
            seed_embeds:      [B, embed_dim]
            candidate_embeds: [B, S, embed_dim]
            scope_counts:     [B] — number of valid candidates per seed
            K:                int — total subgraph size (will sample K-1 neighbors)

        Returns:
            selected: [B, K-1] — indices into scope dimension
            dist:     Categorical distribution (for REINFORCE)
        """
        B, S, D = candidate_embeds.shape

        # Expand seed to match candidates
        source = seed_embeds.unsqueeze(1).expand(B, S, D).reshape(B * S, D)
        target = candidate_embeds.reshape(B * S, D)

        # Head 1: dot-product attention after projection
        att1 = self._node_attention(source, target, self.sample_W)   # [B*S, 1]

        # Head 2: second independent head
        att2 = self._node_attention(source, target, self.sample_W2)  # [B*S, 1]

        # Head 3: uniform weights (exploration regularizer)
        counts_expanded = scope_counts.unsqueeze(1).expand(B, S).reshape(B * S, 1).clamp(min=1).float()
        att3 = torch.ones(B * S, 1, device=source.device) / counts_expanded

        # Attention-of-attentions: cat → matmul with softmax(sample_a) → relu
        att = torch.cat([att1, att2, att3], dim=1)                        # [B*S, 3]
        att = F.relu(torch.mm(att, self.softmax_a(self.sample_a)))        # [B*S, 1]
        att = att + 1e-9 * torch.ones_like(att)                           # epsilon
        att = att.reshape(B, S)                                           # [B, S]

        # Mask padding positions
        mask = torch.arange(S, device=att.device).unsqueeze(0).expand(B, S) >= scope_counts.unsqueeze(1)
        att.masked_fill_(mask, 0.0)

        # PASS uses Categorical(probs=), which normalizes internally
        dist = torch.distributions.Categorical(probs=att)
        selected = dist.sample((K - 1,)).T  # [B, K-1]

        # Save for REINFORCE
        self.batch_selected = selected
        self.batch_dist = dist

        # Save selected embeddings for reuse in RelGT forward
        self.selected_embeds = torch.gather(
            candidate_embeds, 1,
            selected.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K-1, embed_dim]

        return selected, dist

    def reinforce_loss(self, loss_up):
        """
        Exact PASS REINFORCE formula.

        Args:
            loss_up: [B, embed_dim] — gradient of task loss w.r.t. intermediate
                     representation (model.X1.grad)

        Returns:
            scalar loss for the sampling policy
        """
        # Recompute log probs for the actions taken
        logp = self.batch_dist.log_prob(
            self.batch_selected.T
        ).T  # [B, K-1]

        sel_embeds = self.selected_embeds  # [B, K-1, embed_dim]

        # PASS formula: X = log_prob * features, averaged over selected
        X = logp.unsqueeze(2) * sel_embeds    # [B, K-1, embed_dim]
        X = X.mean(dim=1)                      # [B, embed_dim]

        # Dot product with upstream gradient as reward signal
        batch_loss = torch.bmm(
            loss_up.unsqueeze(1), X.unsqueeze(2)
        )  # [B, 1, 1]

        return batch_loss.mean()
