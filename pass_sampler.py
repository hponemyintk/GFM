"""
PASS-GNN heterogeneous sampler for RelGT.

Implements the Performance-Adaptive Sampling Strategy (PASS) from:
  "Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph
  Neural Networks" (Yoon et al., KDD 2021, LinkedIn).

Paper formulas (Equations 4-7):
    q_imp(j|i)  = (Ws h_i) . (Ws h_j)
    q_rand(j|i) = 1 / N(i)
    q_tilde     = as[0] * q_imp + as[1] * q_rand
    q(j|i)      = q_tilde / sum_k q_tilde

The sampler is adapted for heterogeneous graphs by reusing RelGT's per-type
TorchFrame encoders (shared reference, not a copy) and adding a learnable
per-type embedding to capture cross-type interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PASSHeteroSampler(nn.Module):
    def __init__(
        self,
        tfs_encoder: nn.Module,
        num_types: int,
        embed_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        # Shared reference — not a copy. Caller must NOT register these params
        # twice with the optimizer (use own_parameters()).
        self.tfs_encoder = tfs_encoder

        self.num_types = num_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.type_embeddings = nn.Embedding(num_types, embed_dim)

        # Paper Eq. 4 — single projection matrix Ws.
        self.Ws = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        nn.init.xavier_uniform_(self.Ws, gain=1.414)

        # Paper Eq. 6 — learnable 2-element attention over {importance, uniform}.
        self.as_ = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Populated in forward(), consumed by reinforce_loss().
        self.batch_selected = None
        self.batch_dist = None
        self.selected_embeds = None

    def own_parameters(self):
        """Sampler-exclusive params. Excludes the shared tfs_encoder — callers
        must use this (not .parameters()) to avoid double-registration in the
        optimizer."""
        return [self.Ws, self.as_, *self.type_embeddings.parameters()]

    def _encode_by_type(self, types_tensor, indices_tensor, hetero_data, device):
        """Shared helper for encode_candidates / encode_seeds.

        Args:
            types_tensor: long tensor of node-type indices, any shape.
            indices_tensor: long tensor of local node indices, same shape as
                types_tensor.
            hetero_data: HeteroData object exposing `.tf` per node type.
            device: destination device.

        Returns:
            Tensor of shape (*types_tensor.shape, embed_dim).
        """
        original_shape = types_tensor.shape
        flat_types = types_tensor.reshape(-1)
        flat_indices = indices_tensor.reshape(-1)

        encoded_flat = torch.zeros(
            flat_types.numel(), self.embed_dim, device=device
        )

        inv_map = self.tfs_encoder.inv_node_type_map  # {idx -> type_str}

        for node_type_str, encoder in self.tfs_encoder.encoders.items():
            t_idx = self.tfs_encoder.node_type_map[node_type_str]
            mask = flat_types == t_idx
            if not mask.any():
                continue

            local_idxs = flat_indices[mask]
            tf = hetero_data[node_type_str].tf[local_idxs.cpu()]
            tf = tf.to(device=device)

            # Sanitize NaN/Inf on raw features — same guard as the original
            # NeighborTfsEncoder.forward path.
            for stype_key, tensor in tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tf.feat_dict[stype_key] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            out = encoder(tf)
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)

            positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            encoded_flat[positions] = out

        encoded = encoded_flat.reshape(*original_shape, self.embed_dim)
        encoded = encoded + self.type_embeddings(types_tensor.to(device).long())
        return encoded

    def encode_candidates(self, scope_batch, hetero_data, device):
        """Encode the (B, S) candidate pool through tfs_encoder + type_emb.

        Returns tensor of shape [B, S, embed_dim].
        """
        scope_types = scope_batch["scope_types"].to(device).long()
        scope_indices = scope_batch["scope_indices"].to(device).long()
        return self._encode_by_type(scope_types, scope_indices, hetero_data, device)

    def encode_seeds(self, scope_batch, hetero_data, device):
        """Encode the (B,) seed nodes. Returns [B, embed_dim]."""
        seed_type = scope_batch["seed_type"].to(device).long()
        seed_index = scope_batch["seed_index"].to(device).long()
        return self._encode_by_type(seed_type, seed_index, hetero_data, device)

    def forward(self, seed_embeds, candidate_embeds, scope_counts, K):
        """Sample K-1 neighbors per seed using the PASS policy.

        Args:
            seed_embeds: [B, D]
            candidate_embeds: [B, S, D]
            scope_counts: [B] — number of valid (non-padding) candidates per row
            K: total subgraph size including the seed token; we sample K-1.

        Returns:
            selected: [B, K-1] long tensor of indices into candidate_embeds
            dist: the torch.distributions.Categorical used for sampling.
        """
        B, S, D = candidate_embeds.shape
        device = candidate_embeds.device

        # Detach implements Theorem 4.1 — h_i, h_j are treated as constants by
        # the REINFORCE estimator. Gradients flow only to Ws, as_, and
        # type_embeddings here.
        source = seed_embeds.unsqueeze(1).expand(B, S, D).reshape(B * S, D)
        target = candidate_embeds.reshape(B * S, D)

        ss = torch.mm(source.detach(), self.Ws)
        tt = torch.mm(target.detach(), self.Ws)

        q_imp = torch.bmm(ss.unsqueeze(1), tt.unsqueeze(2)).squeeze(-1).squeeze(-1)
        q_imp = q_imp.reshape(B, S)

        scope_counts = scope_counts.to(device).long()
        q_rand = (1.0 / scope_counts.clamp(min=1).float().unsqueeze(1)).expand(B, S)

        as_w = F.softmax(self.as_, dim=0)
        q_tilde = as_w[0] * q_imp + as_w[1] * q_rand

        arange_S = torch.arange(S, device=device).unsqueeze(0)
        pad_mask = arange_S >= scope_counts.unsqueeze(1)
        q_tilde = q_tilde.masked_fill(pad_mask, 0.0)
        q_tilde = q_tilde.clamp(min=0.0) + 1e-9
        q_tilde = q_tilde.masked_fill(pad_mask, 0.0)

        # Renormalize in case all-zero rows slipped through.
        row_sums = q_tilde.sum(dim=1, keepdim=True).clamp(min=1e-9)
        probs = q_tilde / row_sums

        dist = torch.distributions.Categorical(probs=probs)

        num_select = K - 1
        if num_select <= 0:
            raise ValueError(f"K must be > 1 for PASS sampling (got K={K})")

        # multinomial without replacement where possible; fall back to with-
        # replacement for any row that does not have num_select valid
        # candidates.
        safe_rows = scope_counts >= num_select
        if safe_rows.all():
            selected = torch.multinomial(probs, num_select, replacement=False)
        else:
            selected = torch.multinomial(probs, num_select, replacement=True)
            if safe_rows.any():
                safe_selected = torch.multinomial(
                    probs[safe_rows], num_select, replacement=False
                )
                selected[safe_rows] = safe_selected

        selected = selected.long()

        self.batch_selected = selected
        self.batch_dist = dist
        self.selected_embeds = torch.gather(
            candidate_embeds, 1, selected.unsqueeze(-1).expand(-1, -1, D)
        )

        return selected, dist

    def reinforce_loss(self, loss_up):
        """REINFORCE policy-gradient loss — matches LinkedIn's reference
        (PASS-GNN/model.py:87-97).

        Args:
            loss_up: [B, D] gradient dL/d(seed_embed) captured via
                retain_grad() on the seed token's embedding in the RelGT
                forward path.
        """
        # logp: [B, K-1]
        logp = self.batch_dist.log_prob(self.batch_selected.transpose(0, 1)).transpose(0, 1)

        sel = self.selected_embeds.detach()  # [B, K-1, D]
        X = (logp.unsqueeze(2) * sel).mean(dim=1)  # [B, D]

        # [B, 1, 1]
        batch_loss = torch.bmm(loss_up.unsqueeze(1), X.unsqueeze(2))
        return batch_loss.mean()
