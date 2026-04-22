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
        use_reinforce_baseline: bool = False,
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
        self.Ws = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        nn.init.xavier_uniform_(self.Ws, gain=1.414)

        # Paper Eq. 6 — learnable 2-element attention over {importance, uniform}.
        # Init biased toward the uniform head (post-softmax ≈ [0.478, 0.522]),
        # mirroring LinkedIn's sample_a = [10e-3, 10e-3, 10e-1] which gives
        # post-softmax ≈ [0.324, 0.324, 0.353] — a mild random-head preference.
        self.as_ = nn.Parameter(torch.tensor([0.01, 0.1]))

        # REINFORCE EMA baseline for variance reduction.
        self.use_reinforce_baseline = use_reinforce_baseline
        self.register_buffer('baseline_ema', torch.tensor(0.0))
        self.register_buffer('baseline_initialized', torch.tensor(False))
        self.baseline_momentum = 0.99

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

    def forward(self, seed_embeds, candidate_embeds, scope_counts, K,
                scope_hops=None):
        """Sample K-1 neighbors per seed using the PASS policy.

        Args:
            seed_embeds: [B, D]
            candidate_embeds: [B, S, D]
            scope_counts: [B] — number of valid (non-padding) candidates per row
            K: total subgraph size including the seed token; we sample K-1.
            scope_hops: [B, S] optional — hop labels (1/2 = real, 3 = fallback).
                Seeds whose entire scope is hop=3 use uniform sampling.

        Returns:
            selected: [B, K-1] long tensor of indices into candidate_embeds
            dist: the torch.distributions.Categorical used for sampling.
        """
        B, S, D = candidate_embeds.shape
        device = candidate_embeds.device

        # Seed-only mode: K=1 means no neighbors are sampled at all. Short-
        # circuit before any q_imp / Categorical work and return empty
        # selections so callers can still build a [B, 1] subgraph from the
        # seed alone.
        if K - 1 <= 0:
            empty_idx = torch.empty(B, 0, dtype=torch.long, device=device)
            self.batch_selected = empty_idx
            self.batch_dist = None
            self.selected_embeds = torch.empty(B, 0, D, device=device)
            self.fallback_mask = None
            return empty_idx, None

        # Identify fallback seeds (all scope entries are hop=3 random nodes).
        if scope_hops is not None:
            sc = scope_counts.to(device).long()
            arange = torch.arange(S, device=device).unsqueeze(0)
            valid_mask = arange < sc.unsqueeze(1)  # [B, S]
            has_real = ((scope_hops != 3) & valid_mask).any(dim=1)  # [B]
            self.fallback_mask = ~has_real  # True for seeds with only hop=3
        else:
            self.fallback_mask = None

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

        arange_S = torch.arange(S, device=device).unsqueeze(0)
        pad_mask = arange_S >= scope_counts.unsqueeze(1)

        # Uniform baseline over valid (non-pad) positions.
        valid = (~pad_mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        q_rand = valid / denom

        # Mix raw q_imp with q_rand BEFORE non-negativity clamp — matches
        # LinkedIn (PASS-GNN/model.py:108-115): `relu(cat[att1,att2,att3] @
        # softmax(sample_a)) + eps`. Previous code softmaxed q_imp first,
        # which collapsed to a peaky distribution and compressed gradients
        # once Ws learned any structure. Raw dot-product + ReLU + Categorical
        # keeps concentration linear in score magnitude (not exponential).
        as_w = F.softmax(self.as_, dim=0)
        q_tilde = as_w[0] * q_imp + as_w[1] * q_rand
        q_tilde = F.relu(q_tilde)

        # Force pure uniform for fallback seeds (no real neighbors). Use
        # torch.where (not in-place index assignment) so the new F.relu
        # above stays differentiable.
        if self.fallback_mask is not None and self.fallback_mask.any():
            q_tilde = torch.where(
                self.fallback_mask.unsqueeze(1), q_rand, q_tilde
            )

        q_tilde = q_tilde + 1e-9  # floor to keep Categorical well-defined
        q_tilde = q_tilde.masked_fill(pad_mask, 0.0)

        row_sums = q_tilde.sum(dim=1, keepdim=True).clamp(min=1e-9)
        probs = q_tilde / row_sums

        dist = torch.distributions.Categorical(probs=probs)

        num_select = K - 1

        # Sample with replacement to match Categorical.log_prob's i.i.d.
        # assumption — LinkedIn's PASS uses policy.sample_n (with replacement).
        selected = torch.multinomial(probs, num_select, replacement=True).long()

        self.batch_selected = selected
        self.batch_dist = dist
        self.selected_embeds = torch.gather(
            candidate_embeds, 1, selected.unsqueeze(-1).expand(-1, -1, D)
        )

        return selected, dist

    def reinforce_loss(self, loss_up):
        """REINFORCE policy-gradient loss.

        When use_reinforce_baseline is False, matches LinkedIn's reference
        (PASS-GNN/model.py:87-97). When True, uses an EMA baseline for
        variance reduction (standard REINFORCE with baseline).

        Args:
            loss_up: [B, D] gradient dL/d(seed_embed) captured via
                retain_grad() on the seed token's embedding in the RelGT
                forward path.
        """
        # logp: [B, K-1]
        logp = self.batch_dist.log_prob(self.batch_selected.transpose(0, 1)).transpose(0, 1)

        # Exclude fallback seeds — their scope is random noise, not real
        # neighbors, so gradients from them would only add variance.
        real_mask = None  # [B] bool, True for seeds with real neighbors
        if self.fallback_mask is not None and self.fallback_mask.any():
            real_mask = ~self.fallback_mask
            if not real_mask.any():
                return torch.tensor(0.0, device=logp.device, requires_grad=True)

        if self.use_reinforce_baseline:
            sel = self.selected_embeds.detach()  # [B, K-1, D]
            # Per-sample reward: how well the selected neighbors align with
            # the task gradient direction.
            rewards = (loss_up.unsqueeze(1) * sel).sum(dim=-1).mean(dim=1)  # [B]

            if real_mask is not None:
                rewards = rewards[real_mask]
                logp = logp[real_mask]

            # Update EMA baseline
            with torch.no_grad():
                batch_mean = rewards.mean()
                if not self.baseline_initialized:
                    self.baseline_ema.copy_(batch_mean)
                    self.baseline_initialized.fill_(True)
                else:
                    self.baseline_ema.mul_(self.baseline_momentum).add_(
                        batch_mean, alpha=1 - self.baseline_momentum
                    )

            # Advantage = reward - baseline
            advantages = rewards - self.baseline_ema  # [B'] (only real seeds)

            # Policy gradient: advantage * sum of log-probs over K-1 selections
            logp_sum = logp.sum(dim=1)  # [B']
            sample_loss = (advantages.detach() * logp_sum).mean()
            return sample_loss
        else:
            sel = self.selected_embeds.detach()  # [B, K-1, D]

            if real_mask is not None:
                sel = sel[real_mask]
                logp_filtered = logp[real_mask]
                loss_up = loss_up[real_mask]
            else:
                logp_filtered = logp

            X = (logp_filtered.unsqueeze(2) * sel).mean(dim=1)  # [B', D]
            batch_loss = torch.bmm(loss_up.unsqueeze(1), X.unsqueeze(2))
            return batch_loss.mean()
