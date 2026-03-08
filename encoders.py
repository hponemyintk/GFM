import re
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GloveTextEmbedding

import torch_frame

import torch_geometric.transforms as T
from torch_geometric.data import Data

# ------------------- Encoder classes -------------------- #

class NeighborNodeTypeEncoder(nn.Module):
    """
    Encoder that maps table name indices to projected GloVe embeddings.

    GloVe embeddings are precomputed at init time for all known table names
    and stored as a buffer. Forward pass is a simple index + linear projection.
    """

    def __init__(self, node_type_map, embedding_dim):
        """
        Args:
            node_type_map (dict): Mapping from table names to integer indices.
            embedding_dim (int): Dimension of the output projected vectors.
        """
        super(NeighborNodeTypeEncoder, self).__init__()

        # Build ordered list of table names: index 0..N-1 from map, index N = mask
        num_types = max(node_type_map.values()) + 1
        if set(node_type_map.values()) != set(range(num_types)):
            raise ValueError(
                f"node_type_map values must be contiguous 0..{num_types - 1}, "
                f"got {sorted(node_type_map.values())}"
            )
        inv_map = {v: k for k, v in node_type_map.items()}
        all_names = [inv_map[i] for i in range(num_types)] + ["mask"]

        # Precompute GloVe embeddings (local variable, not stored as submodule)
        embedder = GloveTextEmbedding(device="cpu")
        with torch.no_grad():
            all_embeddings = embedder(all_names)  # [num_types+1, 300]

        self.register_buffer("glove_embeddings", all_embeddings)

        # Trainable projection
        self.proj = nn.Linear(300, embedding_dim)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, type_indices):
        """
        Args:
            type_indices (Tensor): Integer indices of shape [Batch, K]

        Returns:
            Tensor: Projected GloVe embeddings of shape [Batch, K, embedding_dim]
        """
        x = self.glove_embeddings[type_indices]  # [B, K, 300]
        return self.proj(x)


class NeighborHopEncoder(nn.Module):
    """
    Encoder for hop distances.
    Uses an embedding layer to convert hop counts into dense vectors.
    """
    def __init__(self, max_neighbor_hop, embedding_dim):
        """
        Args:
            max_neighbor_hop (int): The maximum hop distance in your data.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super(NeighborHopEncoder, self).__init__()
        # +1 because we assume hops start from 0 or 1 and go to max_neighbor_hop inclusive
        self.embedding = nn.Embedding(num_embeddings=max_neighbor_hop + 2, embedding_dim=embedding_dim)
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
    
    def forward(self, hop_distances):
        """
        Args:
            hop_distances (Tensor): Tensor of shape (...), containing integer hop distances.
        
        Returns:
            Tensor: Embedded representations of shape (..., embedding_dim).
        """
        shifted = hop_distances + 1
        return self.embedding(shifted)

from torch_geometric.nn import PositionalEncoding

class NeighborTimeEncoder(nn.Module):
    """
    Two-stage time encoder using positional encoding followed by a linear layer.
    """
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): Dimension of the output embedding.
        """
        super(NeighborTimeEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.mask_vector = nn.Parameter(torch.zeros(embedding_dim))
        
    def reset_parameters(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.mask_vector, mean=0.0, std=0.02)

    def forward(self, rel_time):
        """
        Args:
            rel_time (Tensor): Tensor of shape [B, K] containing time values in seconds.
        Returns:
            Tensor: Encoded time features with shape [B, K, embedding_dim].
        """
        # Get the original batch dimensions
        B, K = rel_time.shape

        # Flatten the input from [B, K] to [B*K]
        flattened_time = rel_time.view(-1)

        # Apply positional encoding to the flattened input
        pos_encoded = self.pos_encoder(flattened_time)  # shape: [B*K, embedding_dim]

        # Apply a linear transformation
        linear_out = self.linear(pos_encoded)  # shape: [B*K, embedding_dim]
        linear_out = linear_out.view(B, K, -1)
        
        # create a mask: 1 where time is masked (i.e. < 0), else 0.
        mask = (rel_time < 0).unsqueeze(-1).float()
        mask_vector = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        # where mask==1, use mask_vector; else use linear_out.
        out = (1 - mask) * linear_out + mask * mask_vector
        return out
    
    

from typing import Dict, Any, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_frame

from torch import Tensor
from torch_frame.data import TensorFrame, MultiNestedTensor
from torch_frame.data.stats import StatType


# ============================================================
# UNIVERSAL ENCODERS (Table Agnostic)
# ============================================================

class SharedNumericalEncoder(nn.Module):
    """
    Projects continuous values using a shared MLP.
    Each value is encoded as [value, is_missing] so the model can
    distinguish true zeros from missing data.
    """

    def __init__(self, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: Any) -> Tensor:
        if hasattr(x, "values") and not callable(x.values):
            x = x.values

        if x.dim() > 2:
            x = x.squeeze(-1)

        B, C = x.shape

        # Clamp infinities while preserving NaN for missingness detection
        x = x.clamp(min=-1e6, max=1e6)
        nan_mask = torch.isnan(x).float()  # [B, C]
        x = torch.nan_to_num(x, nan=0.0)

        # [B, C, 2]: value + missingness indicator
        x_aug = torch.stack([x, nan_mask], dim=-1)
        x_flat = x_aug.view(B * C, 2)

        out = self.mlp(x_flat)

        return out.view(B, C, -1)


class SharedCategoricalEncoder(nn.Module):
    """
    Uses hashing to map any category from any table to a fixed embedding space.
    """

    def __init__(self, out_channels: int, num_hash_buckets: int = 9311):
        super().__init__()

        self.num_hash_buckets = num_hash_buckets
        self.embedding = nn.Embedding(num_hash_buckets, out_channels)

    def forward(self, x: Any) -> Tensor:
        if hasattr(x, "values") and not callable(x.values):
            x = x.values

        if x.dim() > 2:
            x = x.squeeze(-1)

        hashed_x = x.long() % self.num_hash_buckets

        return self.embedding(hashed_x)


class SharedMultiCategoricalEncoder(nn.Module):
    """
    Handles cells containing lists of categories (e.g. Movie Genres).
    Hash each element → embed → mean pool.
    """

    def __init__(self, out_channels: int, num_hash_buckets: int = 9311):
        super().__init__()

        self.num_hash_buckets = num_hash_buckets
        self.embedding = nn.Embedding(
            num_hash_buckets,
            out_channels,
            padding_idx=0,
        )

    def forward(self, x: Any) -> Tensor:

        is_nested = (
            hasattr(x, "values")
            and not callable(x.values)
            and hasattr(x, "offset")
        )

        if not is_nested:
            # ------------------------------------
            # PATH A: Dense tensors
            # ------------------------------------
            if x.dim() > 2 and x.size(-1) == 1:
                x = x.squeeze(-1)
            elif x.dim() == 2:
                x = x.unsqueeze(-1)

            B = x.size(0)
            C = x.size(1) if x.dim() > 1 else 1

            x = x.view(B, C, -1)
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.relu(x)

            hashed_x = x.long() % self.num_hash_buckets

            emb = self.embedding(hashed_x)

            mask = (x > 0).float().unsqueeze(-1)

            sum_emb = (emb * mask).sum(dim=2)
            counts = mask.sum(dim=2).clamp(min=1)

            return sum_emb / counts

        else:
            # ------------------------------------
            # PATH B: MultiNestedTensor
            # ------------------------------------

            hashed_values = x.values.long() % self.num_hash_buckets
            flat_embeddings = self.embedding(hashed_values)

            counts = x.offset[1:] - x.offset[:-1]
            counts = counts.clamp(min=1).unsqueeze(-1)

            B, Num_Cols = x.size(0), x.size(1)

            out = torch.zeros(
                B * Num_Cols,
                self.embedding.embedding_dim,
                device=x.device,
            )

            cell_indices = torch.arange(B * Num_Cols, device=x.device)
            repeats = x.offset[1:] - x.offset[:-1]

            index = torch.repeat_interleave(cell_indices, repeats)

            out.index_add_(0, index, flat_embeddings)

            out = out / counts

            return out.view(B, Num_Cols, -1)


class SharedTimestampEncoder(nn.Module):
    """
    Encodes time using periodic features (sin/cos).
    Supports multi-component timestamps by pooling across components.
    """

    def __init__(self, out_channels: int):
        super().__init__()

        self.out_channels = out_channels
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x: Any) -> Tensor:

        if hasattr(x, "values") and not callable(x.values):
            x = x.values

        B = x.size(0)
        C = x.size(1) if x.dim() > 1 else 1

        x = x.view(B, C, -1)

        half_dim = self.out_channels // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float, device=x.device) * -emb
        )

        x_expanded = x.unsqueeze(-1) * emb.view(1, 1, 1, -1)

        emb_cat = torch.cat(
            [x_expanded.sin(), x_expanded.cos()],
            dim=-1,
        )

        if self.out_channels % 2 == 1:
            emb_cat = F.pad(emb_cat, (0, 1, 0, 0))

        emb_pooled = emb_cat.mean(dim=2)

        return self.linear(emb_pooled)


class SharedEmbeddingEncoder(nn.Module):
    """
    Projects upstream embeddings to model dimension.
    Creates one learned Linear projector per unique embedding dimension
    discovered from col_stats_dict (via StatType.EMB_DIM).
    """

    def __init__(self, out_channels: int, emb_dims: Optional[set] = None):
        super().__init__()

        self.out_channels = out_channels

        if emb_dims is None:
            emb_dims = set()

        self.projectors = nn.ModuleDict({
            str(d): nn.Linear(d, out_channels) for d in emb_dims
        })

    def forward(self, x: Any) -> Tensor:

        if hasattr(x, "values") and not callable(x.values):
            x = x.values

        if x.dim() == 2:
            # x is [B, num_cols * emb_dim] — reshape to [B, num_cols, emb_dim]
            total_dim = x.size(-1)
            matches = [d_str for d_str in self.projectors if total_dim % int(d_str) == 0]
            if len(matches) == 0:
                raise KeyError(
                    f"Embedding total dim {total_dim} not divisible by any known "
                    f"per-column dim: {set(self.projectors.keys())}"
                )
            if len(matches) > 1:
                raise KeyError(
                    f"Ambiguous embedding reshape: total dim {total_dim} is divisible "
                    f"by multiple known dims: {matches}. Ensure embedding columns "
                    f"have distinct dimensions or provide 3D input."
                )
            d = int(matches[0])
            x = x.view(x.size(0), -1, d)
            return self.projectors[matches[0]](x)

        D = x.size(-1)
        if str(D) not in self.projectors:
            raise KeyError(
                f"No projector for embedding dim {D}. "
                f"Known dims: {set(self.projectors.keys())}"
            )
        return self.projectors[str(D)](x)


# ============================================================
# TABLE AGNOSTIC MASTER ENCODER
# ============================================================

class TableAgnosticStypeEncoder(nn.Module):

    def __init__(self, channels: int, emb_dims: Optional[set] = None):
        super().__init__()

        self.channels = channels

        self.encoders = nn.ModuleDict({
            str(torch_frame.numerical): SharedNumericalEncoder(channels),
            str(torch_frame.categorical): SharedCategoricalEncoder(channels),
            str(torch_frame.multicategorical): SharedMultiCategoricalEncoder(channels),
            str(torch_frame.timestamp): SharedTimestampEncoder(channels),
            str(torch_frame.embedding): SharedEmbeddingEncoder(channels, emb_dims=emb_dims),
        })

    def forward(self, tf: TensorFrame) -> Tensor:

        atom_embeddings: List[Tensor] = []

        for stype_name in tf.feat_dict.keys():

            stype_str = str(stype_name)

            if stype_str not in self.encoders:
                continue

            feat = tf.feat_dict[stype_name]

            x_stype = self.encoders[stype_str](feat)

            atom_embeddings.append(x_stype)

        if len(atom_embeddings) == 0:
            return torch.zeros(
                (tf.num_rows, 0, self.channels),
                device=tf.device,
            )

        x = torch.cat(atom_embeddings, dim=1)

        return x


# ============================================================
# NEIGHBOR TFS ENCODER (Table Agnostic)
# ============================================================

class NeighborTfsEncoder(nn.Module):
    """
    Table agnostic transformer encoder for neighbor TensorFrames.
    """

    def __init__(
        self,
        channels: int,
        node_type_map: Dict[str, int],
        col_names_dict: Optional[Dict] = None,
        col_stats_dict: Optional[Dict] = None,
        default_stype_encoder_cls_kwargs: Optional[Dict] = None,
        torch_frame_model_cls=None,
        torch_frame_model_kwargs=None,
        num_layers: int = 4,
        nhead: int = 4,
    ):

        super().__init__()

        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for nt, idx in node_type_map.items()}
        self.channels = channels

        # Discover all unique embedding dimensions from col_stats_dict
        emb_dims = set()
        if col_names_dict and col_stats_dict:
            for node_type, stype_dict in col_names_dict.items():
                for col in stype_dict.get(torch_frame.embedding, []):
                    table_stats = col_stats_dict.get(node_type, {})
                    cs = table_stats.get(col, {})
                    dim = cs.get(StatType.EMB_DIM)
                    if dim is not None and dim > 0:
                        emb_dims.add(dim)

        self.table_agnostic_encoder = TableAgnosticStypeEncoder(channels, emb_dims=emb_dims)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=channels * 2,
            batch_first=True,
            norm_first=True,
        )

        self.shared_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls_embedding = nn.Parameter(
            torch.randn(1, 1, channels)
        )

        self.reset_parameters()

        # Register per-table numerical mean/std as buffers for Z-score normalization
        self._node_type_to_safe = {}
        _safe_to_node_type = {}  # reverse map for collision detection
        zscore_table_count = 0
        if col_names_dict and col_stats_dict:
            for node_type, stype_dict in col_names_dict.items():
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', node_type)
                if safe_name in _safe_to_node_type:
                    raise ValueError(
                        f"Z-score buffer name collision: '{node_type}' and "
                        f"'{_safe_to_node_type[safe_name]}' both sanitize to "
                        f"'{safe_name}'. Rename one of the tables to avoid ambiguity."
                    )
                _safe_to_node_type[safe_name] = node_type
                self._node_type_to_safe[node_type] = safe_name
                num_cols = stype_dict.get(torch_frame.numerical, [])
                if not num_cols:
                    continue
                table_stats = col_stats_dict.get(node_type, {})
                means = []
                stds = []
                for col in num_cols:
                    cs = table_stats.get(col, {})
                    means.append(float(cs.get(StatType.MEAN, 0.0) or 0.0))
                    stds.append(float(cs.get(StatType.STD, 1.0) or 1.0))
                self.register_buffer(
                    f'_num_mean_{safe_name}',
                    torch.tensor(means, dtype=torch.float32),
                )
                self.register_buffer(
                    f'_num_std_{safe_name}',
                    torch.tensor(stds, dtype=torch.float32),
                )
                zscore_table_count += 1

        # Persists through save/load — used to detect missing col_names_dict at inference
        self.register_buffer(
            '_num_zscore_tables',
            torch.tensor(zscore_table_count, dtype=torch.long),
        )

        # Precompute GloVe embeddings for column names (semantic column identity)
        all_col_names: List[str] = []
        col_name_set: set = set()
        if col_names_dict:
            for node_type, stype_dict in col_names_dict.items():
                for stype, col_list in stype_dict.items():
                    for col_name in col_list:
                        if col_name not in col_name_set:
                            all_col_names.append(col_name)
                            col_name_set.add(col_name)

        self._col_name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(all_col_names)
        }

        if all_col_names:
            col_embedder = GloveTextEmbedding(device="cpu")
            with torch.no_grad():
                col_embeddings = col_embedder(all_col_names)  # [N_unique, 300]
            self.register_buffer("_col_glove_embeddings", col_embeddings)
            self._glove_embedder = col_embedder  # kept for unseen columns
        else:
            self.register_buffer("_col_glove_embeddings", torch.zeros(0, 300))
            self._glove_embedder = None

        self.col_name_proj = nn.Linear(300, channels)

    def reset_parameters(self):

        nn.init.normal_(self.cls_embedding, std=0.01)

        for p in self.shared_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.table_agnostic_encoder.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        if hasattr(self, "col_name_proj"):
            self.col_name_proj.reset_parameters()

    def _normalize_numerical(self, big_tf, node_type: str):
        """Z-score normalize numerical columns using precomputed per-table stats.
        NaN values propagate through arithmetic so SharedNumericalEncoder
        can detect and handle them with a learned missingness indicator."""
        if torch_frame.numerical not in big_tf.feat_dict:
            return
        safe_name = self._node_type_to_safe.get(node_type)
        if safe_name is None or not hasattr(self, f'_num_mean_{safe_name}'):
            return
        mean = getattr(self, f'_num_mean_{safe_name}')
        std = getattr(self, f'_num_std_{safe_name}')
        feat = big_tf.feat_dict[torch_frame.numerical]
        if hasattr(feat, "values") and not callable(feat.values):
            feat = feat.values
        # NaN propagates naturally: (NaN - mean) / std = NaN
        # Clamp to ±10 std devs to control outliers (clamp preserves NaN)
        big_tf.feat_dict[torch_frame.numerical] = ((feat - mean) / (std + 1e-8)).clamp(-10, 10)

    def _get_col_semantic_embeddings(self, big_tf, device):
        """Build [num_cols, channels] semantic embedding for the columns in big_tf.

        Columns are ordered to match TableAgnosticStypeEncoder's concatenation:
        for each stype in feat_dict iteration order, extend with that stype's
        column names from big_tf.col_names_dict.

        For column names not seen at init, computes GloVe on-the-fly via the
        stored embedder, or falls back to a zero vector.
        """
        ordered_col_names: List[str] = []
        for stype in big_tf.feat_dict.keys():
            stype_str = str(stype)
            if stype_str not in self.table_agnostic_encoder.encoders:
                continue
            if hasattr(big_tf, 'col_names_dict') and stype in big_tf.col_names_dict:
                ordered_col_names.extend(big_tf.col_names_dict[stype])

        if not ordered_col_names:
            return None

        glove_vecs: List[Tensor] = []
        for name in ordered_col_names:
            if name in self._col_name_to_idx:
                idx = self._col_name_to_idx[name]
                glove_vecs.append(self._col_glove_embeddings[idx])
            else:
                # Unseen column: compute on-the-fly or fall back to zeros
                if self._glove_embedder is not None:
                    with torch.no_grad():
                        vec = self._glove_embedder([name])[0].to(device)
                else:
                    vec = torch.zeros(300, device=device)
                glove_vecs.append(vec)

        glove_stack = torch.stack(glove_vecs, dim=0)  # [num_cols, 300]
        return self.col_name_proj(glove_stack)  # [num_cols, channels]

    def forward(
        self,
        batch_dict: Dict[str, Any],
        neighbor_types: Tensor,
    ) -> Tensor:

        if self._num_zscore_tables > 0 and len(self._node_type_to_safe) == 0:
            raise RuntimeError(
                "Model was trained with Z-score normalization for "
                f"{self._num_zscore_tables.item()} table(s), but "
                "col_names_dict/col_stats_dict were not provided at "
                "construction time. Pass the same col_names_dict and "
                "col_stats_dict used during training to NeighborTfsEncoder."
            )

        grouped_tfs = batch_dict["grouped_tfs"]
        grouped_indices = batch_dict["grouped_indices"]

        flat_batch_idx = batch_dict["flat_batch_idx"]
        flat_nbr_idx = batch_dict["flat_nbr_idx"]

        B, K = neighbor_types.shape
        N = len(flat_batch_idx)

        device = neighbor_types.device

        encoded_flat_tensor = torch.zeros(
            (N, self.channels),
            device=device,
        )

        for t_int, big_tf in grouped_tfs.items():

            big_tf = big_tf.to(device)

            for stype, tensor in big_tf.feat_dict.items():
                if stype == torch_frame.numerical:
                    continue  # NaN preserved for missingness detection
                if isinstance(tensor, torch.Tensor):
                    big_tf.feat_dict[stype] = torch.nan_to_num(
                        tensor,
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    )

            # Z-score normalize numerical features
            node_type = self.inv_node_type_map[t_int]
            self._normalize_numerical(big_tf, node_type)

            x_cols = self.table_agnostic_encoder(big_tf)

            # Add column-name semantic embeddings
            col_sem = self._get_col_semantic_embeddings(big_tf, device)
            if col_sem is not None and col_sem.shape[0] == x_cols.shape[1]:
                x_cols = x_cols + col_sem.unsqueeze(0)  # broadcast over batch

            batch_size = x_cols.size(0)

            cls_tokens = self.cls_embedding.expand(batch_size, -1, -1)

            x_seq = torch.cat([cls_tokens, x_cols], dim=1)

            x_out = self.shared_transformer(x_seq)

            x_final = x_out[:, 0, :]

            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)

            encoded_flat_tensor[idx_tensor] = x_final

        output = torch.zeros(
            (B, K, self.channels),
            device=device,
        )

        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)

        output[indices_i, indices_j] = encoded_flat_tensor

        return output







    
    
from torch_geometric.nn import GINConv

class GNNPEEncoder(nn.Module):
    """
    A GNN-based positional encoder that:
      1) Assigns each node a random scalar feature from a Normal(0,1).
      2) Linearly projects it to embedding_dim.
      3) Runs a small GIN GNN on (x, edge_index, batch).
      4) Aggregates the intermediate outputs of the GNN using one of:
        - "none": use only the final layer's output,
        - "cat": concatenate all layer outputs,
        - "mean": average all layer outputs,
        - "max": max pool across all layer outputs.
      5) Returns a [B, K, embedding_dim] shaped embedding to match the rest of the pipeline.
    """
    def __init__(self, embedding_dim: int, num_layers: int = 4, pooling: str = 'none', pe_dim: int = 0):
        super().__init__()
        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.layer_embedding_dim = embedding_dim // 4
        self.pe_dim = pe_dim
        
        if self.pe_dim > 0:
            self.input_proj = nn.Linear(self.pe_dim, self.layer_embedding_dim)
        else:
           self.input_proj = nn.Linear(1, self.layer_embedding_dim)

        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.layer_embedding_dim, self.layer_embedding_dim*2),
                nn.BatchNorm1d(self.layer_embedding_dim*2),
                nn.ReLU(),
                nn.Linear(self.layer_embedding_dim*2, self.layer_embedding_dim)
            )
            self.conv.append(GINConv(mlp, train_eps=True))
        
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(self.layer_embedding_dim))
        
        if self.pooling == 'cat':
            final_input_dim = self.layer_embedding_dim * num_layers
        elif self.pooling in ['none', 'mean', 'max']:
            final_input_dim = self.layer_embedding_dim
        else:
            raise ValueError("Invalid pooling method. Choose from 'none', 'cat', 'mean', 'max'.")
        
        self.final_transform = nn.Linear(final_input_dim, embedding_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        for conv in self.conv:
            for layer in conv.nn:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        nn.init.xavier_uniform_(self.final_transform.weight)
        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index, batch):
        """
        Args:
            edge_index (torch.Tensor): shape [2, E], the adjacency for the subgraph(s).
            batch (torch.Tensor): shape [total_nodes], specifying subgraph membership for each node.

        Returns:
            (torch.Tensor): shape [B, K, embedding_dim], a node-level embedding for each node
                            in the subgraph, where B is the batch size, K is the # of nodes in
                            each subgraph if each subgraph is the same size, or sum(K_i) if variable.
        """
        device = edge_index.device
        total_nodes = batch.size(0) 

        if self.pe_dim > 0:
            data = Data(edge_index=edge_index, num_nodes=total_nodes)
            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)
            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes, 1, device=device)
            
        x = self.input_proj(x_input)
        
        outputs = []
        for i, conv in enumerate(self.conv):
            x_res = x  
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            outputs.append(x)
        
        if self.pooling == 'none':
            x_final = outputs[-1]
        elif self.pooling == 'cat':
            x_final = torch.cat(outputs, dim=-1)
        elif self.pooling == 'mean':
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.mean(outputs_tensor, dim=-1)
        elif self.pooling == 'max':
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.max(outputs_tensor, dim=-1)[0]

        x = self.final_transform(x_final)
        
        B = batch.max().item() + 1 
        K = total_nodes // B
        out = x.view(B, K, -1)

        return out