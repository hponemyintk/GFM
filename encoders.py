import re
import warnings
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

    def iter_active_stypes(self, feat_dict):
        """Yield stypes from feat_dict that have a registered encoder.

        This defines the canonical column ordering for concatenation.
        Both forward() and NeighborTfsEncoder._get_col_semantic_embeddings()
        must use this to stay aligned.
        """
        for stype in feat_dict.keys():
            if str(stype) in self.encoders:
                yield stype

    def forward(self, tf: TensorFrame) -> Tensor:

        atom_embeddings: List[Tensor] = []

        for stype_name in self.iter_active_stypes(tf.feat_dict):

            feat = tf.feat_dict[stype_name]

            x_stype = self.encoders[str(stype_name)](feat)

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
    """Table-agnostic transformer encoder for neighbor TensorFrames.

    Schema state (per-prefixed-type Z-score buffers, column-name GloVe
    embeddings, the inverse type-id map) is **not** baked into
    ``__init__`` -- it's installed via :meth:`register_dataset`, which
    can be called multiple times to extend the encoder's known schema
    (e.g. at adoption time on a new dataset).

    For backward compatibility, ``__init__`` still accepts
    ``node_type_map`` + ``col_names_dict`` + ``col_stats_dict``: when
    all three are provided it auto-calls :meth:`register_dataset` once.
    Phase-4 cross-dataset adoption will instead construct with
    architectural args only and call :meth:`register_dataset` per
    incoming dataset.
    """

    def __init__(
        self,
        channels: int,
        node_type_map: Optional[Dict[str, int]] = None,
        col_names_dict: Optional[Dict] = None,
        col_stats_dict: Optional[Dict] = None,
        default_stype_encoder_cls_kwargs: Optional[Dict] = None,
        torch_frame_model_cls=None,
        torch_frame_model_kwargs=None,
        num_layers: int = 4,
        nhead: int = 4,
    ):

        super().__init__()

        self.channels = channels

        # ------------------------------------------------- schema state
        # Empty until register_dataset() is called. Populated incrementally
        # so the encoder can accept additional datasets at adoption time.
        self.node_type_map: Dict[str, int] = {}
        self.inv_node_type_map: Dict[int, str] = {}
        self._node_type_to_safe: Dict[str, str] = {}
        self._col_name_to_idx: Dict[str, int] = {}
        # Runtime cache for GloVe vectors of column names that arrive at
        # forward time but weren't in any register_dataset call. Populated
        # lazily; per-rank (GloVe is deterministic so DDP ranks agree).
        self._col_unseen_cache: Dict[str, Tensor] = {}
        self._glove_embedder = None  # constructed on first register_dataset

        # ----------------------------------------- architectural state
        # Shared across registered datasets. Built with empty emb_dims;
        # _extend_emb_dims() adds Linear projectors for new dims as
        # register_dataset() encounters them.
        self.table_agnostic_encoder = TableAgnosticStypeEncoder(
            channels, emb_dims=set(),
        )

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

        self.col_name_proj = nn.Linear(300, channels)

        self.reset_parameters()

        # Empty schema-buffers. register_dataset() either re-registers
        # them at full size or appends + re-registers.
        self.register_buffer(
            "_col_glove_embeddings", torch.zeros(0, 300),
        )
        self.register_buffer(
            "_num_zscore_tables", torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "_num_col_semantic_cols", torch.tensor(0, dtype=torch.long),
        )

        # Backward-compat: if a full schema was passed at construction,
        # register it now. This preserves the dev-kyaw / pre-PR-1.2
        # call signature used by main_node_ddp.py + train_multi_task.py
        # + the existing test files.
        if col_names_dict is not None and col_stats_dict is not None:
            if node_type_map is None:
                raise ValueError(
                    "When col_names_dict and col_stats_dict are provided "
                    "to NeighborTfsEncoder.__init__, node_type_map must "
                    "also be provided. Either pass all three or call "
                    "register_dataset() explicitly after construction."
                )
            self.register_dataset(
                node_type_map, col_names_dict, col_stats_dict,
            )
        elif node_type_map is not None:
            # Type-map-only registration. Mirrors the pre-refactor
            # __init__ behavior where ``self.node_type_map`` /
            # ``inv_node_type_map`` were always set from the
            # constructor arg even when col_names_dict / col_stats_dict
            # were None. forward()'s ``inv_node_type_map[t_int]`` lookup
            # requires this to be populated.
            for nt, idx in node_type_map.items():
                self.node_type_map[nt] = idx
                self.inv_node_type_map[idx] = nt

    def register_dataset(
        self,
        node_type_map: Dict[str, int],
        col_names_dict: Dict[str, Dict[Any, List[str]]],
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
    ) -> None:
        """Register a dataset's schema. Idempotent on the same prefixed
        types; raises ``ValueError`` on a safe-name collision.

        Extends per-table Z-score buffers, the column-name GloVe table,
        the SharedEmbeddingEncoder's per-emb-dim projectors, and the
        node-type maps. Callable repeatedly with disjoint datasets so
        adoption-time code can layer on new schemas without
        reconstructing the backbone.
        """
        # 1. Extend node_type_map / inv_node_type_map.
        for nt, idx in node_type_map.items():
            existing = self.node_type_map.get(nt)
            if existing is not None and existing != idx:
                raise ValueError(
                    f"node_type_map collision for '{nt}': existing "
                    f"idx {existing}, new idx {idx}"
                )
            self.node_type_map[nt] = idx
            self.inv_node_type_map[idx] = nt

        # 2. Extend SharedEmbeddingEncoder projectors for any new
        #    embedding dims discovered from col_stats_dict.
        new_emb_dims: set = set()
        for nt, stype_dict in col_names_dict.items():
            for col in stype_dict.get(torch_frame.embedding, []):
                cs = col_stats_dict.get(nt, {}).get(col, {})
                dim = cs.get(StatType.EMB_DIM)
                if dim is not None and dim > 0:
                    new_emb_dims.add(dim)
        self._extend_emb_dims(new_emb_dims)

        # 3. Register per-table Z-score buffers; raise on safe-name
        #    collision against any prefix we've already registered.
        _safe_to_nt_seen = {
            v: k for k, v in self._node_type_to_safe.items()
        }
        new_zscore_tables = 0
        for nt, stype_dict in col_names_dict.items():
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', nt)
            if (
                safe_name in _safe_to_nt_seen
                and _safe_to_nt_seen[safe_name] != nt
            ):
                raise ValueError(
                    f"Z-score buffer name collision: '{nt}' and "
                    f"'{_safe_to_nt_seen[safe_name]}' both sanitize to "
                    f"'{safe_name}'. Rename one of the tables."
                )
            if nt in self._node_type_to_safe:
                continue  # already registered, idempotent
            _safe_to_nt_seen[safe_name] = nt
            self._node_type_to_safe[nt] = safe_name
            num_cols = stype_dict.get(torch_frame.numerical, [])
            if not num_cols:
                continue
            means = []
            stds = []
            for col in num_cols:
                cs = col_stats_dict.get(nt, {}).get(col, {})
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
            new_zscore_tables += 1
        self._num_zscore_tables.data = (
            self._num_zscore_tables + new_zscore_tables
        )

        # 4. Extend column-name GloVe buffer with any new column names.
        # Dedupe both against already-registered names AND within this
        # call's iteration (otherwise "price" appearing under two node
        # types in a single register_dataset() call would be embedded
        # twice).
        new_col_names: List[str] = []
        new_col_name_set: set = set()
        for nt, stype_dict in col_names_dict.items():
            for stype, col_list in stype_dict.items():
                for col_name in col_list:
                    if (
                        col_name not in self._col_name_to_idx
                        and col_name not in new_col_name_set
                    ):
                        new_col_names.append(col_name)
                        new_col_name_set.add(col_name)
        if new_col_names:
            if self._glove_embedder is None:
                self._glove_embedder = GloveTextEmbedding(device="cpu")
            with torch.no_grad():
                new_embeds = self._glove_embedder(new_col_names)  # [N, 300]
            existing = self._col_glove_embeddings
            for i, name in enumerate(new_col_names):
                self._col_name_to_idx[name] = existing.shape[0] + i
            # Re-register with the existing buffer's device so a model
            # already moved to GPU stays on GPU.
            target_device = (
                existing.device if existing.numel() > 0 else new_embeds.device
            )
            cat = torch.cat(
                [existing.to(new_embeds.device), new_embeds], dim=0,
            ).to(target_device)
            self.register_buffer("_col_glove_embeddings", cat)
        self._num_col_semantic_cols.data = (
            self._num_col_semantic_cols + len(new_col_names)
        )

    def _extend_emb_dims(self, new_dims: set) -> None:
        """Add Linear(d, channels) projectors for embedding dims that
        aren't yet registered on the SharedEmbeddingEncoder. Untrained
        weights -- adoption-time emb_dims pay this price (the dim's
        projection starts at default Kaiming init)."""
        enc = self.table_agnostic_encoder.encoders[
            str(torch_frame.embedding)
        ]
        for d in new_dims:
            key = str(d)
            if key not in enc.projectors:
                enc.projectors[key] = nn.Linear(d, self.channels)

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

        Column ordering is defined by TableAgnosticStypeEncoder.iter_active_stypes
        to stay aligned with the value encoding concatenation order.

        Known columns are batch-indexed from the precomputed buffer. Unseen
        columns are computed on-the-fly via GloVe and cached for the lifetime
        of the process (avoids repeated CPU inference across batches in DDP).
        """
        if not hasattr(big_tf, 'col_names_dict'):
            return None

        # Collect column names in the same order as value encoding
        ordered_col_names: List[str] = []
        for stype in self.table_agnostic_encoder.iter_active_stypes(big_tf.feat_dict):
            if stype in big_tf.col_names_dict:
                ordered_col_names.extend(big_tf.col_names_dict[stype])

        if not ordered_col_names:
            return None

        # Resolve unseen columns: compute GloVe once, cache forever
        has_unseen = False
        for name in ordered_col_names:
            if name not in self._col_name_to_idx:
                has_unseen = True
                if name not in self._col_unseen_cache:
                    if self._glove_embedder is not None:
                        with torch.no_grad():
                            vec = self._glove_embedder([name])[0]
                    else:
                        vec = torch.zeros(300)
                    self._col_unseen_cache[name] = vec  # stored on CPU

        if not has_unseen:
            # Fast path: all columns known — single batch index
            idx_tensor = torch.tensor(
                [self._col_name_to_idx[n] for n in ordered_col_names],
                dtype=torch.long, device=device,
            )
            glove_stack = self._col_glove_embeddings[idx_tensor]
        else:
            # Mixed path: gather from buffer + cache
            vecs: List[Tensor] = []
            for name in ordered_col_names:
                if name in self._col_name_to_idx:
                    vecs.append(self._col_glove_embeddings[self._col_name_to_idx[name]])
                else:
                    vecs.append(self._col_unseen_cache[name].to(device))
            glove_stack = torch.stack(vecs, dim=0)

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

        if self._num_col_semantic_cols > 0 and len(self._col_name_to_idx) == 0:
            raise RuntimeError(
                "Model was trained with column semantic embeddings for "
                f"{self._num_col_semantic_cols.item()} column(s), but "
                "col_names_dict was not provided at construction time. "
                "Pass the same col_names_dict used during training to "
                "NeighborTfsEncoder."
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
            if col_sem is not None:
                if col_sem.shape[0] == x_cols.shape[1]:
                    x_cols = x_cols + col_sem.unsqueeze(0)  # broadcast over batch
                else:
                    warnings.warn(
                        f"Column semantic count ({col_sem.shape[0]}) != value "
                        f"column count ({x_cols.shape[1]}) for table type "
                        f"{t_int}. Skipping column semantic embeddings for "
                        f"this batch. This may indicate a data pipeline bug.",
                        stacklevel=2,
                    )

            batch_size = x_cols.size(0)

            cls_tokens = self.cls_embedding.expand(batch_size, -1, -1)

            x_seq = torch.cat([cls_tokens, x_cols], dim=1)

            # Chunk + gradient-checkpoint the shared transformer.
            #
            # Why chunk: (1) CUDA's efficient-attention kernel caps
            # batch at 65535. (2) A 4-layer transformer's per-sample
            # activations (~670 KB) accumulate -- with batch=512 *
            # K=300 = 153,600 slots and a dominant type taking most
            # of them, raw activation accumulation hits ~100 GiB,
            # 2.5x the A100 40 GiB cap. Plain chunking only bounds
            # *peak concurrent* memory; autograd still pins every
            # chunk's activations until backward(), so the total
            # accumulated graph still OOMs.
            #
            # Why checkpoint: torch.utils.checkpoint frees each
            # chunk's intermediate activations after forward and
            # re-runs the forward during backward to regenerate
            # them. Peak memory drops to ONE chunk's worth (~2.7
            # GiB at chunk=4096) instead of ALL chunks summed.
            # The cost is ~2x compute on this transformer only;
            # since GT convolutions dominate per-step time, the
            # overall slowdown is ~5-10%. Math is bit-for-bit
            # identical -- gradient values are recomputed, not
            # changed.
            #
            # Eval/inference (self.training == False) takes the
            # one-shot path; no grad graph means no accumulation
            # to defend against, and skipping checkpoint saves the
            # extra forward.
            _TF_CHUNK = 4096
            if self.training and x_seq.size(0) > _TF_CHUNK:
                from torch.utils.checkpoint import checkpoint as ckpt
                chunks = x_seq.split(_TF_CHUNK, dim=0)
                x_out = torch.cat(
                    [
                        ckpt(self.shared_transformer, c, use_reentrant=False)
                        for c in chunks
                    ],
                    dim=0,
                )
            else:
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