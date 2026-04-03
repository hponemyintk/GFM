import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LocalModule(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        node_only_readout=False,
        n_layers=1,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.3,
        attention_dropout_rate=0,
        local_attn_type="cross",
        num_latent_tokens=32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.node_only_readout = node_only_readout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads

        self.n_layers = n_layers

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.local_attn_type = local_attn_type
        self.num_latent_tokens = num_latent_tokens

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        encoders = [
            EncoderLayer(
                self.hidden_dim,
                self.ffn_dim,
                self.dropout_rate,
                self.attention_dropout_rate,
                self.num_heads,
            )
            for _ in range(self.n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if self.local_attn_type == "cross":
            self.latent_tokens = nn.Parameter(
                torch.randn(1, num_latent_tokens, hidden_dim) * 0.02
            )
            self.encoder_cross_attn = CrossAttentionLayer(
                hidden_dim, self.ffn_dim, dropout_rate, attention_dropout_rate, num_heads,
            )
            self.decoder_cross_attn = CrossAttentionLayer(
                hidden_dim, self.ffn_dim, dropout_rate, attention_dropout_rate, num_heads,
            )
        else:
            self.attn_layer = nn.Linear(2 * hidden_dim, 1)

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.final_ln.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        if self.local_attn_type == "cross":
            nn.init.normal_(self.latent_tokens, std=0.02)
            self.encoder_cross_attn.reset_parameters()
            self.decoder_cross_attn.reset_parameters()
        else:
            self.attn_layer.reset_parameters()

    def forward(self, batched_data, pretrain_token=False):
        tensor = self.att_embeddings_nope(batched_data)

        if self.local_attn_type == "cross":
            return self._cross_attention_forward(tensor, pretrain_token)
        else:
            return self._self_attention_forward(tensor, pretrain_token)

    def _cross_attention_forward(self, tensor, pretrain_token):
        B = tensor.shape[0]
        seed_repr = tensor[:, 0:1, :]  # [B, 1, hidden_dim]

        latent = self.latent_tokens.expand(B, -1, -1)  # [B, K_latent, hidden_dim]
        latent = self.encoder_cross_attn(latent, tensor)

        for enc_layer in self.layers:
            latent = enc_layer(latent)

        latent = self.final_ln(latent)

        if pretrain_token:
            return latent

        output = self.decoder_cross_attn(seed_repr, latent)  # [B, 1, hidden_dim]
        return output.squeeze(1)

    def _self_attention_forward(self, tensor, pretrain_token):
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)

        if pretrain_token:
            return output

        _target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

        node_tensor = split_tensor[0]
        _neighbor_tensor = split_tensor[1]

        if self.node_only_readout:
            indices = torch.arange(1, self.seq_len, 1)
            neighbor_tensor = _neighbor_tensor[:, indices]
            target = _target[:, indices]
        else:
            target = _target
            neighbor_tensor = _neighbor_tensor

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)

        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
    def reset_parameters(self):
        for layer in self.ffn_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        
        x = self.ffn_net(x)
        
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        
        return x

class CrossAttentionLayer(nn.Module):
    """Perceiver-style cross-attention: Q from query tokens, K/V from context tokens."""
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def reset_parameters(self):
        self.query_norm.reset_parameters()
        self.context_norm.reset_parameters()
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, query, context):
        # query: [B, L_q, D], context: [B, L_kv, D]
        residual = query
        q_norm = self.query_norm(query)
        c_norm = self.context_norm(context)

        Q = self.q_proj(q_norm)
        K = self.k_proj(c_norm)
        V = self.v_proj(c_norm)

        B, L_q, D = Q.shape
        L_kv = K.shape[1]
        head_dim = D // self.num_heads

        Q = Q.view(B, L_q, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L_kv, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L_kv, self.num_heads, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, L_q, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.attention_dropout(attn_output)

        query = residual + attn_output

        residual = query
        query = residual + self.ffn(self.ffn_norm(query))
        return query


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None):
        # self-attention block with flash attention 
        residual = x
        x_norm = self.self_attention_norm(x)  # [B, L, D]
        
        Q = self.q_proj(x_norm)  # [B, L, D]
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        B, L, D = Q.shape
        head_dim = D // self.num_heads
        
        # reshape Q, K, V to shape [B, num_heads, L, head_dim].
        Q = Q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        
        # PyTorch’s fast scaled dot-product attention (flash attention).
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout_rate,
            is_causal=False  
        )  # Returns [B, num_heads, L, head_dim]
        
        # reshape back to [B, L, D].
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        
        attn_output = self.out_proj(attn_output)
        attn_output = self.self_attention_dropout(attn_output)
        
        x = residual + attn_output
        
        # Feed-forward block 
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x