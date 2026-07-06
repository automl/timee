import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timee.model.util import (
    activation_from_str,
    validate_rope,
)


class Patch(nn.Module):
    """Segment a time series into non-overlapping patches.

    Implementation taken from Chronos-Bolt.
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py#L50
    """

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """Standardize along the last dimension with optional arcsinh nonlinearity.

    Implementation adapted from Chronos-Bolt.
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py#L71
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        eval_pos: Optional[int] = None,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            x_stats = x[:, :, :eval_pos] if eval_pos is not None else x
            loc = torch.nan_to_num(torch.nanmean(x_stats, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                (x_stats - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)


class ClassEmbedding(nn.Embedding):
    """Embed integer class labels; use a zero vector for query (test) positions.

    Implementation adapted from nanoTabICL.
    https://github.com/soda-inria/nanotabicl/blob/ddb7d4810e5b811323e77437ed106c654c07551e/model.py#L63
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.weight,
            -1 / math.sqrt(self.num_embeddings),
            1 / math.sqrt(self.num_embeddings),
        )

    def forward(self, y: torch.Tensor, eval_pos: int) -> torch.Tensor:
        train_y_emb = super().forward(y[:, :eval_pos].long().squeeze(-1))
        batch_size = y.shape[0]
        n_query = y.shape[1] - eval_pos
        query_emb = torch.zeros(
            batch_size,
            n_query,
            self.embedding_dim,
            device=train_y_emb.device,
            dtype=train_y_emb.dtype,
        )
        return torch.cat([train_y_emb, query_emb], dim=1)


class PatchEncoder(nn.Module):
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DecoderMLP(nn.Module):
    def __init__(
        self, d_model: int, hidden_dim: int, output_dim: int, activation: str, dropout: float
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation_from_str(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, activation: str, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.activation = activation_from_str(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CLSTokens(nn.Module):
    """Learnable CLS tokens prepended to each series' patch sequence."""

    def __init__(self, num_cls_tokens: int, d_model: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(num_cls_tokens, d_model))
        nn.init.trunc_normal_(self.tokens, std=0.02)

    def forward(self, batch_size: int, num_series: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).unsqueeze(0).expand(batch_size, num_series, -1, -1)


class RoPE(nn.Module):
    """Rotary position embeddings.

    Implementation taken from:
    https://github.com/amazon-science/chronos-forecasting/blob/6d68ed7c4ed2805d122d77b4660765b4089de5ca/src/chronos/chronos2/layers.py#L18
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class MHA(nn.Module):
    """Multi-head attention with optional RoPE applied to patch tokens only."""

    def __init__(self, n_heads: int, d_model: int, d_kv: int, dropout: float, use_rope: bool):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.d_mha = d_kv * n_heads

        self.q = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.k = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.v = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.o = nn.Linear(self.d_mha, d_model, bias=False)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(dim=d_kv)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b s (h d) -> b h s d", h=self.n_heads, d=self.d_kv)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b h s d -> b s (h d)")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_rope:
            validate_rope(position_ids)

        Q = self._split_heads(self.q(x))
        K = self._split_heads(self.k(x))
        V = self._split_heads(self.v(x))

        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        if self.use_rope:
            cos, sin = self.rope(x, position_ids)
            n_prefix = Q.shape[2] - position_ids.shape[1]
            Q_prefix, Q_patches = Q[:, :, :n_prefix], Q[:, :, n_prefix:]
            K_prefix, K_patches = K[:, :, :n_prefix], K[:, :, n_prefix:]
            Q_patches, K_patches = RoPE.apply_rotary_pos_emb(
                Q_patches, K_patches, cos, sin, unsqueeze_dim=1
            )
            Q = torch.cat([Q_prefix, Q_patches], dim=2)
            K = torch.cat([K_prefix, K_patches], dim=2)

        attn_output = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0,
        )
        return self.o(self._combine_heads(attn_output))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_kv: int,
        mlp_hidden_dim: int,
        dropout: float,
        activation: str,
        use_rope: bool,
    ):
        super().__init__()
        self.mha = MHA(n_heads, d_model, d_kv, dropout, use_rope)
        self.mha_layernorm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_hidden_dim, activation, dropout)
        self.mlp_layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.mha_layernorm(x + self.dropout(self.mha(x, attention_mask, position_ids)))
        x = self.mlp_layernorm(x + self.dropout(self.mlp(x)))
        return x
