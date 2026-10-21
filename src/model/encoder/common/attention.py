"""Attention blocks for anchor cross-attention and geometry-sparse refinement."""
import warnings

import torch
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_MAX_SDPA_PACKED_BATCH = 16_384
_SDPA_PACKED_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
_POSITION_TOPK_MISSING_DEPENDENCIES = []

try:
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None
    _POSITION_TOPK_MISSING_DEPENDENCIES.append("pytorch3d.ops.knn_points")

from .gmae import FeedForward

if _POSITION_TOPK_MISSING_DEPENDENCIES:
    warnings.warn(
        "RLSplat geometry routing requires dependencies "
        f"that are not available: {', '.join(_POSITION_TOPK_MISSING_DEPENDENCIES)}. "
        "Model construction will raise RuntimeError instead of silently using "
        "slow fallback paths.",
        RuntimeWarning,
        stacklevel=2,
    )


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm_query = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def project_context(self, context: Tensor) -> tuple[Tensor, Tensor]:
        context = self.norm_context(context)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        return k, v

    def forward_projected(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """``key_mask`` is [batch, context] and True where a key participates.

        Every row must keep at least one key: softmax over an all-masked row
        is NaN. Both pinned backends accept a boolean attn_mask.
        """
        if key.shape != value.shape:
            raise ValueError(
                "Projected key and value must have matching shapes, got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if query.shape[:-2] != key.shape[:-2]:
            raise ValueError(
                "Query and projected context must have matching batch shapes, "
                f"got {tuple(query.shape)} and {tuple(key.shape)}"
            )
        attn_mask = None
        if key_mask is not None:
            if key_mask.shape != key.shape[:2]:
                raise ValueError(
                    "key_mask must have shape [batch, context], got "
                    f"{tuple(key_mask.shape)} against key {tuple(key.shape)}"
                )
            if not bool(key_mask.any(dim=-1).all()):
                raise ValueError("key_mask must keep at least one key per row")
            attn_mask = key_mask[:, None, None, :]
        query = self.norm_query(query)
        q = self.to_q(query)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, key, value),
        )
        dropout_p = self.dropout.p if self.training else 0.0
        with sdpa_kernel(_SDPA_PACKED_BACKENDS):
            if q.shape[0] > _MAX_SDPA_PACKED_BATCH:
                num_chunks = -(-q.shape[0] // _MAX_SDPA_PACKED_BATCH)
                mask_chunks = (
                    attn_mask.split(_MAX_SDPA_PACKED_BATCH)
                    if attn_mask is not None
                    else [None] * num_chunks
                )
                out = torch.cat(
                    [
                        F.scaled_dot_product_attention(
                            q_chunk,
                            k_chunk,
                            v_chunk,
                            attn_mask=mask_chunk,
                            dropout_p=dropout_p,
                        )
                        for q_chunk, k_chunk, v_chunk, mask_chunk in zip(
                            q.split(_MAX_SDPA_PACKED_BATCH),
                            k.split(_MAX_SDPA_PACKED_BATCH),
                            v.split(_MAX_SDPA_PACKED_BATCH),
                            mask_chunks,
                            strict=True,
                        )
                    ],
                    dim=0,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)




class GeometrySparseGaussianAttention(nn.Module):
    """Refine packed gaussians by attending to their nearest 3D gaussian
    neighbours. The neighbourhood is geometric (3D KNN, chosen upstream); the
    attention itself is plain softmax(q k / sqrt(d)) v over it, so geometry
    enters only through which neighbours are seen."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        neighbor_count: int,
        ffn: bool = False,
    ) -> None:
        super().__init__()
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")
        self.heads = heads
        self.dim_head = dim_head
        self.neighbor_count = neighbor_count
        inner_dim = heads * dim_head

        self.gaussian_norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_gaussian_residual = nn.Linear(inner_dim, dim)
        nn.init.zeros_(self.to_gaussian_residual.weight)
        nn.init.zeros_(self.to_gaussian_residual.bias)
        self.ff = FeedForward(dim, dim * 2) if ffn else None
        if self.ff is not None:
            nn.init.zeros_(self.ff.net[-2].weight)
            nn.init.zeros_(self.ff.net[-2].bias)

    def forward(
        self,
        gaussian_tokens: Tensor,
        gaussian_positions: Tensor,
        neighbor_idx: Tensor,
        neighbor_mask: Tensor,
        anchor_flat_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_gaussians = gaussian_tokens.shape[0]
        if gaussian_positions.shape != (num_gaussians, 3):
            raise ValueError(
                "gaussian_positions must have shape [total_gaussians, 3], got "
                f"{tuple(gaussian_positions.shape)}"
            )
        if neighbor_idx.shape != neighbor_mask.shape:
            raise ValueError(
                "neighbor_idx and neighbor_mask must have matching shapes, got "
                f"{tuple(neighbor_idx.shape)} and {tuple(neighbor_mask.shape)}"
            )
        if neighbor_idx.ndim != 2 or neighbor_idx.shape[0] != num_gaussians:
            raise ValueError(
                "neighbor_idx must have shape [total_gaussians, neighbors], got "
                f"{tuple(neighbor_idx.shape)}"
            )
        if anchor_flat_idx.shape != (num_gaussians,):
            raise ValueError(
                "anchor_flat_idx must have shape [total_gaussians], got "
                f"{tuple(anchor_flat_idx.shape)}"
            )
        if num_gaussians == 0:
            return gaussian_tokens, gaussian_positions

        qkv = self.to_qkv(self.gaussian_norm(gaussian_tokens))
        qkv = qkv.view(num_gaussians, 3, self.heads, self.dim_head)
        query, key, value = qkv.unbind(dim=1)
        neighbor_key = key[neighbor_idx]
        neighbor_value = value[neighbor_idx]

        attention_logits = torch.einsum(
            "nhd,nkhd->nkh",
            query,
            neighbor_key,
        ) * (self.dim_head ** -0.5)
        attention_logits = attention_logits.masked_fill(
            ~neighbor_mask.unsqueeze(-1),
            torch.finfo(attention_logits.dtype).min,
        )
        attention = attention_logits.float().softmax(dim=1).to(
            dtype=attention_logits.dtype
        )
        attention = attention * neighbor_mask.unsqueeze(-1)

        message = torch.einsum(
            "nkh,nkhd->nhd",
            attention,
            neighbor_value,
        ).reshape(num_gaussians, self.heads * self.dim_head)
        has_neighbor = neighbor_mask.any(dim=-1, keepdim=True).to(gaussian_tokens.dtype)
        gaussian_tokens = gaussian_tokens + has_neighbor * self.to_gaussian_residual(message)
        if self.ff is not None:
            gaussian_tokens = gaussian_tokens + self.ff(gaussian_tokens)
        return gaussian_tokens, gaussian_positions


class PositionTopKCrossBlock(nn.Module):
    """Update packed gaussians from their selected anchor-token memory."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()
        self.anchor_cross_attn = CrossAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
        )
        self.cross_ff = FeedForward(dim, dim * 2)

    def forward_anchor_cross_indexed_packed(
        self,
        gaussian_tokens: Tensor,
        anchor_tokens: Tensor,
        selected_anchor_flat_idx: Tensor,
        anchor_mask: Tensor | None = None,
    ) -> Tensor:
        if gaussian_tokens.shape[0] == 0:
            return gaussian_tokens
        anchor_key, anchor_value = self.anchor_cross_attn.project_context(
            anchor_tokens,
        )
        anchor_key = anchor_key.flatten(0, 1)[selected_anchor_flat_idx]
        anchor_value = anchor_value.flatten(0, 1)[selected_anchor_flat_idx]
        gaussian_query = gaussian_tokens.unsqueeze(1)
        gaussian_query = self.anchor_cross_attn.forward_projected(
            gaussian_query,
            anchor_key,
            anchor_value,
            key_mask=anchor_mask,
        ) + gaussian_query
        gaussian_query = self.cross_ff(gaussian_query) + gaussian_query
        return gaussian_query.squeeze(1)
