"""Grouped-query causal self-attention with a manual (reference) path and an SDPA path."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.config import ModelConfig
from model.layers import apply_rope


class KVCache:
    """Per-layer K/V tensors for incremental decoding.

    Stores the n_kv_heads (5) un-expanded heads, shape (B, n_kv_heads, S, head_dim):
    with GQA the cache is half the size an MHA cache would be. The expansion to 10 heads
    happens on the fly inside attention.
    """

    def __init__(self, n_layers: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        """Number of positions already cached."""
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new k/v (B, n_kv, T, hd) for one layer and return the full cached k/v."""
        if self.k[layer] is not None:
            k = torch.cat((self.k[layer], k), dim=2)
            v = torch.cat((self.v[layer], v), dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


def causal_mask(t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """Additive mask (t_q, t_k): 0 where allowed, -inf above the diagonal.

    Queries are the last t_q of the t_k positions (t_k > t_q when a KV cache holds a
    prefix), so query i may see keys 0 .. i + (t_k - t_q).
    """
    return torch.triu(torch.full((t_q, t_k), float("-inf"), device=device), diagonal=t_k - t_q + 1)


class GQAttention(nn.Module):
    """Causal self-attention with grouped K/V heads (10 query heads, 5 K/V heads).

    impl "manual" spells out softmax(q·kᵀ/√d + mask)·v and is the reference used to
    explain the model; it can also return the attention weights. impl "sdpa" calls
    F.scaled_dot_product_attention and is what training uses. tests/test_model.py
    checks that the two agree.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv_heads, self.n_rep = cfg.n_heads, cfg.n_kv_heads, cfg.n_rep
        self.head_dim = cfg.head_dim
        self.impl = cfg.attn_impl
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)     # 640 -> 640
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)  # 640 -> 320
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)  # 640 -> 320
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)     # 640 -> 640

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, T, d) -> (output (B, T, d), attention weights (B, H, T, S) or None).

        cos/sin cover the positions of x (offset by the cache length when decoding).
        """
        B, T, _ = x.shape
        # x (B,T,640) -> q (B,T,10,64) -> (B,10,T,64)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # x (B,T,640) -> k, v (B,T,5,64) -> (B,5,T,64)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE on q and k only
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)              # cache keeps 5 heads: (B,5,S,64)
        S = k.shape[2]
        # expand K/V to 10 heads: (B,5,S,64) -> (B,10,S,64); heads 2j and 2j+1 share K/V head j
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        attn = None
        impl = "manual" if return_attn else self.impl
        if impl == "manual":
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B,10,T,S)
            scores = scores + causal_mask(T, S, x.device)
            attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            y = attn @ v                                                    # (B,10,T,64)
        else:
            drop = self.dropout if self.training else 0.0
            if T == S:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
            elif T == 1:
                # one new query attends to every cached key; nothing to mask
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
            else:
                # is_causal aligns the mask top-left, which is wrong with a cached prefix
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(T, S, x.device).to(q.dtype),
                                                   dropout_p=drop)
        # (B,10,T,64) -> (B,T,10,64) -> (B,T,640) -> o_proj
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y), (attn if return_attn else None)
