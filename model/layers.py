"""Building blocks: RMSNorm, rotary position embedding (RoPE) and the SwiGLU feed-forward."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square norm: x / sqrt(mean(x²) + eps) * gain, no mean subtraction, no bias.

    The statistic is computed in fp32 even under fp16 autocast, because squaring fp16
    activations overflows easily; the result is cast back to the input dtype before the gain.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last dimension."""
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE tables, rotate-half convention (GPT-NeoX / Hugging Face LLaMA).

    Frequency i (0 <= i < head_dim/2) is theta^(-2i/head_dim). In the rotate-half
    convention, channel i is paired with channel i + head_dim/2 (not with its neighbour
    2i+1 as in the original Meta/GPT-J interleaved layout); the two are equivalent up to
    a fixed permutation of channels.

    cos/sin have shape (max_seq_len, head_dim/2), are fp32, and are non-persistent
    buffers: they follow .to(device) but are not stored in checkpoints.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)                 # (max_seq_len, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin for positions start .. start+length-1, each (length, head_dim/2)."""
        assert start + length <= self.cos.shape[0], "sequence longer than max_seq_len"
        return self.cos[start:start + length], self.sin[start:start + length]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1), where x1 and x2 are the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (B, H, T, head_dim) by its positions; cos/sin are (T, head_dim/2).

    Pair (x_i, x_{i+d/2}) at position m is rotated by angle m·theta_i, so the dot product
    of a rotated query at m and a rotated key at n depends only on m - n.
    Done in fp32 and cast back to x's dtype.
    """
    cos = torch.cat((cos, cos), dim=-1)                           # (T, head_dim)
    sin = torch.cat((sin, sin), dim=-1)
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class SwiGLU(nn.Module):
    """Gated feed-forward: w2(silu(w1(x)) * w3(x)); w1, w3: d -> hidden, w2: hidden -> d."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
