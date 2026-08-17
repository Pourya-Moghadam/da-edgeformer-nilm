from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding; output never depends on future samples."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, padding=0, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalBranch(nn.Module):
    """Causal depthwise-separable temporal branch."""

    def __init__(self, width: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=width,
            bias=False,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.pointwise(self.depthwise(x)))


class ResidualAdapter(nn.Module):
    def __init__(self, width: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.activation(self.down(self.norm(x))))


class LocalCausalSelfAttention(nn.Module):
    """Sliding local attention implemented without constructing a W x W mask."""

    def __init__(self, width: int, heads: int, span: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.span = span
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        padding = self.span - 1
        k_windows = F.pad(k, (0, 0, padding, 0)).unfold(2, self.span, 1)
        v_windows = F.pad(v, (0, 0, padding, 0)).unfold(2, self.span, 1)
        k_windows = k_windows.permute(0, 1, 2, 4, 3)
        v_windows = v_windows.permute(0, 1, 2, 4, 3)

        scores = torch.einsum("bhwd,bhwrd->bhwr", q, k_windows) * self.scale
        offsets = torch.arange(self.span, device=x.device)
        positions = torch.arange(steps, device=x.device).unsqueeze(1)
        invalid = offsets.unsqueeze(0) < (padding - positions)
        scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), -math.inf)
        weights = self.dropout(scores.softmax(dim=-1))
        attended = torch.einsum("bhwr,bhwrd->bhwd", weights, v_windows)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, steps, self.width)
        return self.out(attended)


class EncoderBlock(nn.Module):
    def __init__(
        self, width: int, heads: int, span: int, feedforward_width: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = LocalCausalSelfAttention(width, heads, span, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))
