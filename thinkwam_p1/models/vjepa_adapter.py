"""Residual adaptation of current-observation features from the frozen encoder."""

from __future__ import annotations

import torch
from torch import nn


class VJEPAResidualAdapter(nn.Module):
    """Zero-start token-wise adapter that preserves the P0 feature contract."""

    def __init__(self, dim: int, bottleneck: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.up(self.activation(self.down(self.norm(tokens))))
