"""Observation-action memory used by both JEPA and ActionDiT."""

from __future__ import annotations

import torch
from torch import nn


def _uniform_pool(tokens: torch.Tensor, count: int) -> torch.Tensor:
    if tokens.shape[1] == count:
        return tokens
    if tokens.shape[1] < count:
        repeat = (count + tokens.shape[1] - 1) // tokens.shape[1]
        return tokens.repeat(1, repeat, 1)[:, :count]
    indices = torch.linspace(0, tokens.shape[1] - 1, count, device=tokens.device).long()
    return tokens.index_select(1, indices)


class ObservationActionMemory(nn.Module):
    """Build m_t from O_t and recent executed actions.

    Each learned slot cross-attends to the configured number of uniformly
    selected current V-JEPA tokens and recent executed-action tokens. The
    paper recipes use 16 slots, 256 visual tokens, and a history of 8 actions
    for LIBERO/RoboCasa24 or 16 for RoboTwin.
    Language stays in the Qwen path and is combined with m_t only at the
    ActionDiT cross-attention interface.
    """

    def __init__(
        self,
        vjepa_dim: int,
        action_dim: int,
        memory_dim: int = 1024,
        slots: int = 16,
        observation_tokens: int = 64,
        action_history: int = 8,
        layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.observation_tokens = observation_tokens
        self.action_history = action_history
        self.obs_proj = nn.Linear(vjepa_dim, memory_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, slots, memory_dim) * 0.02)
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(memory_dim, heads, batch_first=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(memory_dim) for _ in range(layers * 2)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(memory_dim, 4 * memory_dim),
                    nn.GELU(),
                    nn.Linear(4 * memory_dim, memory_dim),
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        obs_tokens: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        obs = self.obs_proj(_uniform_pool(obs_tokens, self.observation_tokens))
        actions = self.action_proj(action_history[:, -self.action_history :])
        source = torch.cat([obs, actions], dim=1)
        memory = self.slots.expand(obs.shape[0], -1, -1)
        for i, attn in enumerate(self.cross_attn):
            x = self.norms[2 * i](memory)
            update, _ = attn(x, source, source, need_weights=False)
            memory = memory + update
            memory = memory + self.ffns[i](self.norms[2 * i + 1](memory))
        return memory
