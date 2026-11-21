"""Small deterministic dataset used by the smoke test."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..config import P0Config


class SyntheticP0Dataset(Dataset):
    def __init__(self, config: P0Config, length: int = 8, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        n = config.vjepa_views * 64
        self.samples = []
        for _ in range(length):
            self.samples.append(
                {
                    "obs_features": torch.randn(n, config.vjepa_dim, generator=generator),
                    "future_features": {
                        h: torch.randn(max(k * 2, 8), config.vjepa_dim, generator=generator)
                        for h, k in zip(config.jepa_horizons, config.jepa_token_counts)
                    },
                    "actions": torch.randn(config.action_horizon, config.action_dim, generator=generator),
                    "action_history": torch.randn(config.memory_action_history, config.action_dim, generator=generator),
                    "text_features": torch.randn(4, config.qwen_dim, generator=generator),
                    "proprio": torch.randn(config.proprio_dim, generator=generator),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
