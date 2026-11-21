#!/usr/bin/env python3
"""Run one CPU optimizer update on two synthetic samples through the P0 model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from thinkwam_p0.config import P0Config
from thinkwam_p0.data.collate import collate_p0
from thinkwam_p0.data.synthetic import SyntheticP0Dataset
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.types import P0Batch


def main() -> None:
    config = P0Config(
        online_vjepa=False, qwen_lora=False, action_dit_inner_dim=32, visual_token_count=16,
        vjepa_dim=32,
        qwen_dim=64,
        memory_dim=64,
        jepa_context_dim=64,
        qwen_action_tokens=8,
        memory_slots=4,
        memory_observation_tokens=16,
        memory_action_history=4,
        memory_layers=1,
        memory_heads=4,
        jepa_layers=1,
        jepa_heads=4,
        action_dit_hidden=64,
        action_dit_layers=1,
        action_dit_heads=4,
        action_horizon=20,
        action_dim=7,
        proprio_dim=8,
        jepa_token_counts=(8, 4, 2),
    )
    model = P0Model(config, qwen_backend="tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(SyntheticP0Dataset(config, length=2), batch_size=2, collate_fn=collate_p0)
    for step, raw in enumerate(loader, 1):
        batch = P0Batch(**raw)
        output = model(batch)
        assert torch.isfinite(output["loss"]), output
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        optimizer.step()
        print({"step": step, "loss": round(float(output["loss"]), 6), "action": round(float(output["loss/action_dit"]), 6), "jepa": round(float(output["loss/jepa"]), 6)})
    assert all(p.grad is not None for p in model.qwen.parameters() if p.requires_grad)
    print("P0 smoke passed: Qwen, Memory, JEPA predictor, and ActionDiT received gradients.")


if __name__ == "__main__":
    main()
