import torch

from thinkwam_p0.config import P0Config
from thinkwam_p0.data.collate import collate_p0
from thinkwam_p0.data.synthetic import SyntheticP0Dataset
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.types import P0Batch


def test_temporal_token_decay_and_memory_action_path():
    cfg = P0Config(
        online_vjepa=False, qwen_lora=False, action_dit_inner_dim=32,
        visual_token_count=16,
        vjepa_dim=16,
        qwen_dim=32,
        memory_dim=32,
        jepa_context_dim=32,
        qwen_action_tokens=4,
        memory_slots=4,
        memory_observation_tokens=8,
        memory_action_history=2,
        memory_layers=1,
        memory_heads=4,
        jepa_layers=1,
        jepa_heads=4,
        action_dit_hidden=32,
        action_dit_layers=1,
        action_dit_heads=4,
        jepa_token_counts=(8, 4, 2),
    )
    assert cfg.jepa_token_counts == (8, 4, 2)
    model = P0Model(cfg, qwen_backend="tiny")
    raw = collate_p0([SyntheticP0Dataset(cfg, 1)[0]])
    output = model(P0Batch(**raw))
    output["loss"].backward()
    assert output["memory_tokens"].shape == (1, cfg.memory_slots, cfg.memory_dim)
    # Action loss must reach the Qwen conditioning path through cross K/V.
    assert model.qwen_action_to_context[-1].weight.grad is not None
    assert model.action_dit.blocks[0].attn.k_proj_weight.grad is not None


def test_qwen_and_jepa_context_token_compression():
    cfg = P0Config(
        online_vjepa=False, qwen_lora=False, action_dit_inner_dim=32,
        vjepa_dim=16,
        qwen_dim=32,
        visual_token_count=16,
        qwen_visual_tokens=4,
        jepa_observation_tokens=4,
        memory_dim=32,
        jepa_context_dim=32,
        qwen_action_tokens=4,
        memory_slots=4,
        memory_observation_tokens=4,
        memory_action_history=2,
        memory_layers=1,
        memory_heads=4,
        jepa_layers=1,
        jepa_heads=4,
        action_dit_hidden=32,
        action_dit_layers=1,
        action_dit_heads=4,
        jepa_token_counts=(8, 4, 2),
    )
    model = P0Model(cfg, qwen_backend="tiny")
    features = torch.randn(2, 16, cfg.vjepa_dim)
    assert model._qwen_visual_input(features).shape == (2, 4, cfg.vjepa_dim)
    assert model.predictor.observation_tokens == 4
