import copy

import torch

from scripts.train_p0 import load_config
from thinkwam_p0.models.action_dit import ActionDiT


def test_time_future_and_context_influence_actions_with_checkpointing():
    torch.manual_seed(7)
    model = ActionDiT(
        action_dim=3, proprio_dim=4, context_dim=24, horizon=4,
        hidden=16, inner_dim=32, layers=4, heads=4, future_tokens=3,
        dropout=0.0, state_dropout=0.0,
    )
    checkpointed = copy.deepcopy(model)
    checkpointed.use_gradient_checkpointing = True
    context = torch.randn(2, 6, 24, requires_grad=True)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 4)
    time = torch.tensor([0.25, 0.75])

    output = model.velocity(actions, time, state, context)
    checkpoint_output = checkpointed.velocity(actions, time, state, context)
    torch.testing.assert_close(output, checkpoint_output)
    assert not torch.allclose(output, model.velocity(actions, 1 - time, state, context))
    assert not torch.allclose(output, model.velocity(actions, time, state, context + 1))
    output.square().mean().backward()
    checkpoint_output.square().mean().backward()
    for name, parameter in model.named_parameters():
        other = dict(checkpointed.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad)
    # Future slots must actually affect action outputs through self-attention.
    assert model.future.weight.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0
    model.eval()
    sampled = model.sample(context.detach(), state, steps=2)
    assert sampled.shape == actions.shape
    assert torch.isfinite(sampled).all()


def test_current_recipe_preserves_visual_grid_and_reference_action_width():
    config = load_config('configs/paper/libero_p0.yaml')
    assert config.resolved_qwen_visual_tokens == 1152
    assert config.resolved_jepa_observation_tokens == 1152
    assert config.action_context_dim == config.jepa_context_dim == 1024
    assert config.jepa_token_counts == (288, 144, 72)
    assert config.vjepa_dim == 1024
    with torch.device('meta'):
        model = ActionDiT(
            config.action_dim, config.proprio_dim, config.action_context_dim,
            hidden=config.action_dit_hidden, layers=config.action_dit_layers,
            heads=config.action_dit_heads, inner_dim=config.action_dit_inner_dim,
            future_tokens=config.action_dit_future_tokens,
        )
    assert len(model.blocks) == 16
    assert model.inner_dim == 1536
    assert model.future.weight.shape == (32, 1536)
    assert model.out_proj.out_features == 1024
    for index, block in enumerate(model.blocks):
        assert block.attn.num_heads == 32
        assert block.attn.head_dim == 48
        assert block.use_cross_attention == (index % 2 == 0)
        assert block.attn.kdim == (1024 if index % 2 == 0 else 1536)
