from __future__ import annotations

import pytest
import torch

from scripts.train_p0 import load_config as load_p0_config
from scripts.train_p1 import load_config as load_p1_config
from thinkwam_p0.models.p0 import P0Model

from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p1.data.synthetic import SyntheticP1Dataset
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.types import P1Batch


def tiny_config() -> P1Config:
    return P1Config(
        online_vjepa=False, qwen_lora=False, action_dit_inner_dim=32,
        vjepa_dim=32,
        qwen_dim=32,
        visual_token_count=16,
        vjepa_views=1,
        vjepa_image_size=64,
        vjepa_patch_size=16,
        qwen_action_tokens=4,
        action_dim=3,
        proprio_dim=4,
        action_horizon=4,
        action_dit_hidden=32,
        action_dit_layers=1,
        action_dit_heads=4,
        flow_steps=2,
        memory_dim=32,
        memory_slots=4,
        memory_observation_tokens=8,
        memory_action_history=3,
        memory_layers=1,
        memory_heads=4,
        jepa_context_dim=32,
        jepa_layers=1,
        jepa_heads=4,
        jepa_horizons=(7, 15, 31),
        jepa_token_counts=(4, 2, 1),
        affordance_decoder_dim=32,
        affordance_decoder_layers=1,
        affordance_decoder_heads=4,
        video_model_dim=32,
        video_model_layers=1,
        video_model_heads=4,
        video_context_tokens=8,
        video_condition_steps=1,
        require_affordance_targets=True,
        freeze_qwen_steps=0,
        freeze_vjepa_steps=0,
        freeze_memory_steps=0,
        affordance_loss_ramp_steps=0,

    )


def test_p1_two_pass_graph_has_all_training_gradients():
    config = tiny_config()
    model = P1Model(config, qwen_backend="tiny")
    dataset = SyntheticP1Dataset(config, length=2)
    batch = P1Batch(**collate_p1([dataset[0], dataset[1]]))
    projected_zt = []
    handles = [
        projection.register_forward_pre_hook(
            lambda _module, inputs: projected_zt.append(inputs[0].detach().clone())
        )
        for projection in model.visual_to_action.values()
    ]
    output = model(batch)
    assert output["first_actions"].shape == (2, 4, 3)
    assert output["refined_actions"].shape == (2, 4, 3)
    assert output["affordance_logits"].shape == (2, 16, 1)
    assert output["affordance_logits_latent"].shape == (2, 16, 1)
    assert output["memory_tokens"].shape == (2, 4, 32)
    assert "memory_tokens_refined" in output
    assert hasattr(model, "affordance_to_memory")
    assert not hasattr(model, "phase_head")
    torch.testing.assert_close(output["memory_tokens_refined"], output["memory_tokens"])
    assert float(output["affordance_gate"]) == 0.0
    assert set(output["corrected_visual_tokens"]) == {7, 15, 31}
    assert set(output["normalized_zt_tokens"]) == {7, 15, 31}
    for tokens in output["normalized_zt_tokens"].values():
        assert torch.allclose(tokens.norm(p=2, dim=-1), torch.ones_like(tokens[..., 0]), atol=1e-5)
    assert len(projected_zt) == len(config.jepa_horizons)
    for actual, horizon in zip(projected_zt, config.jepa_horizons):
        torch.testing.assert_close(actual, output["normalized_zt_tokens"][horizon])
    # This test starts before its affordance upstream schedule opens, so this
    # loss must not yet send gradients into the shared representation.
    model.zero_grad(set_to_none=True)
    output["loss/affordance"].backward()
    assert model.affordance.head[-1].weight.grad is not None
    for module in (model.qwen, model.predictor, model.memory):
        assert all(parameter.grad is None for parameter in module.parameters())
    model.zero_grad(set_to_none=True)
    output = model(batch)
    output["loss"].backward()
    assert model.affordance.head[-1].weight.grad is not None
    assert model.predictor.context_layers[0].linear1.weight.grad is not None
    assert model.video_residual.head["t7"][-1].weight.grad is not None
    assert model.action_dit.out.weight.grad is not None
    projected_zt.clear()
    model.eval()
    sampled = model.predict(
        batch.obs_features, batch.action_history, proprio=batch.proprio,
        text_features=batch.text_features, steps=1,
    )
    assert sampled.shape == batch.actions.shape
    assert torch.isfinite(sampled).all()
    assert len(projected_zt) == len(config.jepa_horizons)
    for tokens in projected_zt:
        torch.testing.assert_close(tokens.norm(dim=-1), torch.ones_like(tokens[..., 0]))
    for handle in handles:
        handle.remove()


def test_staged_affordance_upstream_and_memory_gate_open_gradually():
    config = tiny_config()
    config.affordance_upstream_start_step = 1
    config.affordance_upstream_ramp_steps = 2
    config.affordance_upstream_max_scale = 0.1
    config.affordance_memory_start_step = 1
    config.affordance_memory_ramp_steps = 2
    config.affordance_memory_max_scale = 0.1
    model = P1Model(config, qwen_backend="tiny")
    dataset = SyntheticP1Dataset(config, length=2)
    batch = P1Batch(**collate_p1([dataset[0], dataset[1]]))
    warm = model(batch, global_step=0)
    model.zero_grad(set_to_none=True)
    warm["loss/affordance"].backward()
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    model.zero_grad(set_to_none=True)
    opened = model(batch, global_step=2)
    opened["loss/affordance"].backward()
    assert model.predictor.context_layers[0].linear1.weight.grad is not None
    assert float(opened["affordance_gate"]) > 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_action_zt_normalization_removes_magnitude_and_preserves_gradients(dtype):
    tokens = torch.tensor([[[3.0, 4.0], [-4.0, 3.0]]], dtype=dtype, requires_grad=True)
    unit = P1Model._normalize_zt_for_action(tokens)
    # This scale overflows an FP16 norm unless normalization uses FP32.
    large = P1Model._normalize_zt_for_action(tokens * 10000)
    assert unit.dtype == dtype
    torch.testing.assert_close(unit, large, atol=0.004, rtol=0.004)
    torch.testing.assert_close(unit.float().norm(dim=-1), torch.ones(1, 2), atol=0.004, rtol=0.004)
    unit[..., 0].sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum() > 0
    zero = torch.zeros_like(tokens, requires_grad=True)
    normalized_zero = P1Model._normalize_zt_for_action(zero)
    torch.testing.assert_close(normalized_zero, zero)


def test_libero_p1_shared_capacity_matches_current_p0():
    p0_config = load_p0_config("configs/paper/libero_p0.yaml")
    p1_config = load_p1_config("configs/paper/libero_p1.yaml")
    shared_dimensions = (
        "vjepa_checkpoint", "vjepa_dim", "vjepa_image_size", "vjepa_patch_size", "vjepa_views",
        "qwen_path", "qwen_dim", "qwen_action_tokens", "qwen_max_length", "text_max_length",
        "qwen_lora", "qwen_lora_rank", "qwen_lora_alpha", "qwen_lora_target_modules",
        "visual_token_count", "resolved_qwen_visual_tokens", "resolved_jepa_observation_tokens",
        "jepa_horizons", "jepa_token_counts", "jepa_context_dim", "jepa_layers", "jepa_heads",
        "memory_dim", "memory_slots", "memory_observation_tokens", "memory_action_history",
        "memory_layers", "memory_heads", "action_context_dim", "conditioning_mlp_hidden",
        "action_dim", "proprio_dim", "action_horizon", "action_dit_hidden", "action_dit_layers",
        "action_dit_heads", "action_dit_inner_dim", "action_dit_future_tokens",
    )
    for field in shared_dimensions:
        assert getattr(p1_config, field) == getattr(p0_config, field), field
    # Check full production head shapes without allocating their weights or
    # loading backbone checkpoints. Both models share the same Qwen wrapper.
    with torch.device("meta"):
        p0 = P0Model(p0_config, qwen_backend="tiny")
        p1 = P1Model(p1_config, qwen_backend="tiny")
    p1_state = p1.state_dict()
    for name, tensor in p0.state_dict().items():
        assert name in p1_state, name
        assert tensor.shape == p1_state[name].shape, name
    assert p1.video_residual.context_tokens == 256


@pytest.mark.parametrize(
    "recipe,prefetch",
    [("libero_p1", 4), ("robotwin2", 4), ("robocasa24_p1", 2)],
)
def test_p1_paper_runtime_alignment(recipe, prefetch):
    config = load_p1_config(f"configs/paper/{recipe}.yaml")
    assert config.compile_action_dit is True
    assert config.action_dit_compile_mode == "reduce-overhead"
    assert config.num_workers == 8
    assert config.dataloader_prefetch_factor == prefetch
    assert config.dataloader_pin_memory is True


def test_smoke_fixture_keeps_compilation_disabled():
    assert load_p1_config("tests/fixtures/p1_smoke.yaml").compile_action_dit is False


def test_p1_compiled_velocity_covers_rollouts_losses_and_inference(monkeypatch):
    config = tiny_config()
    model = P1Model(config, qwen_backend="tiny")
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    compile_options = []
    velocity_calls = []

    def tracked_compile(fn, **kwargs):
        compile_options.append(kwargs)

        def compiled(*args):
            velocity_calls.append(args[3].shape[1])
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", tracked_compile)
    # P0 and P1 use the same compiler options and retain checkpoint names.
    P0Model(config, qwen_backend="tiny").compile_action_dit()
    model.compile_action_dit()
    assert compile_options[0] == compile_options[1]
    model.load_state_dict(original_state, strict=True)
    assert set(model.state_dict()) == set(original_state)

    dataset = SyntheticP1Dataset(config, length=2)
    batch = P1Batch(**collate_p1([dataset[0], dataset[1]]))
    output = model(batch)
    # Both differentiable rollouts and both flow losses use the compiled core.
    assert len(velocity_calls) == 2 * config.video_condition_steps + 2
    assert len(set(velocity_calls)) == 2
    output["loss"].backward()
    assert torch.isfinite(model.action_dit.out.weight.grad).all()
    assert torch.isfinite(model.video_residual.head["t7"][-1].weight.grad).all()

    # The normal training path requests scalar metrics only. The refined
    # diagnostic rollout is not part of any loss and must be skipped.
    velocity_calls.clear()
    model.zero_grad(set_to_none=True)
    metrics = model(batch, metrics_only=True)
    assert "loss" in metrics
    assert len(velocity_calls) == config.video_condition_train_steps + 2

    velocity_calls.clear()
    model.eval()
    actions = model.predict(
        batch.obs_features, batch.action_history, proprio=batch.proprio,
        text_features=batch.text_features, steps=1,
    )
    assert len(velocity_calls) == 2
    assert actions.shape == batch.actions.shape
    assert torch.isfinite(actions).all()
