from __future__ import annotations

import torch

from scripts.train_p0 import tensorboard_metric_enabled as p0_tensorboard_metric_enabled
from scripts.train_p1 import load_config, tensorboard_metric_enabled
from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p1.data.synthetic import SyntheticP1Dataset
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.types import P1Batch


def tiny_unified_config() -> P1Config:
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
        video_model_dim=32,
        video_model_layers=1,
        video_model_heads=4,
        video_context_tokens=8,
        video_condition_steps=1,
        video_condition_train_steps=1,
        affordance_enabled=False,

        require_affordance_targets=False,

        p1_stage="joint",
    )


def test_unified_model_keeps_action_refinement_without_guidance_heads():
    config = tiny_unified_config()
    model = P1Model(config, qwen_backend="tiny")
    dataset = SyntheticP1Dataset(config, length=2)
    batch = P1Batch(**collate_p1([dataset[0], dataset[1]]))

    output = model(batch)
    assert model.affordance is None
    assert model.affordance_to_memory is None
    assert not any(name.startswith("phase") for name in model.state_dict())
    assert output["first_actions"].shape == (2, 4, 3)
    assert output["refined_actions"].shape == (2, 4, 3)
    assert output["affordance_logits"] is None
    torch.testing.assert_close(output["memory_tokens_refined"], output["memory_tokens"])
    torch.testing.assert_close(output["affordance_gate"], torch.zeros(()))
    assert float(output["loss/affordance"]) == 0.0

    output["loss"].backward()
    assert model.video_residual.head["t7"][-1].weight.grad is not None
    assert model.action_dit.out.weight.grad is not None
    assert model.qwen.visual_proj[0].weight.grad is not None
    assert model.qwen.action_placeholders.grad is not None


def test_staged_model_preserves_warm_started_qwen_heads():
    config = tiny_unified_config()
    config.p1_stage = "staged"
    config.freeze_qwen_steps = 10
    model = P1Model(config, qwen_backend="tiny")
    model.set_stage(20)
    assert not model.qwen.visual_proj[0].weight.requires_grad
    assert not model.qwen.action_placeholders.requires_grad


def test_paper_robotwin_budget_and_joint_initialization():
    config = load_config("configs/paper/robotwin2.yaml")
    assert config.max_steps == 90000
    assert config.batch_size == 20
    assert config.gradient_accumulation_steps == 1
    assert config.p1_stage == "joint"
    assert config.p0_checkpoint == ""
    assert (config.freeze_qwen_steps, config.freeze_vjepa_steps,
            config.freeze_memory_steps) == (0, 0, 0)
    assert not config.affordance_enabled


def test_paper_robotwin_config_matches_online_three_view_contract():
    config = load_config("configs/paper/robotwin2.yaml")
    assert config.dataset == "robotwin2.0"
    assert config.online_vjepa is True
    assert config.feature_cache == ""
    assert config.robotwin_manifest == "manifest_3view.json"
    assert config.vjepa_views == 3
    assert config.action_dim == 14
    assert config.action_horizon == 32
    assert config.memory_action_history == 16
    assert config.visual_token_count == 1152
    assert config.jepa_horizons == (8, 16, 32)
    assert config.jepa_token_counts == (288, 144, 72)
    assert config.affordance_enabled is False
    assert config.require_affordance_targets is False
    assert config.p1_stage == "joint"


def test_tensorboard_omits_disabled_zero_and_zero_weight_metrics():
    config = tiny_unified_config()

    # RoboTwin disables affordance guidance, so its losses are not logged.
    assert not tensorboard_metric_enabled("loss/affordance", torch.tensor(1.0), config)
    assert not tensorboard_metric_enabled("affordance_gate", torch.tensor(0.0), config)

    # Any scalar that stays at zero is omitted, while an active non-zero loss
    # remains visible. A zero-weight objective is omitted even if computed.
    assert not tensorboard_metric_enabled("loss/action_dit", torch.tensor(0.0), config)
    assert tensorboard_metric_enabled("loss/action_dit", torch.tensor(1.0), config)
    config.sigreg_loss_weight = 0.0
    assert not tensorboard_metric_enabled("loss/sigreg", torch.tensor(1.0), config)
    config.jepa_loss_weight = 0.0
    assert not tensorboard_metric_enabled("loss/jepa", torch.tensor(1.0), config)

    # P0 uses the same filtering contract for its JEPA/SIGReg metrics.
    assert not p0_tensorboard_metric_enabled("loss/sigreg", torch.tensor(1.0), config)
