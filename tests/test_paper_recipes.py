import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from thinkwam_p0.config import P0Config
from thinkwam_p0.models.predictor import TemporalJEPA
from thinkwam_p1.config import P1Config

ROOT = Path(__file__).resolve().parents[1]


def _load(name, cls):
    values = yaml.safe_load((ROOT / "configs/paper" / name).read_text())
    for key in ("jepa_horizons", "jepa_token_counts"):
        values[key] = tuple(values[key])
    return cls(**values)


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_jepa_loss_is_mean_token_cosine_distance_with_equal_horizon_weights():
    predictor = TemporalJEPA(
        8, 8, 8, horizons=(1, 3), token_counts=(4, 1),
        context_dim=8, layers=1, heads=2, projection_hidden=16,
    )
    predictions = {1: torch.randn(2, 4, 8), 3: torch.randn(2, 1, 8)}
    targets = {h: torch.randn(2, 4, 8) for h in predictions}
    losses = predictor.loss(predictions, targets)
    expected = []
    for horizon, count in ((1, 4), (3, 1)):
        target = predictor.target_tokens(targets[horizon], count)
        distance = 1.0 - (F.normalize(predictions[horizon], dim=-1) * target).sum(dim=-1)
        expected.append(distance.mean())
        assert torch.allclose(losses[f"jepa/t{horizon}"], expected[-1], atol=1e-6)
    assert torch.allclose(losses["jepa/total"], torch.stack(expected).mean(), atol=1e-6)


@pytest.mark.parametrize(
    "name,steps,lr,min_lr,batch",
    [("libero_p0.yaml", 40_000, 1e-4, 1e-5, 24), ("robocasa24_p0.yaml", 20_000, 1e-4, 1e-5, 24)],
)
def test_paper_p0_budgets(name, steps, lr, min_lr, batch):
    config = _load(name, P0Config)
    assert (config.max_steps, config.learning_rate, config.min_learning_rate, config.batch_size) == (
        steps, lr, min_lr, batch,
    )
    assert config.action_loss_weight == 1.0 and config.jepa_loss_weight == 0.5
    assert config.sigreg_loss_weight == 0.0 and config.weight_decay == 0.0
    assert (config.optimizer_beta1, config.optimizer_beta2) == (0.9, 0.95)
    assert config.jepa_token_counts == (288, 144, 72) and config.memory_slots == 16


def test_paper_p1_recipes_match_the_configuration_table():
    libero = _load("libero_p1.yaml", P1Config)
    robotwin = _load("robotwin2.yaml", P1Config)
    robocasa = _load("robocasa24_p1.yaml", P1Config)
    expected = [
        (libero, dict(steps=40_000, lr=1e-4, min_lr=1e-5, batch=24, horizon=16, action=7, history=8,
                     horizons=(7, 15, 31), views=2, weights=(1.2, 1.4, 0.5, 0.2), aff=0.05)),
        (robotwin, dict(steps=90_000, lr=5e-5, min_lr=5e-6, batch=20, horizon=32, action=14, history=16,
                       horizons=(8, 16, 32), views=3, weights=(1.2, 1.4, 0.5, 0.2), aff=0.0)),
        (robocasa, dict(steps=40_000, lr=1e-4, min_lr=1e-6, batch=20, horizon=16, action=7, history=8,
                       horizons=(7, 15, 31), views=3, weights=(1.0, 1.0, 0.5, 0.2), aff=0.1)),
    ]
    for config, value in expected:
        assert (config.max_steps, config.learning_rate, config.min_learning_rate, config.batch_size) == (
            value["steps"], value["lr"], value["min_lr"], value["batch"],
        )
        # One LR schedule for every trainable module.
        assert {config.memory_learning_rate, config.qwen_lora_learning_rate,
                config.vjepa_adapter_learning_rate} == {config.learning_rate}
        assert (config.action_horizon, config.action_dim, config.memory_action_history) == (
            value["horizon"], value["action"], value["history"],
        )
        assert config.jepa_horizons == value["horizons"] and config.jepa_token_counts == (288, 144, 72)
        assert config.vjepa_views == value["views"] and config.memory_slots == 16
        assert (config.first_action_loss_weight, config.action_refinement_loss_weight,
                config.jepa_loss_weight, config.video_residual_loss_weight) == value["weights"]
        assert config.affordance_loss_weight == value["aff"]
        assert config.video_condition_train_steps == 1 and config.flow_steps == 4
        assert config.text_max_length == 128 and config.sigreg_loss_weight == 0.0
        assert not any(key.startswith(("phase_", "plus_", "robocasa_event", "task_sampling"))
                       for key in vars(config))
    assert libero.p1_stage == "staged" and libero.freeze_qwen_steps == 10_000
    assert libero.affordance_memory_start_step == 0 and libero.affordance_memory_max_scale == 0.05
    assert libero.affordance_loss_ramp_steps == 3000 and libero.image_augmentation_probability == 0.5
    assert not robotwin.affordance_enabled and not robotwin.image_augmentations
    assert robotwin.p1_stage == "joint" and not robotwin.p0_checkpoint
    assert robocasa.p1_stage == "joint" and robocasa.freeze_memory_steps == 0
    assert robocasa.affordance_memory_ramp_steps == 0 and robocasa.affordance_upstream_ramp_steps == 0
    assert robocasa.affordance_memory_max_scale == robocasa.affordance_upstream_max_scale == 0.05


def test_robotwin_evaluation_plan(tmp_path, monkeypatch):
    make_plan = _module(ROOT / "scripts/robotwin_eval/make_plan.py")
    assert len(make_plan.ROBOTWIN20_TASKS) == len(set(make_plan.ROBOTWIN20_TASKS)) == 20
    config = _load("robotwin2.yaml", P1Config)
    config.robotwin_stats_path = str(tmp_path / "stats.json")
    config.robotwin_frame_cache = str(tmp_path)
    Path(config.robotwin_stats_path).write_text('{}')
    (tmp_path / config.robotwin_manifest).write_text('{}')
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"step": 90000, "config": vars(config),
                "model": {"fixture": torch.ones(1)}}, checkpoint)
    monkeypatch.setattr(sys, "argv", ["make_plan", "--root", str(tmp_path / "eval"),
                        "--checkpoint", str(checkpoint), "--simulator", str(tmp_path)])
    make_plan.main()
    plan = yaml.safe_load((tmp_path / "eval/plan.json").read_text())
    assert plan["replan_steps"] == 32 and plan["episodes_per_task"] == 100
    assert plan["planned_per_condition"] == 2000 and len(plan["jobs"]) == 40


def test_robocasa24_evaluation_schedule():
    protocol = _module(ROOT / "scripts/robocasa24_eval/protocol.py")
    cases = protocol.planned_cases(8)
    assert len(protocol.TASK_MAX_STEPS) == 24 and len(cases) == 1200
    assert len({case["case_id"] for case in cases}) == 1200
    assert protocol.CHUNK == protocol.REPLAN_STEPS == 16
