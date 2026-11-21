"""Reject unrelated, incomplete or unfinished checkpoints before P1 starts."""

import copy

import pytest
import torch
from torch import nn

from thinkwam_p0.training.warm_start import (
    paper_p0_config, restore_shared_p0_weights, validate_p0_warm_start,
)


class SmallP1(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("qwen", "memory", "predictor", "qwen_action_to_context", "memory_to_action", "action_dit"):
            setattr(self, name, nn.Linear(2, 2))
        self.video_residual = nn.Linear(2, 2)
        self.vjepa = nn.Linear(2, 2)


def checkpoint(dataset="libero"):
    config = paper_p0_config(dataset)
    model = SmallP1()
    state = {"step": config.max_steps, "config": vars(config).copy(), "model": {
        key: torch.full_like(value, 3) for key, value in model.state_dict().items()
        if not key.startswith(("vjepa.", "video_residual."))
    }}
    return model, config, state


@pytest.mark.parametrize("dataset,steps", [("libero", 40000), ("robocasa24", 20000)])
def test_complete_matching_final_p0_loads_shared_modules_only(dataset, steps):
    model, config, state = checkpoint(dataset)
    original_new = model.video_residual.weight.detach().clone()
    original_encoder = model.vjepa.weight.detach().clone()
    assert state["step"] == steps
    validate_p0_warm_start(state, config)
    summary = restore_shared_p0_weights(model, state)
    assert summary["loaded"] == 12
    torch.testing.assert_close(model.qwen.weight, torch.full((2, 2), 3.))
    torch.testing.assert_close(model.video_residual.weight, original_new)
    torch.testing.assert_close(model.vjepa.weight, original_encoder)


@pytest.mark.parametrize("change", ["dataset", "step", "budget", "loss", "extra_field", "config", "model"])
def test_wrong_recipe_or_incomplete_metadata_is_rejected(change):
    _, config, state = checkpoint()
    if change == "dataset":
        state["config"]["dataset"] = "robocasa24"
    elif change == "step":
        state["step"] = 39999
    elif change == "budget":
        state["config"]["max_steps"] = 75000
    elif change == "loss":
        state["config"]["sigreg_loss_weight"] = .02
    elif change == "extra_field":
        state["config"]["plus_mixture_fraction"] = .35
    else:
        state.pop(change)
    with pytest.raises(ValueError):
        validate_p0_warm_start(state, config)


@pytest.mark.parametrize("change", ["empty", "missing", "shape", "extra"])
def test_partial_or_wrong_shaped_weights_fail_before_changing_any_parameter(change):
    model, _, state = checkpoint()
    before = copy.deepcopy(model.state_dict())
    if change == "empty":
        state["model"] = {}
    elif change == "missing":
        state["model"].pop("memory.weight")
    elif change == "shape":
        state["model"]["memory.weight"] = torch.zeros(3, 3)
    else:
        state["model"]["video_residual.weight"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="Incomplete P0 warm start"):
        restore_shared_p0_weights(model, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
