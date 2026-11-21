"""The refinement path uses z_tilde = sg(z_hat) + delta_z.

The second-pass action loss must train the proposal-conditioned residual
branch through the addition without sending gradients into the base future
prediction z_hat, while the JEPA loss still reaches z_hat.
"""

from pathlib import Path

import torch
import yaml

from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p1.data.synthetic import SyntheticP1Dataset
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.types import P1Batch

FIXTURE = Path(__file__).parent / "fixtures" / "p1_smoke.yaml"


def _build():
    values = yaml.safe_load(FIXTURE.read_text())
    for key in ("jepa_horizons", "jepa_token_counts"):
        values[key] = tuple(values[key])
    config = P1Config(**values)
    torch.manual_seed(0)
    model = P1Model(config, qwen_backend="tiny")
    model.train()
    dataset = SyntheticP1Dataset(config, length=2, seed=0)
    batch = P1Batch(**collate_p1([dataset[0], dataset[1]]))
    return model, batch


def _run(model, batch, step):
    captured = {}
    original = model.predictor.forward

    def recording_forward(*args, **kwargs):
        predictions, context = original(*args, **kwargs)
        for value in predictions.values():
            value.retain_grad()
        captured["predictions"] = predictions
        return predictions, context

    model.predictor.forward = recording_forward
    try:
        output = model(batch, global_step=step)
    finally:
        model.predictor.forward = original
    return output, captured["predictions"]


def test_refined_action_loss_does_not_reach_base_prediction():
    torch.set_num_threads(1)
    model, batch = _build()
    # Open the staged freezes so the predictor is trainable.
    step = max(model.config.freeze_memory_steps, model.config.freeze_qwen_steps,
               model.config.freeze_vjepa_steps) + 1
    output, predictions = _run(model, batch, step)
    output["loss/action_dit_refined"].backward()

    for horizon, value in predictions.items():
        assert value.grad is None or torch.count_nonzero(value.grad) == 0, horizon
    residual_grads = [p.grad for p in model.video_residual.parameters() if p.requires_grad]
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in residual_grads)


def test_jepa_loss_still_trains_base_prediction():
    torch.set_num_threads(1)
    model, batch = _build()
    step = max(model.config.freeze_memory_steps, model.config.freeze_qwen_steps,
               model.config.freeze_vjepa_steps) + 1
    output, predictions = _run(model, batch, step)
    output["loss/jepa"].backward()
    assert any(
        value.grad is not None and torch.count_nonzero(value.grad) > 0
        for value in predictions.values()
    )
