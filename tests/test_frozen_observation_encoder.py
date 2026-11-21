"""The visual backbone stays frozen while observation adaptation learns."""

import torch
from torch import nn

from thinkwam_p1.models import observation_encoder


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.8)

    def forward(self, images):
        pooled = images.mean(dim=(2, 3, 4))
        return self.dropout(self.projection(pooled))[:, None, :].expand(-1, 4, -1)


def test_frozen_encoder_outputs_allow_adapter_gradients_and_stay_in_eval(monkeypatch):
    monkeypatch.setattr(observation_encoder.vit_module, 'vit_large', lambda **_: SmallBackbone())
    encoder = observation_encoder.FrozenObservationEncoder()
    encoder.train()
    assert not encoder.training and not encoder.featurizer.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    inputs = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
    first = encoder(inputs)
    torch.testing.assert_close(first, encoder(inputs), rtol=0, atol=0)
    assert not first.requires_grad and not torch.is_inference(first)
    adapter = nn.Linear(8, 3)
    adapter(first.float()).square().mean().backward()
    assert adapter.weight.grad is not None and adapter.weight.grad.abs().sum() > 0
    assert inputs.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_p1_train_keeps_base_encoder_frozen_across_validation_transitions():
    from test_p1_architecture import tiny_config
    from thinkwam_p1.models.p1 import P1Model

    config = tiny_config()
    config.freeze_qwen_steps = config.freeze_vjepa_steps = 10000
    config.freeze_memory_steps = 5000
    encoder = nn.Sequential(nn.Linear(3, config.vjepa_dim), nn.Dropout(0.8))
    model = P1Model(config, qwen_backend='tiny', vjepa_encoder=encoder)
    for step in (0, 5000, 10000):
        model.set_stage(step)
        model.eval()
        model.train()
        model.set_stage(step)
        assert not encoder.training
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        assert model.qwen.training == (step >= 10000)
        assert model.vjepa_adapter.training == (step >= 10000)
        assert model.memory.training == model.predictor.training == (step >= 5000)
