import math
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import train_p0
from thinkwam_p0.config import P0Config
from thinkwam_p0.data.collate import collate_p0
from thinkwam_p0.data.synthetic import SyntheticP0Dataset
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.types import P0Batch


def small_config(**kwargs):
    return P0Config(
        online_vjepa=False, qwen_lora=False, action_dit_inner_dim=32, visual_token_count=16,
        vjepa_checkpoint="", vjepa_dim=16, qwen_dim=32, memory_dim=32,
        jepa_context_dim=32, qwen_action_tokens=4, memory_slots=4,
        memory_observation_tokens=8, memory_action_history=2,
        memory_layers=1, memory_heads=4, jepa_layers=1, jepa_heads=4,
        action_dit_hidden=32, action_dit_layers=1, action_dit_heads=4,
        jepa_token_counts=(8, 4, 2), **kwargs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 regression")
def test_bf16_validation_predict_and_return_to_training():
    cfg = small_config()
    device = torch.device("cuda")
    model = P0Model(cfg, qwen_backend="tiny").to(device)
    raw = collate_p0([SyntheticP0Dataset(cfg, 1)[0]])
    # Online V-JEPA produces BF16 tokens, while Memory's linear weights are FP32.
    raw["obs_features"] = raw["obs_features"].bfloat16()
    raw["future_features"] = {h: x.bfloat16() for h, x in raw["future_features"].items()}
    assert model.memory.obs_proj.weight.dtype == torch.float32
    metrics = train_p0.evaluate_open_loop(model, [raw], device, False, 1)
    assert all(math.isfinite(value) for value in metrics.values())
    assert model.training

    batch = P0Batch(**train_p0.move_to_device(raw, device))
    model.eval()
    # Caller needs no external autocast for the public closed-loop entry point.
    actions = model.predict(
        batch.obs_features, batch.action_history,
        text_features=batch.text_features, proprio=batch.proprio,
    )
    assert actions.shape == batch.actions.shape
    assert torch.isfinite(actions).all()
    model.train()
    with model.autocast_context():
        output = model(batch)
    output["loss"].backward()
    assert torch.isfinite(model.memory.obs_proj.weight.grad).all()


@pytest.mark.parametrize("training", [False, True])
def test_validation_restores_mode_on_failure(training):
    model = P0Model(small_config(), qwen_backend="tiny").train(training)
    with pytest.raises(RuntimeError, match="zero batches"):
        train_p0.evaluate_open_loop(model, [], torch.device("cpu"), False, 1)
    assert model.training == training


def test_checkpoint_survives_validation_failure(tmp_path, monkeypatch):
    cfg = small_config(
        dataset="synthetic", output_dir=str(tmp_path), batch_size=1, num_workers=0,
        max_steps=1, tensorboard=False, eval_every=1,
    )
    dataset = SyntheticP0Dataset(cfg, 4)
    monkeypatch.setattr(train_p0, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_p0, "init_distributed", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_p0, "make_dataset_source", lambda *_: SimpleNamespace(validate=lambda: {}))
    monkeypatch.setattr(train_p0, "FeatureCacheDataset", lambda *_: dataset)
    monkeypatch.setattr(sys, "argv", ["train_p0", "--config", "unused", "--backend", "tiny", "--save-every", "1"])

    def fail_validation(*_):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(train_p0, "evaluate_open_loop", fail_validation)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        train_p0.main()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = torch.load(checkpoint_dir / "checkpoint_step_0000001.pt", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["model"] and checkpoint["optimizer"]["state"]
    assert checkpoint["scheduler"]["last_epoch"] == 1
    assert not (checkpoint_dir / "best.pt").exists()
    assert not list(checkpoint_dir.glob("*.tmp"))
