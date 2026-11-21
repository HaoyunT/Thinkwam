from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from thinkwam_p0.config import P0Config
from thinkwam_p0.data.collate import collate_p0
from thinkwam_p0.data.synthetic import SyntheticP0Dataset
from thinkwam_p0.models.action_dit import ActionDiT
from thinkwam_p0.models.loss_utils import ddp_mean_from_sum
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.models.predictor import TemporalJEPA
from thinkwam_p0.models.sigreg import SIGReg
from thinkwam_p0.types import P0Batch
from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p1.data.synthetic import SyntheticP1Dataset
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.models.video_residual import ActionConditionedVideoResidualModel
from thinkwam_p1.types import P1Batch


def small_config(config_type=P0Config):
    values = dict(
        online_vjepa=False, qwen_lora=False,
        vjepa_dim=16, qwen_dim=16, visual_token_count=8, qwen_action_tokens=2,
        action_dim=3, proprio_dim=2, action_horizon=4,
        action_dit_hidden=16, action_dit_inner_dim=16, action_dit_layers=1,
        action_dit_heads=4, flow_steps=1, memory_dim=16, memory_slots=2,
        memory_observation_tokens=4, memory_action_history=2,
        memory_layers=1, memory_heads=4, jepa_context_dim=16,
        jepa_layers=1, jepa_heads=4, jepa_token_counts=(4, 2, 1),
        conditioning_mlp_hidden=32,
    )
    if config_type is P1Config:
        values.update(
            affordance_decoder_dim=16, affordance_decoder_layers=1,
            affordance_decoder_heads=4, video_model_dim=16, video_model_layers=1,
            video_model_heads=4, video_context_tokens=4, video_condition_steps=1,
            video_condition_train_steps=1, freeze_qwen_steps=0,
            freeze_vjepa_steps=0, freeze_memory_steps=0,
        )
    return config_type(**values)


def action_model():
    return ActionDiT(
        action_dim=3, proprio_dim=2, context_dim=12, horizon=4,
        hidden=16, inner_dim=16, layers=1, heads=4, future_tokens=2,
        dropout=0.0, state_dropout=0.0,
    ).eval()


def test_action_mask_preserves_full_windows_and_excludes_padded_input():
    model = action_model()
    context, actions, state = torch.randn(2, 3, 12), torch.randn(2, 4, 3), torch.randn(2, 2)
    torch.manual_seed(41)
    legacy = model.loss(context, actions, state)
    torch.manual_seed(41)
    full = model.loss(context, actions, state, torch.ones(2, 4, dtype=torch.bool))
    torch.testing.assert_close(full, legacy, rtol=0, atol=0)

    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    altered = actions.clone()
    altered[~mask] = float("nan")
    torch.manual_seed(42)
    reference = model.loss(context, actions, state, mask)
    torch.manual_seed(42)
    masked = model.loss(context, altered, state, mask)
    torch.testing.assert_close(masked, reference, rtol=0, atol=0)
    masked.backward()
    assert torch.isfinite(model.out.weight.grad).all()


def test_action_mask_normalizes_only_real_labels_and_empty_is_differentiable(monkeypatch):
    model = action_model()
    context, actions = torch.randn(2, 3, 12), torch.randn(2, 4, 3)
    # A deterministic velocity isolates the reduction denominator from attention.
    monkeypatch.setattr(model, "velocity", lambda noisy, t, proprio, context: noisy * 0.0)
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    torch.manual_seed(15)
    noise = torch.randn_like(actions)
    expected = (actions - noise).square()[mask].mean()
    torch.manual_seed(15)
    actual = model.loss(context, actions, action_mask=mask)
    torch.testing.assert_close(actual, expected)
    actions.requires_grad_(True)
    empty = model.loss(context, actions, action_mask=torch.zeros_like(mask))
    assert empty.requires_grad and empty.item() == 0
    empty.backward()
    torch.testing.assert_close(actions.grad, torch.zeros_like(actions))


def future_modules():
    predictor = TemporalJEPA(
        8, 8, 8, horizons=(1, 3), token_counts=(2, 1),
        context_dim=8, layers=1, heads=2, projection_hidden=16,
    ).eval()
    residual = ActionConditionedVideoResidualModel(
        8, 3, hidden_dim=8, layers=1, heads=2,
        horizons=(1, 3), token_counts=(2, 1), context_tokens=2,
    ).eval()
    predictions = {1: torch.randn(2, 2, 8, requires_grad=True), 3: torch.randn(2, 1, 8, requires_grad=True)}
    updates = {h: torch.randn_like(p, requires_grad=True) for h, p in predictions.items()}
    targets = {h: torch.randn(2, 4, 8) for h in predictions}
    return predictor, residual, predictions, updates, targets


def test_future_masks_preserve_full_windows_and_ignore_invalid_targets():
    predictor, residual, predictions, updates, targets = future_modules()
    full = {h: torch.ones(2, dtype=torch.bool) for h in predictions}
    legacy = predictor.loss(predictions, targets)
    masked = predictor.loss(predictions, targets, future_masks=full)
    for key in legacy:
        torch.testing.assert_close(masked[key], legacy[key], rtol=0, atol=0)
    legacy = residual.loss(updates, predictions, targets)
    masked = residual.loss(updates, predictions, targets, future_masks=full)
    for key in legacy:
        torch.testing.assert_close(masked[key], legacy[key], rtol=0, atol=0)

    masks = {1: torch.tensor([True, False]), 3: torch.tensor([False, False])}
    altered = {h: target.clone() for h, target in targets.items()}
    for h, target in altered.items():
        target[~masks[h]] = float("nan")
    jepa = predictor.loss(predictions, altered, future_masks=masks)
    video = residual.loss(updates, predictions, altered, future_masks=masks)
    expected_jepa = predictor.loss({h: p[:1] for h, p in predictions.items()}, {h: t[:1] for h, t in targets.items()})
    expected_video = residual.loss({h: p[:1] for h, p in updates.items()}, {h: p[:1] for h, p in predictions.items()}, {h: t[:1] for h, t in targets.items()})
    torch.testing.assert_close(jepa["jepa/total"], expected_jepa["jepa/t1"])
    torch.testing.assert_close(video["video/residual"], expected_video["video/residual_t1"])
    assert jepa["jepa/t3"].item() == video["video/residual_t3"].item() == 0
    (jepa["jepa/total"] + video["video/residual"]).backward()
    torch.testing.assert_close(predictions[1].grad[1], torch.zeros_like(predictions[1].grad[1]))
    torch.testing.assert_close(predictions[3].grad, torch.zeros_like(predictions[3]))
    torch.testing.assert_close(updates[3].grad, torch.zeros_like(updates[3]))


def test_all_missing_futures_have_finite_zero_loss_and_backward():
    predictor, residual, predictions, updates, targets = future_modules()
    masks = {h: torch.zeros(2, dtype=torch.bool) for h in predictions}
    targets = {h: torch.full_like(t, float("nan")) for h, t in targets.items()}
    jepa = predictor.loss(predictions, targets, future_masks=masks, compute_sigreg=True)
    video = residual.loss(updates, predictions, targets, future_masks=masks)
    for value in (*jepa.values(), *video.values()):
        assert torch.isfinite(value) and value.item() == 0
    (jepa["jepa/total"] + jepa["jepa/sigreg"] + video["video/residual"]).backward()
    for tensor in (*predictions.values(), *updates.values()):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


def test_sigreg_valid_subset_and_single_valid_sample():
    regularizer = SIGReg(num_projections=8).eval()
    samples = torch.randn(3, 4, 8, requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    torch.testing.assert_close(regularizer(samples, sample_mask=mask), regularizer(samples[:, mask]))
    value = regularizer(samples, sample_mask=torch.tensor([True, False, False, False]))
    assert value.item() == 0
    value.backward()
    torch.testing.assert_close(samples.grad, torch.zeros_like(samples))


def _distributed_sigreg_worker(rank, init_file):
    torch.set_num_threads(1)
    torch.manual_seed(64)
    all_samples = torch.randn(3, 4, 8)
    regularizer = SIGReg(num_projections=8).eval()
    expected = regularizer(all_samples[:, 2:])
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank,
        world_size=2, timeout=timedelta(seconds=30),
    )
    try:
        # Rank zero has no future labels, but must still join every collective.
        masks = ([False, False], [True, True])
        samples = all_samples[:, rank * 2:(rank + 1) * 2].clone()
        if rank == 0:
            samples.fill_(float("nan"))
        samples.requires_grad_(True)
        value = regularizer(samples, sample_mask=torch.tensor(masks[rank]))
        torch.testing.assert_close(value, expected)
        value.backward()
        assert torch.isfinite(samples.grad).all()
        if rank == 0:
            torch.testing.assert_close(samples.grad, torch.zeros_like(samples))

        for globally_valid in (0, 1):
            samples = all_samples[:, rank * 2:(rank + 1) * 2].clone().requires_grad_(True)
            mask = torch.tensor([rank == 1 and globally_valid == 1, False])
            value = regularizer(samples, sample_mask=mask)
            assert torch.isfinite(value) and value.item() == 0
            value.backward()
            torch.testing.assert_close(samples.grad, torch.zeros_like(samples))

        # DDP averages local gradients, so a masked loss must scale by the
        # world size before reduction to represent the concatenated-batch mean.
        local_sum = torch.tensor(float(rank + 1), requires_grad=True)
        value, global_count = ddp_mean_from_sum(local_sum, rank + 1)
        assert global_count.item() == 3
        value.backward()
        torch.testing.assert_close(local_sum.grad, torch.tensor(2.0 / 3.0))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_sigreg_ddp_handles_rank_without_valid_future_labels(tmp_path):
    mp.spawn(_distributed_sigreg_worker, args=(str(tmp_path / "sigreg_init"),), nprocs=2, join=True)


@pytest.mark.parametrize("stage", ["p0", "p1"])
def test_terminal_only_batch_trains_valid_action_without_future_targets(stage):
    is_p1 = stage == "p1"
    config = small_config(P1Config if is_p1 else P0Config)
    dataset = SyntheticP1Dataset(config, length=2) if is_p1 else SyntheticP0Dataset(config, length=2)
    samples = [dataset[i] for i in range(2)]
    for sample in samples:
        sample["action_mask"] = torch.tensor([True, False, False, False])
        sample["future_masks"] = {h: torch.tensor(False) for h in config.jepa_horizons}
        sample["actions"][1:] = float("nan")
        for target in sample["future_features"].values():
            target.fill_(float("nan"))
    collator, batch_type = (collate_p1, P1Batch) if is_p1 else (collate_p0, P0Batch)
    batch = batch_type(**collator(samples))
    model = P1Model(config, qwen_backend="tiny") if is_p1 else P0Model(config, qwen_backend="tiny")
    model.train()
    result = model(batch)
    assert torch.isfinite(result["loss"])
    assert result["loss/jepa"].item() == 0
    if is_p1:
        assert result["loss/video_residual"].item() == 0
    result["loss"].backward()
    assert torch.isfinite(model.action_dit.out.weight.grad).all()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(batch)["loss"])


def test_collator_defaults_legacy_samples_to_valid_and_contract_rejects_bad_masks():
    config = small_config()
    dataset = SyntheticP0Dataset(config, length=2)
    masked, legacy = dataset[0], dataset[1]
    masked["action_mask"] = torch.tensor([True, False, False, False])
    masked["future_masks"] = {h: torch.tensor(False) for h in config.jepa_horizons}
    batch = P0Batch(**collate_p0([masked, legacy]))
    assert batch.action_mask[1].all()
    assert all(mask.tolist() == [False, True] for mask in batch.future_masks.values())
    batch.validate(config.jepa_horizons, config.vjepa_dim, config.action_horizon)
    batch.action_mask = batch.action_mask.float()
    with pytest.raises(ValueError, match="action_mask"):
        batch.validate(config.jepa_horizons, config.vjepa_dim, config.action_horizon)
