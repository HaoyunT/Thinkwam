import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from scripts import build_robocasa24_future_cache as builder
from scripts.train_p0 import load_config as load_p0_config
from scripts.train_p1 import load_config as load_p1_config
from thinkwam_p0.config import P0Config
from thinkwam_p0.data.collate import _preprocess_images, collate_p0
from thinkwam_p0.data.future_cache import FutureTargetCache
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.types import P0Batch
from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p1.data.robocasa24 import RoboCasa24Dataset, VIEW_NAMES
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.types import P1Batch


def test_paper_recipes_use_online_observations_and_uncached_future_targets():
    root = Path(__file__).resolve().parents[1]
    for stage, load in [('p0', load_p0_config), ('p1', load_p1_config)]:
        config = load(str(root / f'configs/paper/robocasa24_{stage}.yaml'))
        assert config.dataset == 'robocasa24'
        assert config.online_vjepa is True
        assert config.future_target_cache == ''
        assert config.jepa_horizons == (7, 15, 31)
        assert config.jepa_token_counts == (288, 144, 72)


class SmallEncoder(torch.nn.Module):
    def __init__(self, dim=16, inference=False):
        super().__init__()
        self.dim = dim
        self.inference = inference
        self.calls = []

    def forward(self, images):
        self.calls.append(len(images))
        with torch.inference_mode(self.inference):
            base = images.mean(dim=(1, 2, 3, 4))[:, None, None]
            positions = torch.arange(12, device=images.device)[None, :, None] / 13
            channels = torch.arange(self.dim, device=images.device)[None, None, :] / (self.dim + 1)
            return base + positions + channels


def small_config(stage):
    cls = P0Config if stage == 'p0' else P1Config
    values = dict(
        dataset='robocasa24', online_vjepa=True, vjepa_dim=16, vjepa_views=3,
        visual_token_count=12, qwen_dim=32, qwen_visual_tokens=12,
        qwen_action_tokens=4, memory_dim=32, memory_slots=4,
        memory_observation_tokens=8, memory_action_history=2,
        memory_layers=1, memory_heads=4, jepa_context_dim=32,
        jepa_layers=1, jepa_heads=4, jepa_token_counts=(4, 2, 1),
        jepa_observation_tokens=12, action_context_dim=32,
        conditioning_mlp_hidden=64, action_dim=7, proprio_dim=1,
        action_horizon=16, action_dit_hidden=32, action_dit_layers=1,
        action_dit_heads=4, action_dit_inner_dim=32, flow_steps=1,
        sigreg_loss_weight=0,
    )
    if stage == 'p1':
        values.update(
            affordance_decoder_dim=32,
            affordance_decoder_layers=1, affordance_decoder_heads=4,
            video_model_dim=32, video_model_layers=1, video_model_heads=4,
            video_context_tokens=8, video_condition_steps=1,
            video_condition_train_steps=1, vjepa_adapter_dim=8,
            p1_stage='joint', affordance_loss_ramp_steps=0,
        )
    return cls(**values)


@pytest.mark.parametrize('stage', ['p0', 'p1'])
def test_cached_targets_match_losses_preserve_gradients_and_skip_future_encoding(stage):
    torch.manual_seed(7)
    config = small_config(stage)
    cls = P0Model if stage == 'p0' else P1Model
    encoder = SmallEncoder(inference=stage == 'p0')
    model = cls(config, qwen_backend='tiny', vjepa_encoder=encoder).eval()
    rng = np.random.default_rng(7)
    sample = dict(
        obs_images=rng.integers(0, 255, (3, 8, 8, 3), dtype=np.uint8),
        future_images={h: rng.integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
                       for h in config.jepa_horizons},
        actions=torch.randn(16, 7), action_history=torch.randn(2, 7),
        proprio=torch.zeros(1), affordance_targets=torch.ones(3, 8, 8),
    )
    collate = collate_p0 if stage == 'p0' else collate_p1
    batch_cls = P0Batch if stage == 'p0' else P1Batch
    raw = collate([sample])
    state = torch.get_rng_state()
    online = model(batch_cls(**raw))
    online['loss'].backward()
    assert model.memory.obs_proj.weight.grad is not None
    model.zero_grad(set_to_none=True)
    targets = {}
    for h, count in zip(config.jepa_horizons, config.jepa_token_counts):
        encoded = SmallEncoder()(raw['future_images'][h])
        pooled = F.adaptive_avg_pool1d(encoded.transpose(1, 2), count).transpose(1, 2)
        targets[h] = pooled.half().float().requires_grad_()
    cached_raw = dict(raw, future_images=None, future_features=targets)
    config.future_target_cache = 'unit-test-cache'
    encoder.calls.clear()
    torch.set_rng_state(state)
    cached = model(batch_cls(**cached_raw))
    for key in ['loss', 'loss/jepa'] + (['loss/video_residual'] if stage == 'p1' else []):
        torch.testing.assert_close(cached[key], online[key], atol=5e-4, rtol=5e-4)
    assert encoder.calls == [1]
    cached['loss'].backward()
    assert all(value.grad is None for value in targets.values())
    assert model.memory.obs_proj.weight.grad is not None
    if stage == 'p1':
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vjepa_adapter.parameters())
        assert any(p.grad is not None for p in model.affordance.parameters())
        assert model.video_residual.head['t7'][-1].weight.grad is not None
        assert cached['refined_actions'].shape == (1, 16, 7)


def test_collator_can_read_only_the_current_frame():
    sample = dict(obs_images=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                  future_images=None,
                  future_features={h: torch.ones(n, 16) for h, n in zip((7, 15, 31), (4, 2, 1))},
                  actions=torch.zeros(16, 7), action_history=torch.zeros(2, 7))
    raw = collate_p0([sample])
    P0Batch(**raw).validate((7, 15, 31), 16, 16, online_vjepa=True,
                           vjepa_views=3, future_target_cache=True)
    assert raw['future_images'] is None
    with pytest.raises(ValueError, match='not cached features'):
        P0Batch(**raw).validate((7, 15, 31), 16, 16, online_vjepa=True, vjepa_views=3)


@pytest.fixture
def cache_dataset(tmp_path, monkeypatch):
    import decord

    root = tmp_path / 'data'
    root.mkdir()
    for folder in ['videos', 'metas', 'affordance']:
        (root / folder).mkdir()
    (root / 'videos/demo_0.mp4').write_bytes(b'fixture-video-identity')
    (root / 'metas/demo_0.txt').write_text('open the drawer')
    (root / 'dataset_info.json').write_text(json.dumps({
        'views': list(VIEW_NAMES), 'tile': [[VIEW_NAMES[0], VIEW_NAMES[1]], [VIEW_NAMES[2], None]],
    }))
    (root / 'norm_stats.json').write_text(json.dumps({'action_low': [-1] * 7, 'action_high': [1] * 7}))
    (root / 'data_mapping.json').write_text('{}')
    (root / 'action.json').write_text(json.dumps({'demo_0': [[0] * 7 for _ in range(34)]}))
    np.savez(root / 'affordance/demo_0.npz', **{view: np.ones((34, 4, 4)) for view in VIEW_NAMES})
    checkpoint = tmp_path / 'encoder.pt'
    checkpoint.write_bytes(b'frozen-encoder-identity')
    frames = np.arange(34, dtype=np.uint8)[:, None, None, None] * np.ones((34, 8, 8, 3), dtype=np.uint8)

    class Reader:
        requested = []

        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return len(frames)

        def get_batch(self, indices):
            self.requested.append(list(indices))
            return SimpleNamespace(asnumpy=lambda: frames[indices])

        def close(self):
            pass

    monkeypatch.setattr(decord, 'VideoReader', Reader)
    calls = []

    def encoder_factory(*_args):
        calls.append(True)
        return SmallEncoder(dim=1024)

    monkeypatch.setattr(builder, 'FrozenVJEPA21', encoder_factory)
    config = dict(dataset='robocasa24', dataset_root=str(root), online_vjepa=True,
                  vjepa_checkpoint=str(checkpoint), future_target_cache=str(tmp_path / 'cache'),
                  action_horizon=16, jepa_horizons=[7, 15, 31], jepa_token_counts=[4, 2, 1])
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, 'argv', ['builder', '--config', str(config_path),
                                   '--output', str(tmp_path / 'cache'),
                                   '--device', 'cpu', '--reserve-gib', '0', '--batch-size', '2'])
    builder.main()
    return root, checkpoint, tmp_path / 'cache', Reader, calls


def open_cache(root, checkpoint, output):
    return FutureTargetCache(output, dataset_root=root, checkpoint=checkpoint,
                             horizons=(7, 15, 31), token_counts=(4, 2, 1), action_horizon=16)


def test_cache_resume_dataset_mapping_and_training_labels(cache_dataset):
    root, checkpoint, output, reader, calls = cache_dataset
    cache = open_cache(root, checkpoint, output)
    assert cache.validate_files(root) == {'episodes': 1, 'windows': 34, 'complete': True}
    builder.main()
    assert len(calls) == 1  # No encoder reload when all episode records are complete.
    dataset = RoboCasa24Dataset(str(root), require_affordance=True,
                               future_target_cache=str(output), future_token_counts=(4, 2, 1),
                               vjepa_checkpoint=str(checkpoint))
    reader.requested.clear()
    sample = dataset[1]
    assert reader.requested == [[1]]
    assert sample['future_images'] is None
    assert sample['affordance_targets'].shape == (3, 4, 4)
    for h, count in zip((7, 15, 31), (4, 2, 1)):
        assert sample['future_features'][h].shape == (count, 1024)
        views = np.full((3, 4, 4, 3), 1 + h, dtype=np.uint8)
        encoded = SmallEncoder(dim=1024)(_preprocess_images([views]))
        expected = F.adaptive_avg_pool1d(encoded.transpose(1, 2), count).transpose(1, 2)
        # FP32 reductions across batch sizes can land in adjacent FP16 bins.
        torch.testing.assert_close(sample['future_features'][h], expected[0].half().float(),
                                   atol=5e-4, rtol=1e-3)
    assert dataset.index == [('demo_0', frame) for frame in range(34)]
    terminal = dataset[-1]
    assert terminal['action_mask'].sum() == 1
    assert not any(terminal['future_masks'].values())
    assert all(not features.count_nonzero() for features in terminal['future_features'].values())


def test_cache_rejects_the_legacy_cropped_index_contract(cache_dataset):
    root, checkpoint, output, _reader, _calls = cache_dataset
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['spec']['schema'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='rebuild legacy cache'):
        open_cache(root, checkpoint, output)


def test_cache_tracks_partial_horizons_and_never_fabricates_terminal_features(cache_dataset):
    root, checkpoint, output, _reader, _calls = cache_dataset
    cache = open_cache(root, checkpoint, output)
    assert cache.episodes['demo_0']['valid_future_windows'] == {'7': 27, '15': 19, '31': 3}
    assert cache.get('demo_0', 26)[7].count_nonzero()
    assert not cache.get('demo_0', 27)[7].count_nonzero()
    with pytest.raises(IndexError, match='outside full-frame indexing'):
        cache.get('demo_0', 34)


def test_cache_rejects_stale_encoder_and_truncated_payload(cache_dataset):
    root, checkpoint, output, _reader, _calls = cache_dataset
    checkpoint.write_bytes(b'changed-encoder')
    with pytest.raises(ValueError, match='contract mismatch'):
        open_cache(root, checkpoint, output)
    checkpoint.write_bytes(b'frozen-encoder-identity')
    (output / 'demo_0/future_t7.npy').write_bytes(b'truncated')
    with pytest.raises(ValueError, match='truncated'):
        open_cache(root, checkpoint, output).validate_files(root)


def test_cache_rejects_incomplete_manifest(cache_dataset):
    root, checkpoint, output, _reader, _calls = cache_dataset
    (output / 'manifest.json').unlink()
    with pytest.raises(FileNotFoundError, match='not complete'):
        open_cache(root, checkpoint, output)


def test_waiter_excludes_busy_and_low_memory_gpus(monkeypatch):
    from scripts import wait_robocasa24_future_cache as waiter

    def query(command, **_kwargs):
        if '--query-compute-apps=gpu_uuid' in command:
            text = 'GPU-0\n'
        else:
            text = '0, GPU-0, 0, 80000\n1, GPU-1, 70, 80000\n2, GPU-2, 1, 20000\n3, GPU-3, 0, 1000\n'
        return SimpleNamespace(stdout=text)

    monkeypatch.setattr(waiter.subprocess, 'run', query)
    assert waiter.idle_gpus() == ['GPU-2']
