from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_p0 import load_config as load_p0_config
from scripts.train_p1 import load_config as load_p1_config
from thinkwam_p0.data.splits import episode_holdout_indices
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p1.data.robocasa24 import RoboCasa24Dataset, _normalize_action
from thinkwam_p1.models.p1 import P1Model


ROOT = Path(__file__).resolve().parents[1]


def test_episode_holdout_is_repeatable_and_never_splits_a_demonstration():
    index = [(f'demo_{episode}', frame) for episode in range(10) for frame in range(episode + 2)]
    train, validation = episode_holdout_indices(index, eval_fraction=0.2, seed=7)
    assert (train, validation) == episode_holdout_indices(index, eval_fraction=0.2, seed=7)
    assert sorted(train + validation) == list(range(len(index)))
    train_episodes = {index[i][0] for i in train}
    validation_episodes = {index[i][0] for i in validation}
    assert len(validation_episodes) == 2
    assert not train_episodes & validation_episodes
    with pytest.raises(ValueError, match='two nonempty episodes'):
        episode_holdout_indices([('demo_0', 0), ('demo_0', 1)], eval_fraction=0.1, seed=7)


def test_stage_configs_keep_the_same_data_and_warm_start_shapes():
    p0_config = load_p0_config(str(ROOT / 'configs/paper/robocasa24_p0.yaml'))
    p1_config = load_p1_config(str(ROOT / 'configs/paper/robocasa24_p1.yaml'))
    for name in (
        'dataset_root', 'seed', 'eval_fraction', 'vjepa_views', 'action_dim',
        'proprio_dim', 'action_horizon', 'memory_action_history',
        'jepa_horizons', 'jepa_token_counts', 'resolved_jepa_observation_tokens',
    ):
        assert getattr(p0_config, name) == getattr(p1_config, name), name
    assert p1_config.p0_checkpoint == str(
        Path(p0_config.output_dir) / 'checkpoints/checkpoint_step_0020000.pt'
    )
    assert (p0_config.max_steps, p1_config.max_steps) == (20000, 40000)
    assert (p0_config.batch_size, p1_config.batch_size) == (24, 20)
    assert (p1_config.freeze_qwen_steps, p1_config.freeze_vjepa_steps,
            p1_config.freeze_memory_steps) == (0, 0, 0)
    assert p0_config.output_dir != p1_config.output_dir
    assert p1_config.affordance_enabled and p1_config.require_affordance_targets
    with torch.device('meta'):
        p0 = P0Model(p0_config, qwen_backend='tiny')
        p1 = P1Model(p1_config, qwen_backend='tiny')
    p1_state = p1.state_dict()
    for name, tensor in p0.state_dict().items():
        assert name in p1_state, name
        assert tensor.shape == p1_state[name].shape, name


def test_video_tiles_follow_the_released_camera_order():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[:, :4, :4] = 11
    frames[:, :4, 4:] = 22
    frames[:, 4:, :4] = 33
    frames[:, 4:, 4:] = 255
    views = RoboCasa24Dataset._untile(frames)
    assert views.shape == (2, 3, 4, 4, 3)
    for view, expected in enumerate((11, 22, 33)):
        assert (views[:, view] == expected).all()


def test_action_normalization_uses_released_bounds_and_clips():
    low, high = np.zeros(7), np.full(7, 2.0)
    action = np.array([[0.0] * 7, [1.0] * 7, [2.0] * 7, [4.0] * 7])
    normalized = _normalize_action(action, low, high)
    np.testing.assert_array_equal(normalized, np.array([[-1] * 7, [0] * 7, [1] * 7, [1] * 7]))
    assert normalized.dtype == np.float32


def test_missing_or_empty_language_is_not_replaced_by_an_episode_id(tmp_path):
    dataset = object.__new__(RoboCasa24Dataset)
    dataset.root = tmp_path
    with pytest.raises(FileNotFoundError, match='instruction missing'):
        dataset._read_instruction('demo_0')
    (tmp_path / 'metas').mkdir()
    instruction = tmp_path / 'metas/demo_0.txt'
    instruction.write_text('  \n ')
    with pytest.raises(ValueError, match='instruction is empty'):
        dataset._read_instruction('demo_0')
    instruction.write_text('  open the drawer\n')
    assert dataset._read_instruction('demo_0') == 'open the drawer'




def _label_reader(tmp_path, arrays):
    from collections import OrderedDict
    from thinkwam_p1.data.robocasa24 import VIEW_NAMES
    (tmp_path / 'affordance').mkdir(exist_ok=True)
    np.savez(tmp_path / 'affordance/demo_0.npz', **arrays)
    dataset = object.__new__(RoboCasa24Dataset)
    dataset.root = tmp_path
    dataset.episode_lengths = {'demo_0': 3}
    dataset._affordance_cache = OrderedDict()
    return dataset


def test_affordance_reader_preserves_valid_zero_and_normalizes_positive_maps(tmp_path):
    from thinkwam_p1.data.robocasa24 import VIEW_NAMES
    arrays = {view: np.zeros((3, 2, 2), dtype=np.float16) for view in VIEW_NAMES}
    arrays[VIEW_NAMES[0]][1, 0, 0] = 2
    dataset = _label_reader(tmp_path, arrays)
    assert torch.count_nonzero(dataset._load_affordance('demo_0', 0)) == 0
    value = dataset._load_affordance('demo_0', 1)
    assert value[0, 0, 0] == 1 and torch.count_nonzero(value) == 1


@pytest.mark.parametrize('fault', ['nan', 'inf', 'negative', 'wrong_length', 'wrong_rank', 'different_views', 'missing_view'])
def test_affordance_reader_rejects_corrupt_labels_without_cleaning(tmp_path, fault):
    from thinkwam_p1.data.robocasa24 import VIEW_NAMES
    arrays = {view: np.zeros((3, 2, 2), dtype=np.float16) for view in VIEW_NAMES}
    key = VIEW_NAMES[0]
    if fault in {'nan', 'inf', 'negative'}:
        arrays[key][2, 1, 1] = {'nan': float('nan'), 'inf': float('inf'), 'negative': -1}[fault]
    elif fault == 'wrong_length':
        arrays[key] = np.zeros((2, 2, 2), dtype=np.float16)
    elif fault == 'wrong_rank':
        arrays[key] = np.zeros((3, 4), dtype=np.float16)
    elif fault == 'different_views':
        arrays[key] = np.zeros((3, 4, 4), dtype=np.float16)
    else:
        arrays.pop(key)
    dataset = _label_reader(tmp_path, arrays)
    with pytest.raises((ValueError, KeyError)):
        dataset._load_affordance('demo_0', 0)
