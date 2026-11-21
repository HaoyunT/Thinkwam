import json
import numpy as np
import pytest
import torch

from thinkwam_p1.data import robotwin_online as module


class Tokenizer:
    pad_token_id = 0
    def __call__(self, texts, **kwargs):
        shape = (len(texts), kwargs['max_length'])
        return {'input_ids': np.ones(shape, np.int64), 'attention_mask': np.ones(shape, bool)}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    records = []
    for i, length in enumerate([40, 35]):
        directory = tmp_path / f'episode_{i:06d}'
        directory.mkdir()
        action = np.repeat((i * 100 + np.arange(length))[:, None], 14, axis=1).astype(np.float32)
        np.savez(directory / 'labels.npz', action=action, state=action + 1000)
        for v in range(3):
            pixels = np.broadcast_to((i * 100 + np.arange(length) + v * 40)[:, None, None, None],
                                     (length, 4, 4, 3)).astype(np.uint8).copy()
            np.save(directory / f'view_{v}.npy', pixels)
        records.append(dict(episode_index=i, length=length, language=f'task {i}', videos=[]))
    (tmp_path / 'manifest_3view.json').write_text(json.dumps(dict(schema=1, source_root=str(tmp_path),
        image_size=4, cameras=['head','left','right'], episodes=records)))
    stats = {'action': {'default': {'global_mean': [0.0] * 14, 'global_std': [1.0] * 14}},
             'state': {'default': {'global_mean': [0.0] * 14, 'global_std': [1.0] * 14}}}
    (tmp_path / 'stats.json').write_text(json.dumps(stats))
    # This fixture tests temporal slicing on tiny synthetic 4px images. The
    # separate source-protocol tests exercise the real Clean provenance guard.
    monkeypatch.setattr(module, 'validate_clean_rgb_cache', lambda *a, **kw: {
        'source_files': {},
        'manifest_sha256': module.sha256_file(tmp_path / 'manifest_3view.json'),
        'labels': {f'episode_{i:06d}/labels.npz': module.snapshot(tmp_path / f'episode_{i:06d}/labels.npz')
                   for i in range(2)},
    })
    monkeypatch.setattr(module, 'ensure_frames', lambda root, episode, view, size, **kw:
                        root / f"episode_{episode['episode_index']:06d}" / f'view_{view}.npy')
    return module.OnlineRobotWinDataset(str(tmp_path), frame_cache=str(tmp_path), action_horizon=16,
        action_history=8, horizons=(7,15,31), tokenizer_path='unused', image_size=4,
        expected_views=3, manifest_name='manifest_3view.json', stats_path=str(tmp_path / 'stats.json'))


def test_temporal_alignment_and_episode_boundaries(dataset):
    assert len(dataset) == 13
    first = dataset[0]
    assert first['obs_images'].shape[0] == 3
    assert all(value.shape[0] == 3 for value in first['future_images'].values())
    assert not torch.count_nonzero(first['action_history'])
    assert first['future_images'][31][0,0,0,0] == 31
    end = dataset[8]
    np.testing.assert_array_equal(end['actions'][:,0], np.arange(8,24))
    np.testing.assert_array_equal(end['action_history'][:,0], np.arange(8))
    assert end['future_images'][31][0,0,0,0] == 39
    assert end['obs_images'][1,0,0,0] == 48
    following = dataset[9]
    assert following['actions'][0,0] == 100
    assert not torch.count_nonzero(following['action_history'])
    assert following['proprio'][0] == 1100
    assert dataset[-1]['future_images'][31][0,0,0,0] == 134
    with pytest.raises(IndexError):
        dataset[13]


def test_returned_arrays_cannot_mutate_cached_data(dataset):
    sample = dataset[0]
    sample['obs_images'][:] = 255
    sample['actions'][:] = -10
    sample['text_input_ids'][:] = 0
    again = dataset[0]
    assert again['obs_images'][0,0,0,0] == 0
    assert again['actions'][0,0] == 0
    assert again['text_input_ids'][0] == 1


def test_mismatched_manifest_fails(dataset, tmp_path):
    with pytest.raises(ValueError, match='manifest'):
        module.OnlineRobotWinDataset(str(tmp_path), frame_cache=str(tmp_path), action_horizon=16,
            action_history=8, horizons=(7,15,31), tokenizer_path='unused', image_size=8,
            expected_views=3, manifest_name='manifest_3view.json', stats_path=str(tmp_path / 'stats.json'))
