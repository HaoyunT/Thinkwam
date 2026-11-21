"""Versioned, episode-addressed frozen future targets for online observations."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def cache_spec(dataset_root, checkpoint, horizons, token_counts, action_horizon, feature_dim=1024):
    root = Path(dataset_root)
    if len(horizons) != len(token_counts) or any(int(n) <= 0 for n in token_counts):
        raise ValueError('future cache horizons and positive token counts must align')
    return {
        'schema': 2,
        'indexing': 'all_current_frames_with_horizon_masks:v2',
        'missing_future': 'zero_placeholder_excluded_by_future_masks',
        'target_kind': 'raw_adaptive_avg_pool_then_float16',
        'preprocessing': 'collate_p0._preprocess_images:384:bicubic:imagenet:v1',
        'views': ['agentview_left', 'agentview_right', 'eye_in_hand'],
        'dataset_metadata_sha256': {
            name: sha256_file(root / name)
            for name in ('dataset_info.json', 'action.json', 'norm_stats.json', 'data_mapping.json')
        },
        'encoder_sha256': sha256_file(checkpoint),
        'horizons': list(horizons),
        'token_counts': list(token_counts),
        'action_horizon': int(action_horizon),
        'feature_dim': int(feature_dim),
        'dtype': 'float16',
    }


def episode_windows(actions, horizons, action_horizon):
    """Every real observation is eligible, including terminal partial chunks."""
    return len(actions)


def validate_episode(root, episode, record, spec, *, dataset_root=None):
    folder = Path(root) / episode
    if record['windows'] <= 0:
        raise ValueError(f'{episode}: no valid cached windows')
    expected_valid = {str(h): max(0, record['windows'] - h) for h in spec['horizons']}
    if record.get('valid_future_windows') != expected_valid:
        raise ValueError(f'{episode}: cached horizon validity does not match full-frame indexing')
    for horizon, count in zip(spec['horizons'], spec['token_counts']):
        path = folder / f'future_t{horizon}.npy'
        shape = (record['windows'], count, spec['feature_dim'])
        if path.stat().st_size != record['file_bytes'][str(horizon)]:
            raise ValueError(f'truncated future target cache: {path}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != shape or array.dtype != np.float16:
            raise ValueError(f'incompatible future target array: {path}')
    if dataset_root is not None:
        video = Path(dataset_root) / 'videos' / f'{episode}.mp4'
        if sha256_file(video) != record['source_video_sha256']:
            raise ValueError(f'{episode}: cached targets belong to a different video')


class FutureTargetCache:
    def __init__(self, root, *, dataset_root, checkpoint, horizons, token_counts,
                 action_horizon, feature_dim=1024):
        self.root = Path(root)
        manifest_path = self.root / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Future cache is not complete: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('complete') is not True:
            raise ValueError('future target cache manifest is incomplete')
        if manifest.get('spec', {}).get('schema') != 2:
            raise ValueError('future cache contract mismatch: rebuild legacy cache for full-frame tail supervision (schema 2)')
        expected = cache_spec(dataset_root, checkpoint, horizons, token_counts,
                              action_horizon, feature_dim)
        if manifest['spec'] != expected:
            raise ValueError('future cache data, encoder, preprocessing, or target contract mismatch')
        self.spec = expected
        self.episodes = manifest['episodes']
        self._arrays = OrderedDict()

    def validate_files(self, dataset_root):
        actions = json.loads((Path(dataset_root) / 'action.json').read_text())
        expected = {
            episode: episode_windows(values, self.spec['horizons'], self.spec['action_horizon'])
            for episode, values in actions.items()
        }
        expected = {episode: count for episode, count in expected.items() if count > 0}
        if set(self.episodes) != set(expected):
            raise ValueError('future cache episode coverage differs from action.json')
        for episode, windows in expected.items():
            if self.episodes[episode]['windows'] != windows:
                raise ValueError(f'{episode}: cached window count mismatch')
            validate_episode(self.root, episode, self.episodes[episode], self.spec,
                             dataset_root=dataset_root)
        return {'episodes': len(expected), 'windows': sum(expected.values()), 'complete': True}

    def get(self, episode, frame):
        if episode not in self.episodes or not 0 <= frame < self.episodes[episode]['windows']:
            raise IndexError(f'{episode}: future cache frame {frame} is outside full-frame indexing')
        if episode not in self._arrays:
            self._arrays[episode] = {
                h: np.load(self.root / episode / f'future_t{h}.npy', mmap_mode='r', allow_pickle=False)
                for h in self.spec['horizons']
            }
            if len(self._arrays) > 8:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(episode)
        return {
            h: (
                torch.from_numpy(np.array(array[frame], copy=True)).float()
                if frame + h < self.episodes[episode]['windows']
                else torch.zeros(array.shape[1:], dtype=torch.float32)
            )
            for h, array in self._arrays[episode].items()
        }
