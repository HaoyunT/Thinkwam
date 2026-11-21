#!/usr/bin/env python3
"""Encode each required future frame once, storing only pooled training targets."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_p0 import load_config
from thinkwam_p0.data.collate import _preprocess_images
from thinkwam_p0.data.future_cache import (
    atomic_json, cache_spec, episode_windows, sha256_file, validate_episode,
)
from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21
from thinkwam_p1.data.robocasa24 import RoboCasa24Dataset


def encode_episode(video, output, encoder, *, spec, windows, batch_size, device):
    from decord import VideoReader, cpu

    reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
    if windows != len(reader):
        raise ValueError(f'video/action length mismatch: {video}')
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        h: np.lib.format.open_memmap(output / f'future_t{h}.npy', mode='w+',
                                    dtype=np.float16, shape=(windows, count, spec['feature_dim']))
        for h, count in zip(spec['horizons'], spec['token_counts'])
    }
    for array in arrays.values():
        array[:] = 0
    needed = sorted({t + h for t in range(windows) for h in spec['horizons'] if t + h < windows})
    counts_written = {h: 0 for h in spec['horizons']}
    for start in range(0, len(needed), batch_size):
        frames = needed[start:start + batch_size]
        views = RoboCasa24Dataset._untile(reader.get_batch(frames).asnumpy())
        images = _preprocess_images(list(views)).to(device)
        with torch.inference_mode():
            encoded = encoder(images).float()
            for horizon, count in zip(spec['horizons'], spec['token_counts']):
                valid = [(i, frame - horizon) for i, frame in enumerate(frames)
                         if 0 <= frame - horizon < windows]
                if not valid:
                    continue
                pooled = F.adaptive_avg_pool1d(encoded.transpose(1, 2), count).transpose(1, 2)
                values = pooled[[i for i, _t in valid]].cpu().numpy().astype(np.float16)
                if not np.isfinite(values).all():
                    raise ValueError(f'nonfinite future features: {video}')
                arrays[horizon][[t for _i, t in valid]] = values
                counts_written[horizon] += len(valid)
    expected_valid = {h: max(0, windows - h) for h in spec['horizons']}
    assert counts_written == expected_valid
    for array in arrays.values():
        array.flush()
    arrays.clear()
    record = {
        'windows': windows, 'source_video_sha256': sha256_file(video),
        'valid_future_windows': {str(h): count for h, count in expected_valid.items()},
        'file_bytes': {str(h): (output / f'future_t{h}.npy').stat().st_size for h in spec['horizons']},
        'file_sha256': {str(h): sha256_file(output / f'future_t{h}.npy') for h in spec['horizons']},
    }
    atomic_json(output / 'completed.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/paper/robocasa24_p0.yaml')
    parser.add_argument('--output', required=True, help='Destination for the optional future-feature cache')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reserve-gib', type=float, default=100)
    args = parser.parse_args()
    if args.batch_size < 1 or args.reserve_gib < 0:
        parser.error('batch size must be positive and disk reserve nonnegative')
    config = load_config(args.config)
    root, output = Path(config.dataset_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # One builder owns the cache. Completed episodes survive interruption.
    with (output / '.builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = cache_spec(root, config.vjepa_checkpoint, config.jepa_horizons,
                          config.jepa_token_counts, config.action_horizon, config.vjepa_dim)
        identity = output / 'build_identity.json'
        if identity.exists() and json.loads(identity.read_text()) != spec:
            raise ValueError('cache directory already belongs to different data or encoder settings')
        atomic_json(identity, spec)
        actions = json.loads((root / 'action.json').read_text())
        windows = {name: episode_windows(values, config.jepa_horizons, config.action_horizon)
                   for name, values in actions.items()}
        windows = {name: count for name, count in windows.items() if count > 0}
        if not windows:
            raise ValueError('dataset contains no valid future-target windows')
        records = {}
        for episode in windows:
            done = output / episode / 'completed.json'
            if done.exists():
                record = json.loads(done.read_text())
                if record['windows'] != windows[episode]:
                    raise ValueError(f'{episode}: completed cache window count changed')
                validate_episode(output, episode, record, spec, dataset_root=root)
                records[episode] = record
        remaining_bytes = sum(windows[e] for e in windows if e not in records) * sum(config.jepa_token_counts) * config.vjepa_dim * 2
        if shutil.disk_usage(output).free < remaining_bytes + args.reserve_gib * 1024**3:
            raise RuntimeError('insufficient disk space for future cache plus requested reserve')
        if len(records) != len(windows):
            atomic_json(output / 'progress.json', {
                'at': time.time(), 'pid': os.getpid(), 'state': 'loading_encoder',
                'completed_episodes': len(records), 'total_episodes': len(windows),
                'completed_windows': sum(r['windows'] for r in records.values()),
                'total_windows': sum(windows.values()),
            })
            encoder = FrozenVJEPA21(config.vjepa_checkpoint, config.vjepa_image_size,
                                    config.vjepa_patch_size).to(args.device).eval()
            for episode in sorted(windows):
                if episode in records:
                    continue
                temporary = output / f'.{episode}.building'
                if temporary.exists():
                    shutil.rmtree(temporary)
                record = encode_episode(root / 'videos' / f'{episode}.mp4', temporary, encoder,
                                        spec=spec, windows=windows[episode],
                                        batch_size=args.batch_size, device=args.device)
                temporary.rename(output / episode)
                records[episode] = record
                progress = {
                    'at': time.time(), 'pid': os.getpid(), 'state': 'building',
                    'completed_episodes': len(records), 'total_episodes': len(windows),
                    'completed_windows': sum(r['windows'] for r in records.values()),
                    'total_windows': sum(windows.values()), 'last_episode': episode,
                }
                atomic_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        manifest = {'complete': True, 'spec': spec, 'episodes': records}
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'progress.json', {
            'at': time.time(), 'state': 'complete', 'completed_episodes': len(records),
            'total_episodes': len(windows), 'completed_windows': sum(windows.values()),
            'total_windows': sum(windows.values()),
        })
        print('Future target cache complete', output, flush=True)


if __name__ == '__main__':
    main()
