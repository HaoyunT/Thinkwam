#!/usr/bin/env python3
"""Index clean RGB data without building/downsampling V-JEPA features."""
import argparse
import collections
import json
import os
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinkwam_p0.data.robotwin import RobotWinIndex, ROBOTWIN_CAMERAS
from thinkwam_p1.robotwin_protocol import validate_clean_source, CAMERAS
from thinkwam_p1.data.robotwin_task_sampling import ROBOTWIN_TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest-name', default='manifest_3view.json')
    p.add_argument('--camera-keys', default='cam_high,cam_left_wrist,cam_right_wrist')
    args = p.parse_args()
    root, _ = validate_clean_source(args.root)
    output = Path(args.output).resolve()
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError('manifest-name must be a filename inside the output directory')
    manifest_path = output / args.manifest_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('RGB preparation requires an empty output; existing cached data is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    index = RobotWinIndex(root, variant='clean')
    aliases = {camera.rsplit('.', 1)[-1]: camera for camera in ROBOTWIN_CAMERAS}
    cameras = tuple(aliases.get(item.strip(), item.strip())
                    for item in args.camera_keys.split(',') if item.strip())
    if cameras != CAMERAS:
        raise ValueError(f'Clean paper RGB preparation requires exactly the canonical camera order {CAMERAS}')
    groups = collections.defaultdict(list)
    for e in index.episodes:
        groups[e.data_path].append(e)
    records = []
    for path, episodes in groups.items():
        data = pq.read_table(path, columns=['observation.state', 'action', 'task_index',
                                           'frame_index', 'episode_index']).to_pydict()
        action = np.asarray(data['action'], dtype=np.float32)
        state = np.asarray(data['observation.state'], dtype=np.float32)
        episode_ids = np.asarray(data['episode_index'])
        frames = np.asarray(data['frame_index'])
        tasks = np.asarray(data['task_index'])
        for e in episodes:
            positions = np.flatnonzero(episode_ids == e.episode_index)
            assert len(positions) == e.length
            assert np.array_equal(frames[positions], np.arange(e.length))
            assert np.isfinite(action[positions]).all() and np.isfinite(state[positions]).all()
            assert action[positions].shape == state[positions].shape == (e.length, 14)
            language = index._tasks_cache[e.root][int(tasks[positions[0]])]
            assert language.strip()
            directory = output / f'episode_{e.episode_index:06d}'
            directory.mkdir(exist_ok=True)
            np.savez(directory / 'labels.npz', action=action[positions], state=state[positions])
            videos = [[str(e.video_path(c)), e.video_start(c)] for c in cameras]
            assert all(Path(path).is_file() for path, start in videos)
            records.append(dict(episode_index=e.episode_index, length=e.length,
                                task_name=ROBOTWIN_TASKS[e.episode_index // 50],
                                language=language, videos=videos))
        print(f'indexed {len(records)}/{len(index.episodes)} episodes', flush=True)
    records.sort(key=lambda e: e['episode_index'])
    manifest = dict(schema=1, source_root=str(root), image_size=384,
                    cameras=list(cameras), episodes=records)
    temp = manifest_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, ensure_ascii=False))
    os.replace(temp, manifest_path)
    print('ready', len(records), 'episodes', sum(e['length'] for e in records), 'frames', flush=True)
    print('Labels and source inventory are verified. RGB views are decoded/verified before first use; '
          'scripts/verify_robotwin_rgb.py optionally checks all views in advance.', flush=True)


if __name__ == '__main__':
    main()
