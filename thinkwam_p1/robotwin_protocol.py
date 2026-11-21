"""Clean-only RoboTwin inputs and evidence for decoded RGB cache provenance.

The pinned source files identify the audited 50-task x 50-demo Clean export.
Each RGB view is compared with the decoded Clean source before its first use.
Only a complete per-view comparison (or actual decoding during creation) emits
a receipt. A directory name or an unverified manifest is never sufficient.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from thinkwam_p1.robotwin_clean_source import CLEAN_SOURCE

CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')
METHOD = 'all_view_pixels_equal_to_pinned_clean_ffmpeg_decode_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


CLEAN_SOURCE_SHA256 = digest(CLEAN_SOURCE)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def snapshot(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def root_path(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f'RoboTwin source/cache roots must not redirect through symlinks: {path}')
    return path.resolve(strict=True)


def confined_file(root, path, *, must_exist=True):
    """Reject traversal and symlink escapes, including a symlinked ancestor."""
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute():
        if '..' in path.parts:
            raise ValueError(f'RoboTwin path escapes its root: {path}')
        path = root / path
    if not path.resolve(strict=must_exist).is_relative_to(root):
        raise ValueError(f'RoboTwin path escapes its root: {path}')
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f'RoboTwin input must not be a symlink: {path}')
    if must_exist and not path.is_file():
        raise ValueError(f'RoboTwin input is not a file: {path}')
    return path.resolve()


def validate_clean_source(root, *, include_videos=True):
    """Validate bytes, not a directory label; relocated exact copies are valid."""
    root = root_path(root)
    expected = CLEAN_SOURCE['files']
    for directory, suffix in [('data', '*.parquet'), ('meta/episodes', '*.parquet'), ('videos', '*.mp4')]:
        actual = {p.relative_to(root).as_posix() for p in (root / directory).rglob(suffix)}
        wanted = {p for p in expected if p.startswith(directory + '/')}
        if actual != wanted:
            raise ValueError(f'RoboTwin Clean source inventory differs in {directory}')
    checked = {}
    for relative, reference in expected.items():
        if not include_videos and relative.startswith('videos/'):
            continue
        path = confined_file(root, relative)
        before = snapshot(path)
        if before['size'] != reference['size'] or sha256_file(path) != reference['sha256']:
            raise ValueError(f'Not the verified RoboTwin Clean source: {relative}')
        if snapshot(path) != before:
            raise ValueError(f'RoboTwin source changed during validation: {path}')
        checked[relative] = before
    return root, checked


def validate_robotwin_training_config(config):
    """Entry-point guard; the RGB dataset performs the full cache check."""
    if config.dataset != 'robotwin2.0':
        return
    if not config.online_vjepa or config.feature_cache or getattr(config, 'future_target_cache', ''):
        raise ValueError('RoboTwin paper training requires verified Clean RGB, not latent/other feature caches')
    root, _ = validate_clean_source(config.dataset_root, include_videos=False)
    stats = confined_file(root, config.robotwin_stats_path)
    if stats != root / 'dataset_stats.json':
        raise ValueError('RoboTwin must use its pinned Clean normalization statistics')


def read_clean_manifest(root, cache, manifest_name, image_size=384, expected_views=3):
    root, cache = root_path(root), root_path(cache)
    if image_size != 384 or expected_views != 3:
        raise ValueError('RoboTwin Clean paper RGB requires 384px and the three canonical cameras')
    path = confined_file(cache, manifest_name)
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != 1 or root_path(value['source_root']) != root
            or value.get('image_size') != image_size or tuple(value.get('cameras', ())) != CAMERAS):
        raise ValueError('RGB cache manifest does not match the verified Clean source/cameras')
    episodes = value.get('episodes', [])
    if (len(episodes) != CLEAN_SOURCE['episodes']
            or any(type(e.get('episode_index')) is not int for e in episodes)
            or {e['episode_index'] for e in episodes} != set(range(CLEAN_SOURCE['episodes']))):
        raise ValueError('RoboTwin Clean manifest must contain exactly 2500 unique demos (50 tasks x 50)')
    from thinkwam_p1.data.robotwin_task_sampling import ROBOTWIN_TASKS
    from thinkwam_p0.data.robotwin import RobotWinIndex
    source = {e.episode_index: e for e in RobotWinIndex(root, variant='clean').episodes}
    if set(source) != set(range(CLEAN_SOURCE['episodes'])):
        raise ValueError('Clean source episode inventory is incomplete')
    counts = Counter()
    cache_files = []
    for episode in episodes:
        index = episode['episode_index']
        task = ROBOTWIN_TASKS[index // CLEAN_SOURCE['demos_per_task']]
        counts[task] += 1
        if episode.get('task_name', task) != task or episode.get('task', task) != task:
            raise ValueError('RoboTwin task identity differs from the pinned Clean export')
        original = source[index]
        if type(episode.get('length')) is not int or episode['length'] != original.length:
            raise ValueError(f'Clean source episode length mismatch: {index}')
        videos = episode.get('videos', [])
        if len(videos) != len(CAMERAS):
            raise ValueError(f'Clean episode must have exactly three videos: {index}')
        for camera, (video, start) in zip(CAMERAS, videos):
            video = confined_file(root, video)
            if video != original.video_path(camera).resolve() or start != original.video_start(camera):
                raise ValueError(f'RGB manifest does not match Clean episode video/time: {index}')
        directory = f'episode_{index:06d}'
        cache_files.extend([f'{directory}/labels.npz', *(f'{directory}/view_{v}.npy' for v in range(3))])
    for relative in cache_files:
        confined_file(cache, relative, must_exist=relative.endswith('labels.npz'))
    if len(counts) != CLEAN_SOURCE['tasks'] or set(counts.values()) != {CLEAN_SOURCE['demos_per_task']}:
        raise ValueError('RoboTwin requires all 50 Clean tasks with 50 demonstrations each')
    if sum(e['length'] for e in episodes) != CLEAN_SOURCE['frames']:
        raise ValueError('RoboTwin Clean frame inventory differs from the pinned export')
    return root, cache, path, value, source, cache_files


def verify_labels(root, cache, manifest, source):
    """Compare every action/state value and instruction with pinned Parquet."""
    import numpy as np
    import pyarrow.parquet as pq
    from thinkwam_p0.data.robotwin import RobotWinIndex
    tasks = RobotWinIndex._read_tasks(root)
    groups = defaultdict(list)
    records = {e['episode_index']: e for e in manifest['episodes']}
    for e in source.values():
        groups[e.data_path].append(e)
    for parquet, episodes in groups.items():
        data = pq.read_table(parquet, columns=['observation.state', 'action', 'task_index',
                                              'frame_index', 'episode_index']).to_pydict()
        ids, frames, task_ids = (np.asarray(data[k]) for k in ('episode_index', 'frame_index', 'task_index'))
        actions, states = (np.asarray(data[k], np.float32) for k in ('action', 'observation.state'))
        for episode in episodes:
            positions = np.flatnonzero(ids == episode.episode_index)
            if len(positions) != episode.length or not np.array_equal(frames[positions], np.arange(episode.length)):
                raise ValueError(f'Invalid Clean frame order: {episode.episode_index}')
            path = confined_file(cache, f'episode_{episode.episode_index:06d}/labels.npz')
            with np.load(path, allow_pickle=False) as labels:
                if (set(labels.files) != {'action', 'state'}
                        or not np.array_equal(labels['action'], actions[positions])
                        or not np.array_equal(labels['state'], states[positions])):
                    raise ValueError(f'RoboTwin cached labels differ from Clean source: {path}')
            if records[episode.episode_index].get('language') != tasks[int(task_ids[positions[0]])]:
                raise ValueError(f'Cached instruction differs from Clean source: {episode.episode_index}')


def validate_clean_rgb_cache(root, *, frame_cache, manifest_name='manifest_3view.json',
                             image_size=384, expected_views=3, stats_path=''):
    manifest_before = sha256_file(confined_file(root_path(frame_cache), manifest_name))
    root, cache, manifest_path, manifest, source, files = read_clean_manifest(
        root, frame_cache, manifest_name, image_size, expected_views)
    stats = confined_file(root, stats_path or 'dataset_stats.json')
    if stats != root / 'dataset_stats.json':
        raise ValueError('RoboTwin normalization must use the pinned Clean statistics')
    _, source_stats = validate_clean_source(root)
    label_stats = {name: snapshot(confined_file(cache, name)) for name in files if name.endswith('labels.npz')}
    # Labels are small (~62 MB uncompressed); compare all of them on startup.
    # RGB pixels are verified lazily, in ensure_frames, before being consumed.
    verify_labels(root, cache, manifest, source)
    if any(snapshot(confined_file(root, name)) != stat for name, stat in source_stats.items()):
        raise ValueError('RoboTwin Clean source changed during label verification')
    if any(snapshot(confined_file(cache, name)) != stat for name, stat in label_stats.items()):
        raise ValueError('RoboTwin cached labels changed during source verification')
    if sha256_file(manifest_path) != manifest_before:
        raise ValueError('RoboTwin manifest changed during source verification')
    return dict(source_root=str(root), source_files=source_stats, labels=label_stats,
                source_identity_sha256=CLEAN_SOURCE_SHA256, manifest_sha256=manifest_before)
