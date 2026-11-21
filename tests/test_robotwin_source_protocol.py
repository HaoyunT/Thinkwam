"""Small CPU fixtures for Clean identity, path confinement and lazy pixel proof."""
import copy
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from thinkwam_p0.data.robotwin import resolve_robotwin_roots
from thinkwam_p1 import robotwin_protocol as protocol
from thinkwam_p1.data import robotwin_online as online


@pytest.fixture
def clean(tmp_path, monkeypatch):
    root, cache = tmp_path / 'clean', tmp_path / 'cache'
    root.mkdir(); cache.mkdir()
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text('{"total_episodes": 2}')
    pq.write_table(pa.table({'task_index': [0], 'task': ['lift the bottle']}), root / 'meta/tasks.parquet')
    action = np.arange(56, dtype=np.float32).reshape(4, 14)
    state = action + 100
    pq.write_table(pa.table({'action': action.tolist(), 'observation.state': state.tolist(),
                            'episode_index': [0, 0, 1, 1], 'frame_index': [0, 1, 0, 1],
                            'task_index': [0, 0, 0, 0]}), root / 'data/chunk-000/file-000.parquet')
    metadata = {'episode_index': [0, 1], 'length': [2, 2], 'data/chunk_index': [0, 0],
                'data/file_index': [0, 0], 'dataset_from_index': [0, 2], 'dataset_to_index': [2, 4]}
    videos = []
    for camera in protocol.CAMERAS:
        video = root / f'videos/{camera}/chunk-000/file-000.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fixture: pinned clean video bytes ' + camera.encode())
        videos.append(video)
        metadata.update({f'videos/{camera}/chunk_index': [0, 0], f'videos/{camera}/file_index': [0, 0],
                         f'videos/{camera}/from_timestamp': [0.0, 0.04]})
    pq.write_table(pa.table(metadata), root / 'meta/episodes/chunk-000/file-000.parquet')
    (root / 'dataset_stats.json').write_text(json.dumps({
        key: {'default': {'global_mean': [0.] * 14, 'global_std': [1.] * 14}}
        for key in ['action', 'state']}))
    records = []
    for index in range(2):
        directory = cache / f'episode_{index:06d}'
        directory.mkdir()
        np.savez(directory / 'labels.npz', action=action[index*2:index*2+2], state=state[index*2:index*2+2])
        records.append(dict(episode_index=index, length=2, language='lift the bottle',
                            videos=[[str(video), index * 0.04] for video in videos]))
    manifest = dict(schema=1, source_root=str(root), image_size=384, cameras=list(protocol.CAMERAS), episodes=records)
    manifest_path = cache / 'manifest_3view.json'
    manifest_path.write_text(json.dumps(manifest))
    reference = dict(schema=1, identity='synthetic-clean-fixture', episodes=2, frames=4,
                     tasks=1, demos_per_task=2, files={
                         path.relative_to(root).as_posix(): dict(size=path.stat().st_size, sha256=protocol.sha256_file(path))
                         for path in root.rglob('*') if path.is_file()})
    monkeypatch.setattr(protocol, 'CLEAN_SOURCE', reference)
    monkeypatch.setattr(online, 'CLEAN_SOURCE', reference)
    monkeypatch.setattr(protocol, 'CLEAN_SOURCE_SHA256', protocol.digest(reference))
    monkeypatch.setattr(online, 'CLEAN_SOURCE_SHA256', protocol.digest(reference))
    # Keep decoding synthetic and CPU-only; production uses the real same-command
    # streaming decoder. Each test can corrupt any of its returned pixels.
    pixels = np.arange(2 * 384 * 384 * 3, dtype=np.uint8).reshape(2, 384, 384, 3)
    calls = []
    def decode(*args):
        calls.append(args)
        yield 0, pixels[:1]
        yield 1, pixels[1:]
    monkeypatch.setattr(online, '_decode_clean_frames', decode)
    monkeypatch.setattr(online, '_decoder_hash', lambda *args: 'd' * 64)
    monkeypatch.setattr(online.shutil, 'disk_usage', lambda *args: SimpleNamespace(free=10**15))
    # No executable is launched, but its stat is part of the production contract.
    import imageio_ffmpeg
    monkeypatch.setattr(imageio_ffmpeg, 'get_ffmpeg_exe', lambda: str(videos[0]))
    evidence = protocol.validate_clean_rgb_cache(root, frame_cache=cache)
    return SimpleNamespace(root=root, cache=cache, manifest=manifest, manifest_path=manifest_path,
                           records=records, reference=reference, videos=videos, evidence=evidence,
                           pixels=pixels, calls=calls, monkeypatch=monkeypatch)


def read_view(clean):
    return online.ensure_frames(clean.cache, clean.records[0], 0, 384,
                                source_root=clean.root, source_files=clean.evidence['source_files'])


def test_self_contained_random_cannot_satisfy_clean(tmp_path):
    random = tmp_path / 'random'
    (random / 'meta').mkdir(parents=True)
    (random / 'meta/info.json').write_text('{}')
    with pytest.raises(ValueError, match='source is random'):
        resolve_robotwin_roots(random, variant='clean')
    assert resolve_robotwin_roots(random, variant='random') == (random,)


def test_source_guard_checks_bytes_even_when_directory_is_named_clean(clean):
    clean.videos[0].write_bytes(b'random bytes')
    with pytest.raises(ValueError, match='Not the verified'):
        protocol.validate_clean_source(clean.root)


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'extra', 'wrong_task'])
def test_manifest_requires_exact_clean_demo_inventory(clean, kind):
    value = copy.deepcopy(clean.manifest)
    if kind == 'duplicate': value['episodes'][1] = value['episodes'][0]
    elif kind == 'missing': value['episodes'].pop()
    elif kind == 'extra': value['episodes'].append(value['episodes'][0])
    else: value['episodes'][0]['task_name'] = 'unverified_other_task'
    clean.manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='2500|task identity'):
        protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)


def test_manifest_video_cannot_escape_clean_root(clean, tmp_path):
    outside = tmp_path / 'random.mp4'; outside.write_bytes(b'random')
    value = copy.deepcopy(clean.manifest)
    value['episodes'][0]['videos'][0][0] = str(outside)
    clean.manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='escapes'):
        protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)


def test_cache_symlink_escape_is_rejected(clean, tmp_path):
    target = clean.cache / 'episode_000000/view_0.npy'
    outside = tmp_path / 'outside.npy'; np.save(outside, clean.pixels)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes|symlink'):
        protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)


def test_manifest_filename_cannot_escape_cache(clean):
    with pytest.raises(ValueError, match='escapes'):
        protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache,
                                          manifest_name='../cache/manifest_3view.json')


def test_same_shape_cached_labels_must_equal_clean_parquet(clean):
    path = clean.cache / 'episode_000000/labels.npz'
    with np.load(path) as data:
        action, state = data['action'].copy(), data['state'].copy()
    action[0, 0] += 0.25
    np.savez(path, action=action, state=state)
    with pytest.raises(ValueError, match='labels differ'):
        protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)


def test_instruction_and_video_offset_must_match_clean_source(clean):
    for field in ('language', 'offset'):
        value = copy.deepcopy(clean.manifest)
        if field == 'language': value['episodes'][0]['language'] = 'external correction'
        else: value['episodes'][0]['videos'][0][1] = 3.0
        clean.manifest_path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='instruction|video/time'):
            protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)


def test_existing_rgb_is_compared_before_receipt_and_reused_afterwards(clean):
    target = clean.cache / 'episode_000000/view_0.npy'
    np.save(target, clean.pixels)
    before = target.read_bytes()
    assert read_view(clean) == target
    assert len(clean.calls) == 1 and target.read_bytes() == before
    receipt = json.loads((target.parent / 'view_0.provenance.json').read_text())
    assert receipt['cache_sha256'] == protocol.sha256_file(target)
    assert receipt['source']['video_sha256'] == clean.reference['files'][clean.videos[0].relative_to(clean.root).as_posix()]['sha256']
    assert read_view(clean) == target and len(clean.calls) == 1


def test_changed_receipt_digest_field_forces_pixel_recomparison(clean):
    target = read_view(clean)
    receipt_path = target.parent / 'view_0.provenance.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['cache_sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt))
    assert read_view(clean) == target
    assert len(clean.calls) == 2
    repaired = json.loads(receipt_path.read_text())
    assert repaired['cache_sha256'] == protocol.sha256_file(target)
    assert repaired['receipt_sha256'] == protocol.digest({
        key: value for key, value in repaired.items() if key != 'receipt_sha256'})


def test_lru_reopens_mmap_after_real_atomic_replacement(clean):
    dataset = online.OnlineRobotWinDataset.__new__(online.OnlineRobotWinDataset)
    dataset.root, dataset.frame_cache = clean.root, clean.cache
    dataset.provenance, dataset.episodes = clean.evidence, clean.records
    dataset.image_size, dataset.cameras = 384, protocol.CAMERAS
    dataset.action_mean = dataset.state_mean = np.zeros(14, np.float32)
    dataset.action_std = dataset.state_std = np.ones(14, np.float32)
    dataset._loaded = OrderedDict()
    old_mapping = dataset._episode(0)[2][0]
    target = clean.cache / 'episode_000000/view_0.npy'
    old_inode = target.stat().st_ino
    damaged = np.load(target, mmap_mode='r+')
    damaged[0, 0, 0, 0] = 255
    damaged.flush()
    del damaged
    assert int(old_mapping[0, 0, 0, 0]) == 255
    replacement = target.with_name('replacement.npy')
    np.save(replacement, clean.pixels)
    replacement.replace(target)
    assert target.stat().st_ino != old_inode
    new_mapping = dataset._episode(0)[2][0]
    assert new_mapping is not old_mapping
    assert int(new_mapping[0, 0, 0, 0]) == int(clean.pixels[0, 0, 0, 0]) == 0
    # The previous mmap really still sees the replaced, damaged inode.
    assert int(old_mapping[0, 0, 0, 0]) == 255


def test_wrong_pixels_cannot_be_certified_by_shape_or_a_marker(clean):
    target = clean.cache / 'episode_000000/view_0.npy'
    np.save(target, np.zeros_like(clean.pixels))
    receipt = target.parent / 'view_0.provenance.json'
    receipt.write_text('{"clean": true, "verified": true}')
    before = target.read_bytes()
    with pytest.raises(ValueError, match='differs from its Clean source'):
        read_view(clean)
    assert target.read_bytes() == before
    assert json.loads(receipt.read_text()) == {'clean': True, 'verified': True}


def test_changed_cached_pixels_force_recomparison_and_reject(clean):
    target = read_view(clean)
    changed = np.load(target, mmap_mode='r+')
    changed[0, 0, 0, 0] ^= 1
    changed.flush(); del changed
    with pytest.raises(ValueError, match='differs from its Clean source'):
        read_view(clean)
    assert len(clean.calls) == 2


def test_source_change_after_preflight_cannot_reuse_receipt(clean):
    read_view(clean)
    clean.videos[0].write_bytes(b'changed source')
    with pytest.raises(ValueError, match='changed since preflight'):
        read_view(clean)


def test_new_rgb_is_generated_from_clean_and_then_has_a_receipt(clean):
    target = read_view(clean)
    np.testing.assert_array_equal(np.load(target), clean.pixels)
    assert (target.parent / 'view_0.provenance.json').is_file()
    assert len(clean.calls) == 1


def test_partial_decoder_output_cannot_issue_a_receipt(clean):
    clean.monkeypatch.setattr(online, '_decode_clean_frames', lambda *args: iter([(0, clean.pixels[:1])]))
    with pytest.raises(ValueError, match='every RGB frame'):
        read_view(clean)
    assert not (clean.cache / 'episode_000000/view_0.npy').exists()
    assert not (clean.cache / 'episode_000000/view_0.provenance.json').exists()


def test_source_preflight_is_read_only_and_does_not_claim_unchecked_rgb(clean):
    before = {p.relative_to(clean.cache) for p in clean.cache.rglob('*')}
    evidence = protocol.validate_clean_rgb_cache(clean.root, frame_cache=clean.cache)
    assert len(evidence['labels']) == 2
    assert before == {p.relative_to(clean.cache) for p in clean.cache.rglob('*')}
    assert not list(clean.cache.rglob('*.provenance.json'))
