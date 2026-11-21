"""RoboTwin raw RGB windows for the same online V-JEPA path as LIBERO.

Episode/view segments are decoded with ffmpeg and cached as RGB uint8 pixels;
V-JEPA features and the trainable observation adapter stay online. Missing or
changed verification receipts trigger another comparison with decoded source
pixels. Memory mapping reads the selected frames without loading the complete
RGB array. Atomic files and flock coordinate ranks.
"""
from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import os
import shutil
import selectors
import subprocess
import tempfile
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from thinkwam_p1.robotwin_clean_source import CLEAN_SOURCE
from thinkwam_p1.robotwin_protocol import (
    CAMERAS, METHOD, CLEAN_SOURCE_SHA256, confined_file, digest, sha256_file, snapshot, validate_clean_rgb_cache,
)


@lru_cache(maxsize=4)
def _decoder_hash(path, size, mtime_ns, ctime_ns):
    return sha256_file(path)


def _decode_clean_frames(path, start, length, image_size):
    """Yield bounded RGB blocks using one invocation for creation/verification."""
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
               '-nostdin', '-threads', '1', '-ss', f'{start:.6f}', '-i', str(path),
               '-frames:v', str(length), '-filter_threads', '1',
               '-vf', f'scale={image_size}:{image_size}', '-threads', '1',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    frame_bytes = image_size * image_size * 3
    deadline = time.monotonic() + 300
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            for offset in range(0, length, 16):
                count = min(16, length - offset)
                required = count * frame_bytes
                block = bytearray()
                while len(block) < required:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError(f'Clean RGB decode timed out: {path}')
                    data = os.read(process.stdout.fileno(), min(1024**2, required - len(block)))
                    if not data:
                        errors.seek(0)
                        raise RuntimeError(f'Clean RGB decode ended early: {path}; '
                                           f'{errors.read().decode(errors="replace")[-1200:]}')
                    block.extend(data)
                yield offset, np.frombuffer(block, dtype=np.uint8).reshape(count, image_size, image_size, 3)
            status = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            if status:
                errors.seek(0)
                raise RuntimeError(f'Clean RGB decode failed: {path}; '
                                   f'{errors.read().decode(errors="replace")[-1200:]}')
        finally:
            selector.close()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def ensure_frames(root: Path, episode: dict, view: int, image_size: int, *,
                  source_root: Path, source_files: dict) -> Path:
    """Never return an unverified RGB view, including pre-existing cache files.

    A valid receipt binds all pixels to the pinned Clean video and exact segment.
    A changed/missing receipt triggers a full view comparison, not acceptance
    based on dimensions. A mismatch leaves the existing RGB file untouched.
    """
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if not 0 <= view < len(CAMERAS) or image_size != 384:
        raise ValueError('Only the canonical three-view 384px Clean RGB contract is supported')
    video, start = episode['videos'][view]
    video = confined_file(source_root, video)
    relative_video = video.relative_to(source_root).as_posix()
    reference = CLEAN_SOURCE['files'].get(relative_video)
    if not reference or source_files.get(relative_video) != snapshot(video):
        raise ValueError('RoboTwin source video was not verified or changed since preflight')
    directory = root / f"episode_{episode['episode_index']:06d}"
    confined_file(root, directory / f'view_{view}.npy', must_exist=False)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"view_{view}.npy"
    receipt_path = directory / f'view_{view}.provenance.json'
    lock_path = confined_file(root, directory / f'view_{view}.lock', must_exist=False)
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shape = (episode['length'], image_size, image_size, 3)
        import imageio_ffmpeg
        decoder_path = imageio_ffmpeg.get_ffmpeg_exe()
        decoder_stat = snapshot(decoder_path)
        source = dict(source_identity_sha256=CLEAN_SOURCE_SHA256, video=relative_video,
                      video_sha256=reference['sha256'], video_stat=source_files[relative_video],
                      start=start, length=episode['length'], image_size=image_size,
                      camera=CAMERAS[view], resize=f'scale={image_size}:{image_size}',
                      decoder_sha256=_decoder_hash(decoder_path, decoder_stat['size'],
                                                  decoder_stat['mtime_ns'], decoder_stat['ctime_ns']))
        array = None
        before = None
        if target.exists():
            confined_file(root, target)
            before = snapshot(target)
            array = np.load(target, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError(f"Invalid RGB cache: {target}")
            if receipt_path.exists():
                confined_file(root, receipt_path)
                try:
                    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                except (ValueError, OSError):
                    receipt = {}
                if (set(receipt) == {'schema', 'method', 'source', 'cache_sha256', 'cache_stat', 'receipt_sha256'}
                        and receipt['schema'] == 1 and receipt['method'] == METHOD
                        and receipt['source'] == source and receipt['cache_stat'] == before
                        and isinstance(receipt['cache_sha256'], str) and len(receipt['cache_sha256']) == 64
                        and all(c in '0123456789abcdef' for c in receipt['cache_sha256'])
                        and receipt['receipt_sha256'] == digest({
                            key: value for key, value in receipt.items() if key != 'receipt_sha256'})):
                    return target
        required = int(np.prod(shape))
        if array is None and shutil.disk_usage(root).free < required + 100 * 1024**3:
            raise OSError("Insufficient disk for RGB cache with 100 GiB reserve")
        temp = confined_file(root, directory / f'view_{view}.{os.getpid()}.tmp', must_exist=False)
        generated = None
        try:
            if array is None:
                generated = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint8, shape=shape)
            consumed = 0
            for offset, decoded in _decode_clean_frames(video, start, episode['length'], image_size):
                if (offset != consumed or decoded.dtype != np.uint8
                        or decoded.shape[1:] != shape[1:] or not 0 < len(decoded) <= episode['length'] - consumed):
                    raise ValueError('Clean decoder returned an invalid or non-contiguous RGB block')
                if array is not None:
                    if not np.array_equal(array[offset:offset+len(decoded)], decoded):
                        raise ValueError(f'RGB cache differs from its Clean source: {target}; refusing to consume or overwrite it')
                else:
                    generated[offset:offset+len(decoded)] = decoded
                consumed += len(decoded)
            if consumed != episode['length']:
                raise ValueError('Clean decoder did not verify every RGB frame')
            if snapshot(video) != source_files[relative_video]:
                raise ValueError('Clean source video changed during RGB verification')
            if array is not None:
                if snapshot(target) != before:
                    raise ValueError('RoboTwin RGB cache changed during comparison')
            else:
                generated.flush()
                del generated
                generated = None
                os.replace(temp, target)
        finally:
            if generated is not None:
                del generated
            temp.unlink(missing_ok=True)
        before = snapshot(target)
        receipt = dict(schema=1, method=METHOD, source=source,
                       cache_sha256=sha256_file(target), cache_stat=before)
        receipt['receipt_sha256'] = digest(receipt)
        if snapshot(target) != before:
            raise ValueError('RoboTwin RGB cache changed while hashing verified pixels')
        confined_file(root, receipt_path, must_exist=False)
        temp = confined_file(root, receipt_path.with_suffix(f'.{os.getpid()}.tmp'), must_exist=False)
        try:
            with temp.open('w', encoding='utf-8') as stream:
                json.dump(receipt, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, receipt_path)
        finally:
            temp.unlink(missing_ok=True)
    return target


class OnlineRobotWinDataset(Dataset):
    def __init__(self, root: str, *, frame_cache: str, action_horizon: int,
                 action_history: int, horizons: tuple[int, ...], tokenizer_path: str,
                 text_max_length: int = 128, image_size: int = 384,
                 expected_views: int = 3, manifest_name: str = "manifest_3view.json",
                 stats_path: str = ""):
        self.root = Path(root).resolve()
        self.frame_cache = Path(frame_cache).resolve()
        self.provenance = validate_clean_rgb_cache(root, frame_cache=frame_cache,
            manifest_name=manifest_name, image_size=image_size, expected_views=expected_views,
            stats_path=stats_path)
        manifest_bytes = (self.frame_cache / manifest_name).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self.provenance['manifest_sha256']:
            raise ValueError('RoboTwin manifest changed after provenance preflight')
        manifest = json.loads(manifest_bytes)
        if (manifest['schema'] != 1 or Path(manifest['source_root']).resolve() != self.root
                or manifest['image_size'] != image_size or len(manifest['cameras']) != expected_views):
            raise ValueError('RGB cache manifest does not match the dataset configuration')
        self.cameras = tuple(manifest['cameras'])
        self.episodes = manifest['episodes']
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.action_history = action_history
        self.horizons = tuple(horizons)
        stats_file = Path(stats_path).expanduser() if stats_path else self.root / 'dataset_stats.json'
        if not stats_file.is_file():
            raise FileNotFoundError(f"RoboTwin normalization stats not found: {stats_file}")
        stats = json.loads(stats_file.read_text())
        action_stats = stats.get("action", {}).get("default", stats.get("action", {}))
        state_stats = stats.get("state", {}).get("default", stats.get("observation.state", {}))
        self.action_mean = np.asarray(action_stats.get("global_mean", action_stats.get("mean")), dtype=np.float32)
        self.action_std = np.asarray(action_stats.get("global_std", action_stats.get("std")), dtype=np.float32)
        self.state_mean = np.asarray(state_stats.get("global_mean", state_stats.get("mean")), dtype=np.float32)
        self.state_std = np.asarray(state_stats.get("global_std", state_stats.get("std")), dtype=np.float32)
        if any(array.shape != (14,) for array in (self.action_mean, self.action_std, self.state_mean, self.state_std)):
            raise ValueError(f"RoboTwin stats must contain 14-dimensional action/state statistics: {stats_file}")
        self.action_std = np.maximum(self.action_std, 1e-6)
        self.state_std = np.maximum(self.state_std, 1e-6)
        maximum = max(action_horizon - 1, *self.horizons)
        counts = [max(0, e['length'] - maximum) for e in self.episodes]
        self.ends = np.cumsum(counts).tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError('No valid RoboTwin windows')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        languages = list(dict.fromkeys(e['language'] for e in self.episodes))
        encoded = tokenizer(languages, max_length=text_max_length, padding='max_length',
                            truncation=True, return_tensors='np')
        self.language = {text: (encoded['input_ids'][i].astype(np.int64),
                               encoded['attention_mask'][i].astype(np.bool_))
                         for i, text in enumerate(languages)}
        self._loaded = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def _episode(self, index):
        e = self.episodes[index]
        directory = self.frame_cache / f"episode_{e['episode_index']:06d}"
        label_path = confined_file(self.frame_cache, directory / 'labels.npz')
        label_key = label_path.relative_to(self.frame_cache).as_posix()
        if snapshot(label_path) != self.provenance['labels'].get(label_key):
            raise ValueError('RoboTwin labels changed after their Clean source comparison')
        image_paths = [ensure_frames(self.frame_cache, e, view, self.image_size,
                       source_root=self.root, source_files=self.provenance['source_files'])
                       for view in range(len(self.cameras))]
        image_versions = [snapshot(path) for path in image_paths]
        if index in self._loaded:
            arrays, loaded_versions = self._loaded[index]
            if loaded_versions == image_versions:
                self._loaded.move_to_end(index)
                return arrays
            # An atomic replacement can leave an old mmap alive on a different
            # (possibly damaged) inode even after the path's new file validates.
            del self._loaded[index]
        with np.load(label_path, allow_pickle=False) as labels:
            action = np.asarray(labels['action'], dtype=np.float32)
            state = np.asarray(labels['state'], dtype=np.float32)
        if action.shape != (e['length'], 14) or state.shape != action.shape:
            raise ValueError(f'Invalid ALOHA actions/state in {directory}')
        action = (action - self.action_mean) / self.action_std
        state = (state - self.state_mean) / self.state_std
        images = [np.load(path, mmap_mode='r', allow_pickle=False) for path in image_paths]
        if [snapshot(path) for path in image_paths] != image_versions:
            raise ValueError('RoboTwin RGB cache changed while opening verified memory maps')
        arrays = (action, state, images)
        self._loaded[index] = (arrays, image_versions)
        # Eviction releases mmap handles, not shared on-disk RGB pixels.
        if len(self._loaded) > 16:
            self._loaded.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[episode_index - 1] if episode_index else 0)
        e = self.episodes[episode_index]
        action, state, images = self._episode(episode_index)
        history = np.zeros((self.action_history, 14), dtype=np.float32)
        previous = action[max(0, t - self.action_history):t]
        if len(previous):
            history[-len(previous):] = previous
        ids, mask = self.language[e['language']]
        return {
            'obs_images': np.stack([view[t] for view in images]),
            'future_images': {h: np.stack([view[t + h] for view in images]) for h in self.horizons},
            'actions': torch.from_numpy(action[t:t + self.action_horizon].copy()),
            'action_history': torch.from_numpy(history),
            'proprio': torch.from_numpy(state[t].copy()),
            'text_input_ids': torch.from_numpy(ids.copy()),
            'attention_mask': torch.from_numpy(mask.copy()),
        }
