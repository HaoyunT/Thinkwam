#!/usr/bin/env python3
"""Build standalone RoboTwin latent-feature records for cache diagnostics.

This utility decodes LeRobot v3 AV1 videos with imageio-ffmpeg and writes the
NPZ/manifest format consumed by FeatureCacheDataset. Its configurable variants,
token budgets and optional contact labels are not the paper training protocol.
The current RoboTwin recipe uses Clean RGB through prepare_robotwin_rgb.py and
OnlineRobotWinDataset, with a 32-action chunk and no affordance guidance; its
source guard rejects these latent records as training inputs.

Source Parquet/video files are read-only. Only windows with a complete action
chunk and every requested future frame are written to the output cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thinkwam_p0.data.libero_inputs import make_image_batch, uniform_pool
from thinkwam_p0.data.robotwin import (
    ROBOTWIN_AFFORDANCE_CAMERA_ALIASES,
    ROBOTWIN_CAMERAS,
    RobotWinEpisode,
    RobotWinIndex,
    episode_split,
    load_affordance_episode,
)
from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21
from thinkwam_p1.data.robotwin_affordance import deva_normalize_and_pool, frame_to_heatmaps


def _parse_ints(value: str, expected: int) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != expected or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"expected {expected} positive comma-separated integers")
    return values


def _ffmpeg_path(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        path = shutil.which("ffmpeg")
        if path:
            return path
        raise RuntimeError("AV1 decoding needs imageio-ffmpeg or an ffmpeg executable")


def _decode_segment(
    ffmpeg: str,
    path: Path,
    *,
    start_seconds: float,
    frames: int,
    image_size: int,
) -> np.ndarray:
    """Decode one episode segment to RGB uint8 at the encoder resolution."""

    if frames <= 0:
        return np.empty((0, image_size, image_size, 3), dtype=np.uint8)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{max(0.0, float(start_seconds)):.6f}",
        "-i",
        str(path),
        "-frames:v",
        str(int(frames)),
        "-vf",
        f"scale={image_size}:{image_size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = image_size * image_size * 3
    values = bytearray()
    assert process.stdout is not None
    while len(values) < frame_bytes * frames:
        chunk = process.stdout.read(min(frame_bytes * 32, frame_bytes * frames - len(values)))
        if not chunk:
            break
        values.extend(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    expected = frame_bytes * frames
    if return_code != 0 or len(values) != expected:
        detail = stderr.strip()[-1000:]
        raise RuntimeError(
            f"ffmpeg failed for {path} at {start_seconds:.3f}s: "
            f"returncode={return_code}, decoded={len(values) // frame_bytes}/{frames}; {detail}"
        )
    return np.frombuffer(values, dtype=np.uint8).reshape(frames, image_size, image_size, 3).copy()


class _StreamingVideoDecoder:
    """Sequential raw-frame reader used to avoid one ffmpeg process per episode."""

    def __init__(self, ffmpeg: str, path: Path, image_size: int) -> None:
        self.path = path
        self.image_size = int(image_size)
        self.frame_bytes = self.image_size * self.image_size * 3
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-vf",
            f"scale={self.image_size}:{self.image_size}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.position = 0

    def _read_bytes(self, count: int) -> bytes:
        assert self.process.stdout is not None
        values = bytearray()
        while len(values) < count:
            chunk = self.process.stdout.read(min(self.frame_bytes * 32, count - len(values)))
            if not chunk:
                break
            values.extend(chunk)
        return bytes(values)

    def skip(self, frames: int) -> None:
        frames = max(0, int(frames))
        if frames:
            decoded = self._read_bytes(frames * self.frame_bytes)
            self.position += len(decoded) // self.frame_bytes
            if len(decoded) != frames * self.frame_bytes:
                raise RuntimeError(f"unexpected EOF while seeking {self.path} to frame {self.position + frames}")

    def read(self, frames: int) -> np.ndarray:
        frames = max(0, int(frames))
        decoded = self._read_bytes(frames * self.frame_bytes)
        actual = len(decoded) // self.frame_bytes
        self.position += actual
        if actual != frames:
            raise RuntimeError(f"unexpected EOF in {self.path}: requested {frames}, got {actual}")
        return np.frombuffer(decoded, dtype=np.uint8).reshape(frames, self.image_size, self.image_size, 3).copy()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


class _StreamingVideoBank:
    def __init__(self, ffmpeg: str, image_size: int) -> None:
        self.ffmpeg = ffmpeg
        self.image_size = image_size
        self.readers: dict[str, _StreamingVideoDecoder] = {}

    def read(self, camera: str, path: Path, start_frame: int, frames: int) -> np.ndarray:
        reader = self.readers.get(camera)
        if reader is None or reader.path != path or start_frame < reader.position:
            if reader is not None:
                reader.close()
            reader = _StreamingVideoDecoder(self.ffmpeg, path, self.image_size)
            self.readers[camera] = reader
        if start_frame > reader.position:
            reader.skip(start_frame - reader.position)
        return reader.read(frames)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


def _downsample_maps(maps: dict[str, np.ndarray], cameras: tuple[str, ...], size: int) -> np.ndarray:
    output = []
    for camera in cameras:
        array = np.asarray(maps[camera], dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"affordance array for {camera} must be [T,H,W], got {array.shape}")
        # DeVA normalizes each view/frame before exact area pooling.
        output.append(deva_normalize_and_pool(array, (size, size)))
    return np.stack(output, axis=1).astype(np.float16)


def _find_affordance_jsonl(root: Path, episode: RobotWinEpisode) -> Path | None:
    """Find simulator contact records without requiring rasterized NPZ files."""
    candidates = (
        root / episode.variant / str(episode.episode_index) / "affordance.jsonl",
        root / episode.variant / f"episode{episode.episode_index}" / "affordance.jsonl",
        root / episode.variant / f"{episode.episode_index}.jsonl",
        root / f"{episode.variant}_{episode.episode_index}.jsonl",
        root / str(episode.episode_index) / "affordance.jsonl",
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_jsonl_label_maps(
    root: Path,
    episode: RobotWinEpisode,
    cameras: tuple[str, ...],
    *,
    map_size: int,
) -> np.ndarray | None:
    """Rasterize contact JSONL one frame at a time and retain only 24x24 maps."""
    path = _find_affordance_jsonl(root, episode)
    if path is None:
        return None
    result = np.zeros((episode.length, len(cameras), map_size, map_size), dtype=np.float16)
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            if frame_index < 0 or frame_index >= episode.length:
                raise ValueError(
                    f"{path}:{line_number} frame_index={frame_index} outside episode length {episode.length}"
                )
            if frame_index in seen:
                raise ValueError(f"{path}:{line_number} duplicate frame_index={frame_index}")
            seen.add(frame_index)
            maps = frame_to_heatmaps(record)
            for camera_index, camera in enumerate(cameras):
                short = camera.rsplit(".", 1)[-1]
                names = ROBOTWIN_AFFORDANCE_CAMERA_ALIASES.get(short, (short, camera))
                source_name = next((name for name in names if name in maps), None)
                if source_name is None:
                    raise KeyError(
                        f"{path}:{line_number} has no camera for {camera}; available={sorted(maps)}"
                    )
                result[frame_index, camera_index] = deva_normalize_and_pool(
                    maps[source_name][None], (map_size, map_size)
                )[0]
    if not seen:
        raise ValueError(f"{path} contains no frame records")
    return result


def _load_label_maps(
    affordance_root: Path | None,
    affordance_jsonl_root: Path | None,
    episode: RobotWinEpisode,
    cameras: tuple[str, ...],
    *,
    map_size: int,
    require: bool,
) -> np.ndarray | None:
    if affordance_root is not None and affordance_jsonl_root is not None:
        raise ValueError("provide only one of --affordance-root and --affordance-jsonl-root")
    if affordance_jsonl_root is not None:
        maps = _load_jsonl_label_maps(
            affordance_jsonl_root,
            episode,
            cameras,
            map_size=map_size,
        )
        if maps is None and require:
            expected = affordance_jsonl_root / episode.variant / str(episode.episode_index) / "affordance.jsonl"
            raise FileNotFoundError(f"Missing affordance JSONL export: {expected}")
        return maps
    if affordance_root is None:
        if require:
            raise FileNotFoundError(
                "--require-affordance was set but --affordance-root was not supplied"
            )
        return None
    maps = load_affordance_episode(affordance_root, episode, cameras)
    if maps is None:
        if require:
            expected = affordance_root / episode.variant / str(episode.episode_index)
            raise FileNotFoundError(f"Missing affordance export for {episode.variant}/{episode.episode_index}: {expected}")
        return None
    lengths = {camera: int(np.asarray(array).shape[0]) for camera, array in maps.items()}
    if any(length < episode.length for length in lengths.values()):
        raise ValueError(
            f"affordance maps shorter than episode {episode.variant}/{episode.episode_index}: "
            f"{lengths}, expected {episode.length}"
        )
    return _downsample_maps(maps, cameras, map_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/data/thinkwam/datasets/robotwin2.0-lerobot-v3.0")
    parser.add_argument("--variant", choices=("clean", "random", "full"), default="clean")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vjepa-checkpoint", default="/data/thinkwam/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--qwen-path", default="/data/thinkwam/models/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all valid windows")
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--horizons", default="7,15,31", type=lambda value: _parse_ints(value, 3))
    parser.add_argument("--future-token-counts", default="64,32,16", type=lambda value: _parse_ints(value, 3))
    parser.add_argument("--obs-token-count", type=int, default=64)
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--camera-keys", default="cam_high,cam_left_wrist,cam_right_wrist")
    parser.add_argument("--affordance-root", default="")
    parser.add_argument(
        "--affordance-jsonl-root",
        default="",
        help="simulator contact JSONL root; rasterizes maps in memory and writes only 24x24 targets",
    )
    parser.add_argument("--affordance-map-size", type=int, default=24)
    parser.add_argument("--require-affordance", action="store_true")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument(
        "--decode-mode",
        choices=("stream", "segment"),
        default="stream",
        help="stream each AV1 shard once (default) or seek a fresh ffmpeg process per episode",
    )
    parser.add_argument("--compress", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="validate Parquet metadata and count windows without loading V-JEPA or decoding videos",
    )
    return parser.parse_args()


def _camera_keys(value: str) -> tuple[str, ...]:
    aliases = {camera.rsplit(".", 1)[-1]: camera for camera in ROBOTWIN_CAMERAS}
    selected = tuple(aliases.get(item.strip(), item.strip()) for item in value.split(",") if item.strip())
    if not selected or any(item not in ROBOTWIN_CAMERAS for item in selected):
        raise ValueError(f"--camera-keys must select RoboTwin cameras, got {selected}")
    return selected


@torch.no_grad()
def encode_views(encoder, views, device):
    """Encode an arbitrary ordered set of camera views as [B,V,N,D]."""
    if not views:
        raise ValueError("at least one camera view is required")
    count = len(views[0])
    if any(len(sample) != count for sample in views):
        raise ValueError("all camera batches must have the same length")
    images = torch.stack(
        [make_image_batch(encoder, [sample[view] for sample in views], device)
         for view in range(count)],
        dim=1,
    )
    return encoder(images).float().cpu().numpy()


def _tokenizer(path: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def encode(text: str) -> tuple[np.ndarray, np.ndarray]:
        if text not in cache:
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )
            cache[text] = (
                encoded["input_ids"][0].astype(np.int64),
                encoded["attention_mask"][0].astype(np.bool_),
            )
        return cache[text]

    return encode


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.shard_size <= 0 or args.action_horizon <= 0:
        raise ValueError("batch-size, shard-size, and action-horizon must be positive")
    if args.obs_token_count <= 0 or args.image_size <= 0 or args.affordance_map_size <= 0:
        raise ValueError("obs-token-count, image-size, and affordance-map-size must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between zero and one")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    cameras = _camera_keys(args.camera_keys)
    horizons = tuple(args.horizons)
    future_counts = tuple(args.future_token_counts)
    output = Path(args.output).expanduser()
    index = RobotWinIndex(args.dataset_root, variant=args.variant)
    selected_episodes = [
        episode
        for episode_number, episode in enumerate(index.episodes)
        if episode_number % args.world_size == args.rank
    ]
    def keep_episode(item: RobotWinEpisode) -> bool:
        if args.split == "all":
            return True
        return episode_split(item.variant, item.episode_index, args.seed, args.val_fraction) == args.split

    total_windows = sum(
        len(index.window_starts(item.length, action_horizon=args.action_horizon, horizons=horizons, stride=args.window_stride))
        for item in selected_episodes
        if keep_episode(item)
    )
    print(json.dumps({
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "variant": args.variant,
        "rank": args.rank,
        "world_size": args.world_size,
        "episodes": len(selected_episodes),
        "windows": total_windows,
        "cameras": list(cameras),
    }, sort_keys=True), flush=True)
    if args.index_only:
        return

    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    if args.affordance_root and args.affordance_jsonl_root:
        raise ValueError("provide only one of --affordance-root and --affordance-jsonl-root")
    affordance_root = Path(args.affordance_root).expanduser() if args.affordance_root else None
    affordance_jsonl_root = Path(args.affordance_jsonl_root).expanduser() if args.affordance_jsonl_root else None
    if args.require_affordance:
        if affordance_root is None and affordance_jsonl_root is None:
            raise FileNotFoundError(
                "--require-affordance needs --affordance-root or --affordance-jsonl-root"
            )
        source_root = affordance_root or affordance_jsonl_root
        assert source_root is not None
        if not source_root.is_dir():
            raise FileNotFoundError(f"affordance source root does not exist: {source_root}")
    device = torch.device(args.device)
    encoder = FrozenVJEPA21(args.vjepa_checkpoint).to(device)
    tokenize = _tokenizer(args.qwen_path, args.text_max_length)
    ffmpeg = _ffmpeg_path(args.ffmpeg)
    decoder_bank = _StreamingVideoBank(ffmpeg, args.image_size) if args.decode_mode == "stream" else None
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, int | str]] = []
    shard_samples: list[dict[str, np.ndarray]] = []
    shard_index = 0
    written = 0
    pending: list[dict[str, object]] = []
    label_presence: bool | None = None

    def flush_shard() -> None:
        nonlocal shard_index
        if not shard_samples:
            return
        arrays = {key: np.stack([sample[key] for sample in shard_samples], axis=0) for key in shard_samples[0]}
        filename = f"shard_{shard_index:05d}.npz"
        writer = np.savez_compressed if args.compress else np.savez
        writer(output / filename, **arrays)
        records.append({"file": filename, "length": len(shard_samples)})
        print(f"wrote {filename} ({len(shard_samples)} records)", flush=True)
        shard_index += 1
        shard_samples.clear()

    def flush_pending() -> None:
        nonlocal written
        if not pending:
            return
        obs = encode_views(encoder, [item["current"] for item in pending], device)
        future = {
            horizon: encode_views(
                encoder,
                [item["future"][horizon] for item in pending],
                device,
            )
            for horizon in horizons
        }
        for row, item in enumerate(pending):
            sample = {
                "obs_features": uniform_pool(obs[row:row + 1], args.obs_token_count)[0].astype(np.float16),
                "actions": np.asarray(item["actions"], dtype=np.float32),
                "action_history": np.asarray(item["action_history"], dtype=np.float32),
                "text_input_ids": np.asarray(item["text_input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(item["attention_mask"], dtype=np.bool_),
                "proprio": np.asarray(item["proprio"], dtype=np.float32),
            }
            for horizon, count in zip(horizons, future_counts):
                sample[f"future_t{horizon}"] = uniform_pool(
                    future[horizon][row:row + 1], count
                )[0].astype(np.float16)
            if item.get("affordance_targets") is not None:
                sample["affordance_targets"] = np.asarray(item["affordance_targets"], dtype=np.float16)
            shard_samples.append(sample)
            written += 1
            if len(shard_samples) >= args.shard_size:
                flush_shard()
        pending.clear()
        print(f"cached {written} records", flush=True)

    for episode in selected_episodes:
        if not keep_episode(episode):
            continue
        starts = list(index.window_starts(
            episode.length,
            action_horizon=args.action_horizon,
            horizons=horizons,
            stride=args.window_stride,
        ))
        if not starts:
            continue
        arrays = index.load_episode(episode)
        label_maps = _load_label_maps(
            affordance_root,
            affordance_jsonl_root,
            episode,
            cameras,
            map_size=args.affordance_map_size,
            require=args.require_affordance,
        )
        current_label_presence = label_maps is not None
        if label_presence is None:
            label_presence = current_label_presence
        elif label_presence != current_label_presence:
            raise RuntimeError(
                "affordance coverage is mixed across episodes; provide labels for every selected episode "
                "or omit --affordance-root"
            )
        if decoder_bank is None:
            frames = {
                camera: _decode_segment(
                    ffmpeg,
                    episode.video_path(camera),
                    start_seconds=episode.video_start(camera),
                    frames=episode.length,
                    image_size=args.image_size,
                )
                for camera in cameras
            }
        else:
            frames = {
                camera: decoder_bank.read(
                    camera,
                    episode.video_path(camera),
                    int(round(episode.video_start(camera) * args.fps)),
                    episode.length,
                )
                for camera in cameras
            }
        for start in starts:
            if args.max_records and written + len(pending) >= args.max_records:
                break
            text_ids, attention = tokenize(str(arrays["task"]))
            history = np.zeros((args.history_steps, 14), dtype=np.float32)
            previous = np.asarray(arrays["action"])[max(0, start - args.history_steps):start]
            if len(previous):
                history[-len(previous):] = previous
            item: dict[str, object] = {
                "current": tuple(frames[camera][start] for camera in cameras),
                "future": {
                    horizon: tuple(frames[camera][start + horizon] for camera in cameras)
                    for horizon in horizons
                },
                "actions": np.asarray(arrays["action"])[start:start + args.action_horizon],
                "action_history": history,
                "text_input_ids": text_ids,
                "attention_mask": attention,
                "proprio": np.asarray(arrays["state"])[start],
            }
            if label_maps is not None:
                # Label maps are stored as [T,V,H,W] to match the exporter;
                # one cache sample carries the current frame [V,H,W].
                item["affordance_targets"] = label_maps[start]
            pending.append(item)
            if len(pending) >= args.batch_size:
                flush_pending()
        if args.max_records and written >= args.max_records:
            break

    flush_pending()
    if decoder_bank is not None:
        decoder_bank.close()
    flush_shard()
    if args.max_records and written > args.max_records:
        raise RuntimeError("cache writer exceeded --max-records; reduce batch-size or report this bug")
    if not records:
        raise RuntimeError("No valid RoboTwin windows were written")
    manifest = {
        "dataset": "robotwin2.0",
        "variant": args.variant,
        "split": args.split,
        "records": records,
        "shard_size": args.shard_size,
        "horizons": list(horizons),
        "future_token_counts": list(future_counts),
        "obs_token_count": args.obs_token_count,
        "action_horizon": args.action_horizon,
        "history_steps": args.history_steps,
        "cameras": list(cameras),
        "token_source": "frozen V-JEPA 2.1",
        "vjepa_checkpoint": args.vjepa_checkpoint,
        "affordance_targets": bool(label_presence),
        "source_dataset_root": str(Path(args.dataset_root).expanduser()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": written, "manifest": str(output / "manifest.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
