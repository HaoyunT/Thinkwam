"""Online RGB/affordance adapter for the released DeVA RoboCasa24 data.

The DeVA export stores the three RoboCasa cameras in a tiled MP4 and keeps
per-view affordance maps in ``affordance/<episode>.npz``.  This adapter
untile-reads the cameras and returns the same sample contract as the LIBERO
online adapter, while keeping the simulator-derived labels as training-only
targets.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from thinkwam_p1.robocasa_protocol import read_source_json, source_path, validate_action_statistics


VIEW_NAMES = ("agentview_left", "agentview_right", "eye_in_hand")


def _normalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Apply the released RoboCasa min-max bounds to ``[-1, 1]``."""
    denominator = high - low
    return np.clip(2.0 * (action - low) / denominator - 1.0, -1.0, 1.0).astype(np.float32)


class RoboCasa24Dataset(Dataset):
    """Return 3-view RGB windows, normalized actions, and DeVA maps.

    The videos are the public DeVA 2x2 tiles (bottom-right is blank).  No
    simulator state or ground-truth object IDs are returned; affordance maps
    are used only as supervised targets during training.
    """

    def __init__(
        self,
        root: str,
        *,
        action_horizon: int = 16,
        action_history: int = 8,
        horizons: tuple[int, ...] = (7, 15, 31),
        tokenizer_path: Optional[str] = None,
        text_max_length: int = 128,
        require_affordance: bool = True,
        future_target_cache: str = "",
        future_token_counts: tuple[int, ...] = (288, 144, 72),
        vjepa_checkpoint: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.action_history = int(action_history)
        self.horizons = tuple(int(h) for h in horizons)
        self.text_max_length = int(text_max_length)
        if self.action_horizon < 1 or self.action_history < 0 or any(h < 1 for h in self.horizons):
            raise ValueError("action horizon and future offsets must be positive; history must be nonnegative")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        info_path = source_path(self.root, "dataset_info.json")
        if not info_path.is_file():
            raise FileNotFoundError(f"RoboCasa24 dataset_info.json missing: {info_path}")
        self.info = read_source_json(self.root, "dataset_info.json")
        if tuple(self.info.get("views", ())) != VIEW_NAMES:
            raise ValueError(f"unexpected RoboCasa24 view order: {self.info.get('views')}")
        self.tile = self.info.get("tile")
        if self.tile != [["agentview_left", "agentview_right"], ["eye_in_hand", None]]:
            raise ValueError(f"unsupported RoboCasa24 tile layout: {self.tile}")

        self.actions = read_source_json(self.root, "action.json")
        stats = read_source_json(self.root, "norm_stats.json")
        self.action_low, self.action_high = validate_action_statistics(stats)

        video_dir = source_path(self.root, "videos")
        self.video_paths = sorted(video_dir.glob("demo_*.mp4"))
        if not self.video_paths:
            raise FileNotFoundError(f"No demo_*.mp4 videos under {self.root / 'videos'}")
        if not isinstance(self.actions, dict) or set(self.actions) != {path.stem for path in self.video_paths}:
            raise ValueError("RoboCasa24 action/video episode sets must match exactly")
        self.affordance_dir = source_path(self.root, "affordance") if require_affordance else self.root / "affordance"
        if require_affordance and not self.affordance_dir.is_dir():
            raise FileNotFoundError(f"RoboCasa24 affordance directory missing: {self.affordance_dir}")
        self.require_affordance = bool(require_affordance)
        self.future_cache = None
        if future_target_cache:
            from thinkwam_p0.data.future_cache import FutureTargetCache

            if not vjepa_checkpoint:
                raise ValueError("future targets require the frozen encoder checkpoint identity")
            self.future_cache = FutureTargetCache(
                future_target_cache, dataset_root=self.root, checkpoint=vjepa_checkpoint,
                horizons=self.horizons, token_counts=future_token_counts,
                action_horizon=self.action_horizon,
            )
            self.future_cache.validate_files(self.root)

        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, use_fast=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self._language: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._readers: OrderedDict[str, object] = OrderedDict()
        self._affordance_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.index: list[tuple[str, int]] = []
        self.episode_ids: list[int] = []
        self.episode_lengths: dict[str, int] = {}
        self.episode_tasks: dict[str, str] = {}
        mapping_path = source_path(self.root, "data_mapping.json")
        mapping = read_source_json(self.root, "data_mapping.json") if mapping_path.is_file() else {}
        tasks_by_id = {
            episode_id: task
            for task, record in mapping.items()
            for episode_id in range(int(record["start_idx"]), int(record["end_idx"]) + 1)
        }
        self.action_dim = 7
        self.proprio_dim = 1  # RoboCasa has no proprioceptive state in this release.

        # Decord is imported lazily so metadata validation and unit tests do not
        # require a video decoder until a sample is actually read.
        from decord import VideoReader, cpu

        for path in self.video_paths:
            episode = path.stem
            # Resolve each asset, including links, before allowing the decoder to read it.
            source_path(self.root, f"videos/{path.name}")
            if episode not in self.actions:
                raise KeyError(f"{episode} missing from action.json")
            action = np.asarray(self.actions[episode], dtype=np.float32)
            if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7:
                raise ValueError(f"{episode} actions must have shape [T,7], got {action.shape}")
            if not np.isfinite(action).all():
                raise ValueError(f"{episode} actions contain nonfinite labels")
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            frame_count = len(reader)
            if frame_count != len(action):
                raise ValueError(f"{episode} video/action length mismatch: {frame_count} != {len(action)}")
            if self.require_affordance:
                label_path = source_path(self.root, f"affordance/{episode}.npz")
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
            self._language[episode] = self._tokenize(self._read_instruction(episode))
            episode_id = int(episode.removeprefix("demo_"))
            self.episode_ids.append(episode_id)
            self.episode_lengths[episode] = frame_count
            self.episode_tasks[episode] = tasks_by_id.get(episode_id, "unmapped")
            # Supervision availability is independent for actions and each future horizon.
            for t in range(frame_count):
                self.index.append((episode, t))
            del reader

    def _read_instruction(self, episode: str) -> str:
        path = source_path(self.root, f"metas/{episode}.txt")
        if not path.is_file():
            raise FileNotFoundError(f"RoboCasa24 instruction missing: {path}")
        instruction = path.read_text().strip()
        if not instruction:
            raise ValueError(f"RoboCasa24 instruction is empty: {path}")
        return instruction

    def _tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        if self.tokenizer is None:
            ids = np.zeros((self.text_max_length,), dtype=np.int64)
            mask = np.zeros((self.text_max_length,), dtype=np.bool_)
            mask[0] = True
            return ids, mask
        encoded = self.tokenizer(
            text,
            max_length=self.text_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return encoded["input_ids"][0].astype(np.int64), encoded["attention_mask"][0].astype(np.bool_)

    def __len__(self) -> int:
        return len(self.index)

    def _reader(self, episode: str):
        reader = self._readers.get(episode)
        if reader is None:
            from decord import VideoReader, cpu

            reader = VideoReader(str(source_path(self.root, f"videos/{episode}.mp4")), ctx=cpu(0), num_threads=1)
            self._readers[episode] = reader
            if len(self._readers) > 8:
                _, old_reader = self._readers.popitem(last=False)
                try:
                    old_reader.close()
                except Exception:
                    pass
        else:
            self._readers.move_to_end(episode)
        return reader

    @staticmethod
    def _untile(frames: np.ndarray) -> np.ndarray:
        """Convert tiled [N,448,448,3] frames to [N,3,224,224,3]."""
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[1] % 2 or frames.shape[2] % 2:
            raise ValueError(f"unexpected tiled RoboCasa frame shape: {frames.shape}")
        height, width = frames.shape[1] // 2, frames.shape[2] // 2
        return np.stack(
            [
                frames[:, :height, :width],
                frames[:, :height, width:],
                frames[:, height:, :width],
            ],
            axis=1,
        ).astype(np.uint8, copy=False)

    def _load_affordance(self, episode: str, frame: int) -> torch.Tensor:
        path = source_path(self.root, f"affordance/{episode}.npz")
        arrays = self._affordance_cache.get(episode)
        if arrays is None:
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {}
                for view in VIEW_NAMES:
                    if view not in loaded:
                        raise KeyError(f"{path} missing affordance view {view}")
                    raw = loaded[view]
                    if (raw.ndim != 3 or raw.shape[0] != self.episode_lengths[episode]
                            or min(raw.shape[1:]) < 1 or raw.dtype.kind != "f"):
                        raise ValueError(f"Invalid affordance shape/dtype: {path}/{view} {raw.shape} {raw.dtype}")
                    value = np.asarray(raw, dtype=np.float32)
                    if not np.isfinite(value).all() or (value < 0).any():
                        raise ValueError(f"Affordance labels must be finite and nonnegative: {path}/{view}")
                    arrays[view] = value
                if len({array.shape for array in arrays.values()}) != 1:
                    raise ValueError(f"RoboCasa24 affordance views have different shapes: {path}")
            self._affordance_cache[episode] = arrays
            if len(self._affordance_cache) > 8:
                self._affordance_cache.popitem(last=False)
        else:
            self._affordance_cache.move_to_end(episode)
        maps = []
        for view in VIEW_NAMES:
            value = arrays[view][frame]
            maximum = float(value.max())
            maps.append(value / maximum if maximum > 1e-6 else np.zeros_like(value))
        return torch.from_numpy(np.stack(maps, axis=0).astype(np.float32, copy=False))

    def __getitem__(self, index: int) -> dict:
        episode, t = self.index[index]
        action = _normalize_action(np.asarray(self.actions[episode], dtype=np.float32), self.action_low, self.action_high)
        history = np.zeros((self.action_history, self.action_dim), dtype=np.float32)
        previous = action[max(0, t - self.action_history):t]
        if len(previous):
            history[-len(previous):] = previous
        length = len(action)
        future_masks = {h: torch.tensor(t + h < length, dtype=torch.bool) for h in self.horizons}
        frame_ids = [t] if self.future_cache else sorted({t, *(t + h for h in self.horizons if t + h < length)})
        frames = self._untile(self._reader(episode).get_batch(frame_ids).asnumpy())
        by_frame = {frame: frames[i] for i, frame in enumerate(frame_ids)}
        valid_actions = min(self.action_horizon, length - t)
        target_actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        target_actions[:valid_actions] = action[t:t + valid_actions]
        action_mask = torch.arange(self.action_horizon) < valid_actions
        ids, mask = self._language[episode]
        return {
            "obs_images": by_frame[t],
            "future_images": None if self.future_cache else {
                h: by_frame[t + h] if t + h < length else np.zeros_like(by_frame[t])
                for h in self.horizons
            },
            "future_features": self.future_cache.get(episode, t) if self.future_cache else None,
            "actions": torch.from_numpy(target_actions),
            "action_mask": action_mask,
            "future_masks": future_masks,
            "action_history": torch.from_numpy(history),
            "text_input_ids": torch.from_numpy(ids.copy()).long(),
            "attention_mask": torch.from_numpy(mask.copy()).bool(),
            "proprio": torch.zeros((self.proprio_dim,), dtype=torch.float32),
            "affordance_targets": self._load_affordance(episode, t) if self.require_affordance else None,
        }

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        for reader in getattr(self, "_readers", {}).values():
            try:
                reader.close()
            except Exception:
                pass
