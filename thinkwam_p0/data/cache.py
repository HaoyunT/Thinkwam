"""Feature cache adapter shared by LIBERO, RoboTwin 2.0, and RoboCasa24.

The cache format deliberately decouples expensive frozen V-JEPA encoding from
the trainable P0 loop.  A directory contains ``manifest.json`` and one or more
``.npz`` records with keys: obs_features, future_t7, future_t15, future_t31,
actions, action_history, and ``text_input_ids`` plus ``attention_mask`` for
tokenized instructions, or optional precomputed ``text_features``.
"""

from __future__ import annotations

import json
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureCacheDataset(Dataset):
    def __init__(self, root: str, horizons=(7, 15, 31)):
        self.root = Path(root).expanduser()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Build V-JEPA feature records before training. "
                "See docs/P0_DATA_CACHE.md."
            )
        manifest = json.loads(manifest_path.read_text())
        raw_records = manifest["records"]
        self.shards = []
        self.shard_lengths = []
        self._is_sharded = bool(raw_records and isinstance(raw_records[0], dict))
        if self._is_sharded:
            for record in raw_records:
                self.shards.append(self.root / record["file"])
                self.shard_lengths.append(int(record["length"]))
        else:
            # A filename entry contains one record without a shard dimension.
            self.shards = [self.root / record for record in raw_records]
            self.shard_lengths = [1] * len(self.shards)
        self._ends = np.cumsum(self.shard_lengths).tolist()
        self.horizons = tuple(horizons)
        self._loaded_shards = {}
        if not self.shards:
            raise ValueError(f"Feature cache is empty: {self.root}")

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        shard_index = bisect.bisect_right(self._ends, index)
        previous_end = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local_index = index - previous_end
        if shard_index not in self._loaded_shards:
            with np.load(self.shards[shard_index], allow_pickle=False) as loaded:
                self._loaded_shards[shard_index] = {key: loaded[key] for key in loaded.files}
            # Keep only the most recent two shards to bound host RAM.
            for old_index in list(self._loaded_shards)[:-2]:
                del self._loaded_shards[old_index]
        data = self._loaded_shards[shard_index]
        record = _ArrayRecord(data, local_index) if self._is_sharded else data
        sample = {
            "obs_features": torch.from_numpy(record["obs_features"]).float(),
            "future_features": {h: torch.from_numpy(record[f"future_t{h}"]).float() for h in self.horizons},
            "actions": torch.from_numpy(record["actions"]).float(),
            "action_history": torch.from_numpy(record["action_history"]).float(),
            "text_features": (
                torch.from_numpy(record["text_features"]).float()
                if "text_features" in record
                else None
            ),
            "text_input_ids": (
                torch.from_numpy(record["text_input_ids"]).long()
                if "text_input_ids" in record
                else None
            ),
            "attention_mask": (
                torch.from_numpy(record["attention_mask"]).bool()
                if "attention_mask" in record
                else None
            ),
            # P1 cache records may carry camera-space affordance maps.
            "affordance_targets": (
                torch.from_numpy(record["affordance_targets"]).float()
                if "affordance_targets" in record
                else None
            ),
        }
        sample["proprio"] = torch.from_numpy(record["proprio"]).float() if "proprio" in record else None
        return sample


class _ArrayRecord(dict):
    """Index a loaded shard without copying individual fields."""

    def __init__(self, arrays, index):
        super().__init__({key: value[index] for key, value in arrays.items()})
