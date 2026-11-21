"""Raw-image LIBERO dataset for online full-token V-JEPA training.

The dataset intentionally stores only HDF5 indices.  V-JEPA is run inside the
model on each batch, matching the public JEPA-WAM data path and avoiding a
large per-sample latent cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class OnlineLiberoDataset(Dataset):
    """Return two-view RGB observations and future RGB targets from LIBERO HDF5."""

    def __init__(
        self,
        root: str,
        *,
        action_horizon: int = 16,
        action_history: int = 8,
        horizons: tuple[int, ...] = (7, 15, 31),
        tokenizer_path: Optional[str] = None,
        text_max_length: int = 128,
    ) -> None:
        self.root = Path(root).expanduser()
        self.action_horizon = int(action_horizon)
        self.action_history = int(action_history)
        self.horizons = tuple(int(h) for h in horizons)
        self.text_max_length = int(text_max_length)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.files = sorted(self.root.rglob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"No LIBERO HDF5 files under {self.root}")

        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self._handles: dict[str, h5py.File] = {}
        self._language: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.index: list[tuple[str, str, int]] = []
        self.action_dim = None
        self.proprio_dim = None
        for path in self.files:
            path_key = str(path)
            with h5py.File(path, "r") as h5:
                data = h5["data"]
                try:
                    language = json.loads(data.attrs["problem_info"])["language_instruction"]
                except Exception:
                    language = path.stem.replace("_demo", "").replace("_", " ")
                self._language[path_key] = self._tokenize(language)
                for demo_name in sorted(data.keys()):
                    demo = data[demo_name]
                    n = min(len(demo["actions"]), len(demo["obs"]["agentview_rgb"]), len(demo["obs"]["eye_in_hand_rgb"]))
                    if self.action_dim is None:
                        self.action_dim = int(demo["actions"].shape[-1])
                        self.proprio_dim = int(demo["obs"]["ee_states"].shape[-1] + demo["obs"]["gripper_states"].shape[-1])
                    for t in range(n):
                        if t + max(self.horizons) >= n or t + self.action_horizon > n:
                            continue
                        self.index.append((path_key, demo_name, t))
        if not self.index:
            raise RuntimeError("No valid LIBERO windows found for online V-JEPA training")

    def _tokenize(self, language: str) -> tuple[np.ndarray, np.ndarray]:
        if self.tokenizer is None:
            # The Qwen path always supplies a tokenizer; this fallback keeps
            # tiny smoke tests independent from model files.
            ids = np.zeros((self.text_max_length,), dtype=np.int64)
            mask = np.zeros((self.text_max_length,), dtype=np.bool_)
            mask[0] = True
            return ids, mask
        encoded = self.tokenizer(
            language,
            max_length=self.text_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return encoded["input_ids"][0].astype(np.int64), encoded["attention_mask"][0].astype(np.bool_)

    def __len__(self) -> int:
        return len(self.index)

    def _get_handle(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int) -> dict:
        path, demo_name, t = self.index[index]
        demo = self._get_handle(path)["data"][demo_name]
        obs = demo["obs"]
        agent = obs["agentview_rgb"]
        wrist = obs["eye_in_hand_rgb"]
        # Read only the current action window and the short history. Reading
        # complete demonstrations here materialized thousands of rows per
        # sample and made the HDF5 workers a bottleneck.
        action_ds = demo["actions"]
        action_window = np.asarray(action_ds[t : t + self.action_horizon], dtype=np.float32)
        start = max(0, t - self.action_history)
        history_values = np.asarray(action_ds[start:t], dtype=np.float32)
        history = np.zeros((self.action_history, action_window.shape[-1]), dtype=np.float32)
        if history_values.shape[0] > 0:
            history[-history_values.shape[0] :] = history_values
        current = np.stack([agent[t], wrist[t]], axis=0)
        future = {
            horizon: np.stack([agent[t + horizon], wrist[t + horizon]], axis=0)
            for horizon in self.horizons
        }
        ids, mask = self._language[path]
        return {
            "obs_images": current,
            "future_images": future,
            "actions": torch.from_numpy(action_window),
            "action_history": torch.from_numpy(history),
            "text_input_ids": torch.from_numpy(ids.copy()).long(),
            "attention_mask": torch.from_numpy(mask.copy()).bool(),
            "proprio": torch.from_numpy(
                np.concatenate(
                    [
                        np.asarray(obs["ee_states"][t], dtype=np.float32),
                        np.asarray(obs["gripper_states"][t], dtype=np.float32),
                    ],
                    axis=0,
                )
            ),
        }

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
