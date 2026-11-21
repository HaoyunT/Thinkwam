"""Standard LIBERO P1 samples with camera-space affordance labels."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from thinkwam_p0.data.libero_online import OnlineLiberoDataset


class P1LiberoDataset(Dataset):
    """Read each standard LIBERO window once and attach its contact heatmap."""

    def __init__(self, *args, affordance_key: str = "affordance", **kwargs):
        self.base = OnlineLiberoDataset(*args, **kwargs)
        self.affordance_key = affordance_key

    def __len__(self):
        return len(self.base)

    def _read_label(self, path: str, demo_name: str, frame: int, key: str, *, required: bool):
        demo = self.base._get_handle(path)["data"][demo_name]
        candidates = [demo.get(key), demo.get("obs", {}).get(key)]
        dataset = next((item for item in candidates if item is not None), None)
        if dataset is None:
            if required:
                raise KeyError(
                    f"No `{key}` dataset in {path}:{demo_name}; prepare P1 labels before training"
                )
            return None
        if len(dataset) <= frame:
            raise IndexError(f"{key} dataset is shorter than frame {frame}: {path}:{demo_name}")
        return torch.from_numpy(np.asarray(dataset[frame], dtype=np.float32))

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        path, demo_name, frame = self.base.index[index]
        sample["affordance_targets"] = self._read_label(
            path, demo_name, frame, self.affordance_key, required=True
        )
        return sample
