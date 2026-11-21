"""Lightweight dataset layout summaries for supported export formats.

These summaries do not replace the benchmark-specific training source guards.
"""

from __future__ import annotations

import json
from pathlib import Path


class DatasetSource:
    name = "base"

    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def validate(self) -> dict:
        return {"dataset": self.name, "root": str(self.root)}


class LiberoSource(DatasetSource):
    name = "libero"

    def validate(self) -> dict:
        files = sorted(self.root.glob("**/*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No LIBERO HDF5 demonstrations under {self.root}")
        return {**super().validate(), "hdf5_files": len(files), "action_dim": 7, "fps": 20}


class LeRobotV3Source(DatasetSource):
    def __init__(self, root: str, name: str):
        super().__init__(root)
        self.name = name

    def validate(self) -> dict:
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            # RoboTwin's published export deliberately keeps clean/random as
            # two independent LeRobot datasets.  Accept the parent directory
            # for source preflight while preserving the per-variant metadata
            # contract used by actual readers.
            variants = [
                child for child in (self.root / "clean", self.root / "random")
                if (child / "meta" / "info.json").is_file()
            ]
            if variants:
                summaries = [LeRobotV3Source(str(child), self.name).validate() for child in variants]
                action_dims = {summary["action_dim"] for summary in summaries}
                fps_values = {summary["fps"] for summary in summaries}
                if len(action_dims) != 1 or len(fps_values) != 1:
                    raise ValueError(f"RoboTwin variants disagree on action/fps contract: {summaries}")
                return {
                    **super().validate(),
                    "variants": [summary["root"] for summary in summaries],
                    "total_episodes": sum(int(summary["total_episodes"] or 0) for summary in summaries),
                    "total_frames": sum(int(summary["total_frames"] or 0) for summary in summaries),
                    "action_dim": next(iter(action_dims)),
                    "fps": next(iter(fps_values)),
                    "codebase_version": summaries[0]["codebase_version"],
                }
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        info = json.loads(info_path.read_text())
        if not (self.root / "data").is_dir() or not (self.root / "videos").is_dir():
            raise FileNotFoundError(f"Expected data/ and videos/ under {self.root}")
        action_shape = info.get("features", {}).get("action", {}).get("shape", [None])
        return {
            **super().validate(),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "action_dim": action_shape[0],
            "fps": info.get("fps"),
            "codebase_version": info.get("codebase_version"),
        }


class RoboCasa24Source(DatasetSource):
    """Validate the lightweight DeVA RoboCasa24 export used by online P0."""

    name = "robocasa24"

    def validate(self) -> dict:
        info_path = self.root / "dataset_info.json"
        action_path = self.root / "action.json"
        videos = sorted((self.root / "videos").glob("demo_*.mp4"))
        affordance = sorted((self.root / "affordance").glob("demo_*.npz"))
        if not info_path.is_file() or not action_path.is_file():
            raise FileNotFoundError(
                f"RoboCasa24 requires dataset_info.json and action.json under {self.root}"
            )
        if not videos:
            raise FileNotFoundError(f"No RoboCasa24 videos under {self.root / 'videos'}")
        return {
            **super().validate(),
            "episodes": len(videos),
            "videos": len(videos),
            "affordance_maps": len(affordance),
            "action_dim": 7,
            "views": ["agentview_left", "agentview_right", "eye_in_hand"],
        }


def make_dataset_source(dataset: str, root: str) -> DatasetSource:
    if dataset == "libero":
        return LiberoSource(root)
    if dataset == "robotwin2.0":
        return LeRobotV3Source(root, dataset)
    if dataset == "robocasa365":
        return LeRobotV3Source(root, dataset)
    if dataset == "robocasa24":
        return RoboCasa24Source(root)
    raise ValueError(f"Unsupported P0 dataset: {dataset}")
