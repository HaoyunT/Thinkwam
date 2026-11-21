"""Read-only indexing helpers for RoboTwin 2.0 LeRobot v3 datasets.

The downloaded dataset keeps ``clean`` and ``random`` as independent
LeRobot datasets. This module only reads Parquet metadata and frame-level
actions/states. The online RGB adapter handles video decoding and cache
verification; the paper training source guard permits only the Clean export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROBOTWIN_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

# RoboTwin's simulator names these cameras head/left/right_camera while the
# LeRobot export uses the cam_* names. Accept both names for affordance files.
ROBOTWIN_AFFORDANCE_CAMERA_ALIASES = {
    "cam_high": ("cam_high", "head_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist_camera"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist_camera"),
}


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("RoboTwin loading requires the `pyarrow` package") from exc
    return pq


def resolve_robotwin_roots(root: str | Path, variant: str = "full") -> tuple[Path, ...]:
    """Resolve a variant or a self-contained LeRobot root.

    ``full`` is represented by two roots because the downloaded export does
    not contain a merged ``meta/info.json``.  The caller can still preserve
    the source variant in each :class:`RobotWinEpisode`.
    """

    root = Path(root).expanduser()
    if variant not in {"clean", "random", "full"}:
        raise ValueError("variant must be clean, random, or full")
    if (root / "meta" / "info.json").is_file():
        actual = root.resolve().name
        if variant != "full" and actual in {"clean", "random"} and actual != variant:
            raise ValueError(f"Requested RoboTwin {variant}, but the self-contained source is {actual}")
        return (root,)
    if variant == "full":
        names = ("clean", "random")
    else:
        names = (variant,)
    roots = tuple(root / name for name in names)
    missing = [str(path) for path in roots if not (path / "meta" / "info.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing RoboTwin LeRobot metadata: " + ", ".join(missing))
    return roots


def episode_split(variant: str, episode_index: int, seed: int, val_fraction: float) -> str:
    """Deterministic episode split shared by cache construction and eval."""

    digest = hashlib.sha1(f"{variant}:{seed}:{episode_index}".encode()).digest()
    return "val" if digest[0] / 255.0 < float(val_fraction) else "train"


@dataclass(frozen=True)
class RobotWinEpisode:
    root: Path
    variant: str
    episode_index: int
    length: int
    data_path: Path
    data_from: int
    data_to: int
    task_index: int
    language: str
    video_paths: tuple[tuple[str, Path], ...]
    video_starts: tuple[tuple[str, float], ...]

    def video_path(self, camera: str) -> Path:
        for name, path in self.video_paths:
            if name == camera:
                return path
        raise KeyError(f"camera {camera!r} is not present in episode {self.episode_index}")

    def video_start(self, camera: str) -> float:
        for name, timestamp in self.video_starts:
            if name == camera:
                return timestamp
        raise KeyError(f"camera {camera!r} is not present in episode {self.episode_index}")


class RobotWinIndex:
    """Metadata index for one or both RoboTwin LeRobot variants."""

    _EPISODE_COLUMNS = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ] + [
        item
        for camera in ROBOTWIN_CAMERAS
        for item in (
            f"videos/{camera}/chunk_index",
            f"videos/{camera}/file_index",
            f"videos/{camera}/from_timestamp",
        )
    ]

    def __init__(self, root: str | Path, *, variant: str = "full") -> None:
        self.root = Path(root).expanduser()
        self.roots = resolve_robotwin_roots(self.root, variant)
        self.episodes: list[RobotWinEpisode] = []
        self._tasks_cache: dict[Path, dict[int, str]] = {}
        for source_root in self.roots:
            source_variant = source_root.name if source_root.name in {"clean", "random"} else variant
            tasks = self._read_tasks(source_root)
            self._tasks_cache[source_root] = tasks
            self._read_episodes(source_root, source_variant, tasks)
        if not self.episodes:
            raise RuntimeError(f"No RoboTwin episodes found under {self.roots}")

    @staticmethod
    def _read_tasks(root: Path) -> dict[int, str]:
        pq = _require_pyarrow()
        path = root / "meta" / "tasks.parquet"
        table = pq.read_table(path, columns=["task_index", "task"])
        values = table.to_pydict()
        return {int(index): str(text) for index, text in zip(values["task_index"], values["task"])}

    def _read_episodes(self, root: Path, variant: str, tasks: dict[int, str]) -> None:
        pq = _require_pyarrow()
        episodes_dir = root / "meta" / "episodes"
        for parquet in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
            table = pq.read_table(parquet, columns=self._EPISODE_COLUMNS).to_pydict()
            count = len(table["episode_index"])
            for row in range(count):
                episode_index = int(table["episode_index"][row])
                task_index = self._task_index(root, episode_index, tasks)
                chunk = int(table["data/chunk_index"][row])
                file_index = int(table["data/file_index"][row])
                data_path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
                video_paths = []
                video_starts = []
                for camera in ROBOTWIN_CAMERAS:
                    vchunk = int(table[f"videos/{camera}/chunk_index"][row])
                    vfile = int(table[f"videos/{camera}/file_index"][row])
                    video_paths.append(
                        (
                            camera,
                            root / "videos" / camera / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4",
                        )
                    )
                    video_starts.append((camera, float(table[f"videos/{camera}/from_timestamp"][row])))
                self.episodes.append(
                    RobotWinEpisode(
                        root=root,
                        variant=variant,
                        episode_index=episode_index,
                        length=int(table["length"][row]),
                        data_path=data_path,
                        data_from=int(table["dataset_from_index"][row]),
                        data_to=int(table["dataset_to_index"][row]),
                        task_index=task_index,
                        language=tasks.get(task_index, ""),
                        video_paths=tuple(video_paths),
                        video_starts=tuple(video_starts),
                    )
                )

    @staticmethod
    def _task_index(root: Path, episode_index: int, tasks: dict[int, str]) -> int:
        """Find the first task index for an episode without reading all frames.

        Episode metadata stores task text alternatives but not a scalar task
        index.  The data Parquet row is authoritative; this method therefore
        returns a sentinel and ``load_episode`` replaces it from the first row.
        The sentinel keeps indexing cheap and preserves the language fallback.
        """

        return -1

    @staticmethod
    def _list_column(values: Iterable[object], *, dtype=np.float32) -> np.ndarray:
        return np.asarray(list(values), dtype=dtype)

    def load_episode(self, episode: RobotWinEpisode) -> dict[str, np.ndarray | int | str]:
        """Load only the frame rows belonging to one episode."""

        pq = _require_pyarrow()
        table = pq.read_table(
            episode.data_path,
            columns=["observation.state", "action", "task_index", "frame_index", "episode_index"],
        )
        values = table.to_pydict()
        episode_indices = np.asarray(values["episode_index"], dtype=np.int64)
        positions = np.flatnonzero(episode_indices == episode.episode_index)
        if len(positions) != episode.length:
            raise ValueError(
                f"episode {episode.variant}/{episode.episode_index} has {len(positions)} rows, "
                f"metadata says {episode.length}"
            )
        task_indices = np.asarray(values["task_index"], dtype=np.int64)[positions]
        task_index = int(task_indices[0]) if len(task_indices) else episode.task_index
        tasks = self._tasks_cache.get(episode.root)
        if tasks is None:
            tasks = self._read_tasks(episode.root)
            self._tasks_cache[episode.root] = tasks
        return {
            "state": self._list_column(np.asarray(values["observation.state"], dtype=object)[positions]),
            "action": self._list_column(np.asarray(values["action"], dtype=object)[positions]),
            "task_index": task_indices,
            "frame_index": np.asarray(values["frame_index"], dtype=np.int64)[positions],
            "task": tasks.get(task_index, episode.language),
            "task_index_scalar": task_index,
        }

    @staticmethod
    def window_starts(
        length: int,
        *,
        action_horizon: int,
        horizons: Iterable[int] = (7, 15, 31),
        stride: int = 1,
    ) -> range:
        maximum = max(int(action_horizon) - 1, *(int(value) for value in horizons))
        limit = int(length) - maximum
        if limit <= 0:
            return range(0)
        return range(0, limit, max(1, int(stride)))


def load_affordance_episode(
    root: str | Path,
    episode: RobotWinEpisode,
    cameras: Iterable[str],
) -> dict[str, np.ndarray] | None:
    """Load exporter output for one episode.

    The exporter writes ``<episode>/<camera>.npz``.  For a full dataset the
    recommended layout is ``<affordance_root>/<variant>/<episode>/``; the
    unqualified ``<episode>/`` form is accepted for a single variant.
    """

    root = Path(root).expanduser()
    short = {camera: camera.rsplit(".", 1)[-1] for camera in cameras}
    candidates = (
        root / episode.variant / str(episode.episode_index),
        root / f"{episode.variant}_{episode.episode_index}",
        root / str(episode.episode_index),
    )
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        return None
    loaded: dict[str, np.ndarray] = {}
    for camera, name in short.items():
        names = ROBOTWIN_AFFORDANCE_CAMERA_ALIASES.get(name, (name, camera))
        path = next(
            (directory / f"{candidate}.npz" for candidate in names if (directory / f"{candidate}.npz").is_file()),
            directory / f"{camera}.npz",
        )
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            key_candidates = tuple(names) + (camera,)
            key = next((candidate for candidate in key_candidates if candidate in data.files), None)
            if key is not None:
                array = data[key]
            elif len(data.files) == 1:
                array = data[data.files[0]]
            else:
                raise ValueError(f"Cannot identify array in {path}: {data.files}")
            loaded[camera] = np.asarray(array, dtype=np.float32)
    return loaded


def dataset_summary(root: str | Path, variant: str = "full") -> dict:
    """Return a small JSON-serialisable preflight summary."""

    index = RobotWinIndex(root, variant=variant)
    episodes = len(index.episodes)
    frames = sum(item.length for item in index.episodes)
    return {
        "root": str(Path(root).expanduser()),
        "variant": variant,
        "episodes": episodes,
        "frames": frames,
        "variants": sorted({item.variant for item in index.episodes}),
        "action_dim": 14,
        "proprio_dim": 14,
        "cameras": list(ROBOTWIN_CAMERAS),
    }
