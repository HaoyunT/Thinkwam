"""RoboTwin simulator-side affordance capture.

The existing RoboTwin collector stores RGB and robot state in per-frame pickle
files.  This adapter records the missing supervision at the same frame index:
the physical gripper contact points, camera calibration, and image shape.  It
does not guess a target object or use asset functional points.  The caller
must pass the task-specific actor names whose contact should be supervised.

Typical use inside RoboTwin's ``_take_picture`` (after ``get_obs()`` has
updated the render) is::

    record = capture_task_frame(
        task, frame_index=task.FRAME_IDX,
        actor_names=["048_stapler"],
    )
    writer.append(record)

The resulting JSONL is consumed by
``scripts/export_robotwin_affordance.py``.  Keeping capture and rasterisation
separate makes the labels reproducible and avoids importing SAPIEN into the
training environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .robotwin_affordance import make_frame_record


def capture_task_frame(
    task,
    frame_index: int,
    actor_names: Sequence[str],
    *,
    camera_names: Sequence[str] | None = None,
) -> dict:
    """Capture one serialisable affordance record from a live RoboTwin task.

    ``task.get_obs()`` should normally be called immediately before this
    function by the collector.  Calling ``get_rgb`` here only reads the
    already-rendered images to obtain exact ``height``/``width``; RGB itself
    is not copied into the JSONL.
    """

    names = [str(name) for name in actor_names if str(name)]
    if not names:
        raise ValueError("actor_names must contain the target RoboTwin actor names")
    if not hasattr(task, "get_gripper_actor_contact_position"):
        raise TypeError("task does not expose RoboTwin get_gripper_actor_contact_position()")
    if not hasattr(task, "cameras"):
        raise TypeError("task does not expose RoboTwin cameras")

    camera_configs = task.cameras.get_config()
    rgb_by_camera = task.cameras.get_rgb()
    selected = set(camera_names) if camera_names is not None else None
    configs: dict[str, dict] = {}
    for camera_name, calibration in camera_configs.items():
        if selected is not None and camera_name not in selected:
            continue
        if camera_name not in rgb_by_camera:
            raise KeyError(f"camera {camera_name!r} has calibration but no RGB frame")
        rgb = np.asarray(rgb_by_camera[camera_name]["rgb"])
        if rgb.ndim < 2:
            raise ValueError(f"camera {camera_name!r} RGB has invalid shape {rgb.shape}")
        configs[camera_name] = {
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "intrinsic_cv": calibration["intrinsic_cv"],
            "extrinsic_cv": calibration["extrinsic_cv"],
        }
    if not configs:
        raise ValueError("no selected RoboTwin cameras were available")

    contacts: list[list[float]] = []
    for actor_name in names:
        points = task.get_gripper_actor_contact_position(actor_name) or []
        for point in points:
            value = np.asarray(point, dtype=np.float64).reshape(-1)
            if value.size < 3 or not np.isfinite(value[:3]).all():
                continue
            contacts.append(value[:3].tolist())

    record = make_frame_record(int(frame_index), configs, contacts)
    record["actor_names"] = names
    record["contact_count"] = len(contacts)
    return record


class RobotTwinAffordanceWriter:
    """Append frame records with monotonic indices and durable JSONL writes."""

    def __init__(self, path: str | Path, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush = bool(flush)
        self._last_index: int | None = None
        if self.path.exists() and self.path.stat().st_size:
            # Recover the last complete frame so a restarted collector cannot
            # silently append a duplicate index.
            with self.path.open("rb") as existing:
                existing.seek(max(0, existing.seek(0, 2) - 65536))
                tail = existing.read().splitlines()
            for raw in reversed(tail):
                if raw.strip():
                    self._last_index = int(json.loads(raw)["frame_index"])
                    break
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: Mapping[str, object]) -> None:
        index = int(record["frame_index"])
        if self._last_index is not None and index <= self._last_index:
            raise ValueError(f"frame_index must increase strictly: {index} after {self._last_index}")
        json.dump(dict(record), self._handle, ensure_ascii=False, separators=(",", ":"))
        self._handle.write("\n")
        if self.flush:
            self._handle.flush()
        self._last_index = index

    def append_task_frame(self, task, frame_index: int, actor_names: Sequence[str], *, camera_names=None) -> dict:
        record = capture_task_frame(task, frame_index, actor_names, camera_names=camera_names)
        self.append(record)
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
