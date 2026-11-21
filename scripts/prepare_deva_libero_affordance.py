#!/usr/bin/env python3
"""Align DeVA-LIBERO affordance maps with the local LIBERO HDF5 demos.

The DeVA release stores one ``.npz`` per episode and keeps the original
LIBERO task mapping in ``data_mapping.json``. This utility checks the task/demo
mapping and matching frame counts. With ``--write``, it validates every pair
before attaching the affordance arrays to the existing LIBERO demonstrations:

    data/<demo>/obs/affordance   [T, 2, 24, 24] float16

The two views are max-normalized per frame before independent area pooling
from 128x128 to a 24x24 grid. Mapping and frame-count agreement do not establish
action/image content identity. The labels are stored inside the HDF5 files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


VIEW_NAMES = ("agentview", "eye_in_hand")
SUITE_GROUPS = {
    "libero_act_goal": "libero_goal",
    "libero_act_object": "libero_object",
    "libero_act_spatial": "libero_spatial",
}


def area_weights(source_size: int, target_size: int) -> np.ndarray:
    """Return exact 1-D pixel-area weights, shape [target, source]."""
    source_edges = np.arange(source_size + 1, dtype=np.float64)
    target_edges = np.linspace(0.0, float(source_size), target_size + 1)
    weights = np.zeros((target_size, source_size), dtype=np.float32)
    for index in range(target_size):
        left, right = target_edges[index], target_edges[index + 1]
        overlap = np.maximum(0.0, np.minimum(source_edges[1:], right) - np.maximum(source_edges[:-1], left))
        weights[index] = overlap / max(right - left, 1e-12)
    return weights


def area_resize_batch(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Area-pool [T,H,W] without relying on binary NumPy extensions."""
    target_h, target_w = target_hw
    wy = area_weights(array.shape[1], target_h)
    wx = area_weights(array.shape[2], target_w)
    intermediate = np.einsum("yh,thw->tyw", wy, array, optimize=True)
    return np.einsum("xw,tyw->tyx", wx, intermediate, optimize=True)


def numeric_demo_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def parse_range(value: str) -> tuple[int, int]:
    start, end = value.split("-", 1)
    return int(start), int(end)


def prompt_to_file(root: Path, suite: str, prompt: str) -> Path:
    stem = prompt.strip().rstrip(".").replace(" ", "_") + "_demo.hdf5"
    path = root / suite / stem
    if not path.is_file():
        raise FileNotFoundError(f"cannot map DeVA prompt {prompt!r} to {path}")
    return path


def load_actions(path: Path) -> dict[str, list]:
    return json.loads(path.read_text())


def iter_mappings(mapping: dict, libero_root: Path) -> Iterable[dict]:
    # The first ten entries are the LIBERO-10 task files.  Their keys are
    # already the exact HDF5 task stems.
    for task_name, group in mapping.items():
        if task_name in SUITE_GROUPS or "tasks" in group:
            continue
        path = libero_root / "libero_10" / f"{task_name}_demo.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"cannot map DeVA task {task_name!r} to {path}")
        start, end = int(group["start_idx"]), int(group["end_idx"])
        if end - start + 1 != int(group["count"]):
            raise ValueError(f"invalid mapping range for {task_name}: {group}")
        for episode_id in range(start, end + 1):
            yield {"episode_id": episode_id, "path": path, "demo_index": episode_id - start}

    # The remaining groups are the 10 task suites.  Each task entry gives the
    # original 0..49 demo range and the new DeVA episode range.
    for group_name, suite in SUITE_GROUPS.items():
        group = mapping.get(group_name)
        if group is None:
            raise KeyError(f"missing {group_name} in data_mapping.json")
        for task in group["tasks"]:
            new_start, new_end = parse_range(task["new_idx_range"])
            original_start, original_end = parse_range(task["original_idx_range"])
            if new_end - new_start != original_end - original_start:
                raise ValueError(f"range length mismatch for {group_name}: {task}")
            path = prompt_to_file(libero_root, suite, task["prompt"])
            for episode_id in range(new_start, new_end + 1):
                yield {
                    "episode_id": episode_id,
                    "path": path,
                    # DeVA records the source task's global suite range
                    # (0..499), while each local HDF5 task file contains its
                    # own demo_0..demo_49 namespace.
                    "demo_index": (original_start % 50) + episode_id - new_start,
                }


def hdf5_demos(handle: h5py.File) -> list[str]:
    return sorted(handle["data"].keys(), key=numeric_demo_key)


def load_and_pool(npz_path: Path, target_hw: tuple[int, int] = (24, 24)) -> np.ndarray:
    loaded = np.load(npz_path, allow_pickle=False)
    arrays = []
    for view in VIEW_NAMES:
        if view not in loaded:
            raise KeyError(f"{npz_path} does not contain {view}; keys={loaded.files}")
        array = np.asarray(loaded[view], dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"{npz_path}:{view} expected [T,H,W], got {array.shape}")
        # DeVA stores raw, non-negative physical guidance.  Normalize each
        # view/frame before spatial pooling, matching the released loader's
        # per-frame max normalization while keeping view alignment explicit.
        maximum = array.reshape(array.shape[0], -1).max(axis=1)[:, None, None]
        array = np.divide(array, np.maximum(maximum, 1e-12), where=maximum > 0, out=np.zeros_like(array))
        # Exact area pooling keeps the spatial correspondence with the
        # V-JEPA patch grid and avoids binary NumPy ABI differences in the
        # server environment.
        pooled = area_resize_batch(array, target_hw)
        arrays.append(pooled)
    result = np.stack(arrays, axis=1)
    return np.clip(result, 0.0, 1.0).astype(np.float16, copy=False)


def inspect_pair(record: dict, deva_root: Path, action_lengths: dict[str, list]) -> dict:
    episode_id = int(record["episode_id"])
    npz_path = deva_root / "affordance" / f"demo_{episode_id}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with h5py.File(record["path"], "r") as handle:
        demos = hdf5_demos(handle)
        if record["demo_index"] < 0 or record["demo_index"] >= len(demos):
            raise IndexError(f"demo index {record['demo_index']} outside {record['path']}: {len(demos)} demos")
        demo_name = demos[record["demo_index"]]
        demo = handle["data"][demo_name]
        h5_len = min(len(demo["actions"]), len(demo["obs"]["agentview_rgb"]), len(demo["obs"]["eye_in_hand_rgb"]))
    deva_actions = action_lengths.get(f"demo_{episode_id}")
    if deva_actions is None:
        raise KeyError(f"demo_{episode_id} missing in action.json")
    deva_len = len(deva_actions)
    with np.load(npz_path, allow_pickle=False) as loaded:
        label_lengths = {view: int(loaded[view].shape[0]) for view in VIEW_NAMES}
    if h5_len != deva_len or any(length != deva_len for length in label_lengths.values()):
        raise ValueError(
            f"frame mismatch episode={episode_id} file={record['path'].name} {demo_name}: "
            f"hdf5={h5_len}, action={deva_len}, labels={label_lengths}"
        )
    return {
        "episode_id": episode_id,
        "file": str(record["path"]),
        "demo": demo_name,
        "frames": h5_len,
        "npz": str(npz_path),
    }


def write_pair(record: dict, deva_root: Path, manifest_record: dict, overwrite: bool) -> None:
    label = load_and_pool(Path(manifest_record["npz"]))
    with h5py.File(record["path"], "r+") as handle:
        demo = handle["data"][manifest_record["demo"]]
        obs = demo["obs"]
        if "affordance" in obs:
            if not overwrite:
                existing = obs["affordance"]
                if tuple(existing.shape) != tuple(label.shape):
                    raise ValueError(f"existing affordance shape mismatch at {record['path']}:{manifest_record['demo']}")
                return
            del obs["affordance"]
        temporary = "affordance__deva_tmp"
        if temporary in obs:
            del obs[temporary]
        dataset = obs.create_dataset(
            temporary,
            data=label,
            dtype=np.float16,
            chunks=(1, len(VIEW_NAMES), label.shape[-2], label.shape[-1]),
            compression="lzf",
        )
        dataset.attrs["source"] = "mengqz9/deva_libero"
        dataset.attrs["view_order"] = json.dumps(VIEW_NAMES)
        dataset.attrs["normalization"] = "per_view_frame_max"
        dataset.attrs["spatial_grid"] = "V-JEPA 24x24 patch grid"
        # Store as text because this environment's h5py/numpy combination
        # cannot safely create a native integer attribute.
        dataset.attrs["deva_episode"] = str(manifest_record["episode_id"])
        handle.flush()
        obs.move(temporary, "affordance")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deva-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="write labels after validating every pair")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mapping = json.loads((args.deva_root / "data_mapping.json").read_text())
    action_lengths = load_actions(args.deva_root / "action.json")
    records = list(iter_mappings(mapping, args.libero_root))
    if len(records) != 2000:
        raise RuntimeError(f"expected 2000 aligned episodes, got {len(records)}")

    checked = [inspect_pair(record, args.deva_root, action_lengths) for record in records]
    summary = {
        "episodes": len(checked),
        "frames": sum(item["frames"] for item in checked),
        "source": "mengqz9/deva_libero",
        "label_shape_per_frame": [2, 24, 24],
        "view_order": list(VIEW_NAMES),
        "normalization": "per_view_frame_max",
    }
    print(json.dumps(summary, indent=2))
    if not args.write:
        print("dry_run_ok")
        return

    for index, (record, checked_record) in enumerate(zip(records, checked), start=1):
        write_pair(record, args.deva_root, checked_record, args.overwrite)
        if index == 1 or index % 100 == 0 or index == len(records):
            print(f"wrote {index}/{len(records)}", flush=True)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print("write_ok")


if __name__ == "__main__":
    main()
