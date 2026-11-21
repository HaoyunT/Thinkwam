"""Shared source and episode-split boundaries for the RoboCasa24 paper recipes."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np


VIEW_NAMES = ("agentview_left", "agentview_right", "eye_in_hand")
EPISODES = frozenset(f"demo_{index}" for index in range(1199))
TASKS = frozenset({
    "kitchen_coffee_CoffeePressButton", "kitchen_coffee_CoffeeServeMug", "kitchen_coffee_CoffeeSetupMug",
    "kitchen_doors_CloseDoubleDoor", "kitchen_doors_CloseSingleDoor",
    "kitchen_doors_OpenDoubleDoor", "kitchen_doors_OpenSingleDoor",
    "kitchen_drawer_CloseDrawer", "kitchen_drawer_OpenDrawer",
    "kitchen_microwave_TurnOffMicrowave", "kitchen_microwave_TurnOnMicrowave",
    "kitchen_pnp_PnPCabToCounter", "kitchen_pnp_PnPCounterToCab",
    "kitchen_pnp_PnPCounterToMicrowave", "kitchen_pnp_PnPCounterToSink", "kitchen_pnp_PnPCounterToStove",
    "kitchen_pnp_PnPMicrowaveToCounter", "kitchen_pnp_PnPSinkToCounter", "kitchen_pnp_PnPStoveToCounter",
    "kitchen_sink_TurnOffSinkFaucet", "kitchen_sink_TurnOnSinkFaucet", "kitchen_sink_TurnSinkSpout",
    "kitchen_stove_TurnOffStove", "kitchen_stove_TurnOnStove",
})


def source_path(root, relative):
    """Resolve source-owned assets; neither traversal nor external symlinks are allowed."""
    root = Path(root).expanduser().resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative.as_posix():
        raise ValueError(f"RoboCasa source path must be relative and contained: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"RoboCasa source path escapes the dataset root: {relative}")
    return path


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate RoboCasa metadata key: {key}")
        value[key] = item
    return value


def read_source_json(root, relative):
    return json.loads(source_path(root, relative).read_text(), object_pairs_hook=_unique_object)


def validate_action_statistics(stats, *, require_protocol=False):
    if not isinstance(stats, dict):
        raise ValueError("RoboCasa normalization statistics must be an object")
    for name, expected in {"norm_method": "minmax_symmetric", "action_dim": 7,
                           "use_delta_actions": False}.items():
        if require_protocol or name in stats:
            if name not in stats or type(stats[name]) is not type(expected) or stats[name] != expected:
                raise ValueError(f"Invalid RoboCasa normalization {name}: expected {expected!r}")
    low = np.asarray(stats.get("action_low"), dtype=np.float32)
    high = np.asarray(stats.get("action_high"), dtype=np.float32)
    if low.shape != (7,) or high.shape != (7,):
        raise ValueError("RoboCasa24 action bounds must be seven-dimensional")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not (low < high).all():
        raise ValueError("RoboCasa24 action bounds must be finite with low < high in every dimension")
    # The export's action_horizon=33 is metadata, not the model's chunk length.
    return low, high


def validate_affordance_header(path, length, *, spatial_shape=(224, 224)):
    """Read only NPY headers at preflight; the reader validates all values on use."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = Counter(archive.namelist())
            for view in VIEW_NAMES:
                if names[f"{view}.npy"] != 1:
                    raise ValueError(f"{path} requires exactly one affordance array for {view}")
                with archive.open(f"{view}.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported affordance NPY version in {path}: {version}")
                if shape != (length, *spatial_shape) or dtype.kind != "f":
                    raise ValueError(f"Invalid affordance shape/dtype in {path}/{view}: {shape}, {dtype}")
    except (zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"Invalid or truncated affordance archive: {path}") from exc


def validate_robocasa_source(config):
    """Validate the complete paper export before either P0 or P1 starts."""
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    if getattr(config, "dataset", None) != "robocasa24":
        raise ValueError("Expected the RoboCasa24 data protocol")
    if getattr(config, "online_vjepa", None) is not True:
        raise ValueError("Paper RoboCasa24 requires online_vjepa=true and the complete raw source")
    root = Path(config.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    is_p1 = hasattr(config, "robocasa24_require_affordance")
    require_affordance = getattr(config, "robocasa24_require_affordance", False)
    if is_p1 and require_affordance is not True:
        raise ValueError("Paper RoboCasa24 P1 requires affordance labels")

    info = read_source_json(root, "dataset_info.json")
    if not isinstance(info, dict) or info.get("views") != list(VIEW_NAMES) or info.get("tile") != [[VIEW_NAMES[0], VIEW_NAMES[1]], [VIEW_NAMES[2], None]]:
        raise ValueError("RoboCasa24 requires the released three-view order and 2x2 tile")
    actions = read_source_json(root, "action.json")
    if not isinstance(actions, dict) or set(actions) != EPISODES:
        raise ValueError("Paper RoboCasa24 requires exactly canonical demo_0 through demo_1198")
    low, high = validate_action_statistics(read_source_json(root, "norm_stats.json"), require_protocol=True)
    mapping = read_source_json(root, "data_mapping.json")
    if not isinstance(mapping, dict) or set(mapping) != TASKS:
        raise ValueError("RoboCasa24 mapping must contain exactly the 24 released tasks")
    mapped = set()
    for task, record in mapping.items():
        if not isinstance(record, dict) or any(type(record.get(key)) is not int for key in ("start_idx", "end_idx", "count")):
            raise ValueError(f"Invalid task mapping record: {task}")
        start, end = record["start_idx"], record["end_idx"]
        if not 0 <= start <= end < 1199 or record["count"] != end - start + 1:
            raise ValueError(f"Invalid task mapping range/count: {task}")
        covered = set(range(start, end + 1))
        if mapped & covered:
            raise ValueError(f"Overlapping RoboCasa24 task mapping: {task}")
        mapped.update(covered)
    if mapped != set(range(1199)):
        raise ValueError("RoboCasa24 task mapping does not cover all 1199 episodes")

    folders = [("videos", ".mp4"), ("metas", ".txt")]
    if require_affordance:
        folders.append(("affordance", ".npz"))
    for folder, extension in folders:
        directory = source_path(root, folder)
        found = {path.stem for path in directory.glob(f"*{extension}")}
        if found != EPISODES:
            raise ValueError(f"RoboCasa24 {folder} episode set differs from the canonical action episode set")
        for episode in EPISODES:
            path = source_path(root, f"{folder}/{episode}{extension}")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty RoboCasa24 asset: {path}")

    lengths = {}
    for episode, values in actions.items():
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 7 or not np.isfinite(array).all():
            raise ValueError(f"Invalid action labels for {episode}: expected nonempty finite [T,7]")
        if ((array < low) | (array > high)).any():
            raise ValueError(f"{episode} action labels exceed the released normalization bounds")
        lengths[episode] = len(array)
        if not source_path(root, f"metas/{episode}.txt").read_text().strip():
            raise ValueError(f"RoboCasa24 instruction is empty: {episode}")
        if require_affordance:
            validate_affordance_header(source_path(root, f"affordance/{episode}.npz"), len(array))

    required_manifest = source_path(root, ".required_files_manifest.json")
    if required_manifest.is_file():
        sizes = read_source_json(root, ".required_files_manifest.json")
        if not isinstance(sizes, dict):
            raise ValueError("RoboCasa required-files manifest must be an object")
        for relative, spec in sizes.items():
            path = source_path(root, relative)
            if relative.startswith("affordance/") and not require_affordance:
                continue
            if not isinstance(spec, dict) or type(spec.get("size")) is not int or spec["size"] < 1:
                raise ValueError(f"Invalid required-file size record: {relative}")
            if not path.is_file() or path.stat().st_size != spec["size"]:
                raise ValueError(f"Missing or incorrect-sized RoboCasa24 file: {relative}")
    return {"root": str(root), "episodes": len(actions), "tasks": len(mapping),
            "windows": sum(lengths.values()), "require_affordance": bool(require_affordance),
            "views": list(VIEW_NAMES), "action_dim": 7}


def validate_robocasa_split(dataset, train_indices, val_indices, config):
    """Require both stages to use exactly the seed-7 1080/119 episode holdout."""
    if type(config.seed) is not int or config.seed != 7 or config.eval_fraction != 0.1:
        raise ValueError("Paper RoboCasa24 split requires seed=7 and eval_fraction=0.1")
    if set(dataset.episode_lengths) != EPISODES:
        raise ValueError("RoboCasa24 split requires all 1199 canonical episodes")
    lengths = dataset.episode_lengths
    if any(type(length) is not int or length < 1 for length in lengths.values()):
        raise ValueError("RoboCasa24 split requires nonempty episode lengths")
    index = dataset.index
    if len(index) != sum(lengths.values()) or len(set(index)) != len(index):
        raise ValueError("RoboCasa24 window index omits or duplicates source frames")
    for episode, frame in index:
        if episode not in lengths or type(frame) is not int or not 0 <= frame < lengths[episode]:
            raise ValueError("Invalid RoboCasa24 source window")
    for name, selected in (("train", train_indices), ("validation", val_indices)):
        if any(type(value) is not int or not 0 <= value < len(index) for value in selected):
            raise ValueError(f"Invalid {name} window index")
        if len(set(selected)) != len(selected):
            raise ValueError(f"Duplicate {name} window indices")
    from thinkwam_p0.data.splits import episode_holdout_indices
    expected_train, expected_val = episode_holdout_indices(index, eval_fraction=0.1, seed=7)
    if set(train_indices) != set(expected_train) or set(val_indices) != set(expected_val):
        raise ValueError("RoboCasa24 split differs from the canonical seed-7 P0/P1 episode holdout")
    train_episodes = sorted({index[i][0] for i in expected_train})
    val_episodes = sorted({index[i][0] for i in expected_val})
    if (len(train_episodes), len(val_episodes)) != (1080, 119):
        raise ValueError("RoboCasa24 split must contain exactly 1080/119 disjoint episodes")
    digest = lambda episodes: hashlib.sha256("\n".join(episodes).encode()).hexdigest()
    return {"train_episodes": train_episodes, "val_episodes": val_episodes,
            "train_windows": len(expected_train), "val_windows": len(expected_val),
            "train_sha256": digest(train_episodes), "val_sha256": digest(val_episodes)}
