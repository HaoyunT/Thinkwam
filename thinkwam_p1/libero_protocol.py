"""Validate the standard 40-task, 2,000-demonstration LIBERO training input.

The allowlist, metadata and self-contained-file checks reject renamed tasks
and indirect external inputs. They do not attest historical checkpoint
lineage or prove that every payload byte matches an official release.
"""

import json
from pathlib import Path, PurePosixPath

import h5py

from .libero_tasks import STANDARD_SUITES, STANDARD_TASKS, standard_task_language


def _json_attribute(group, name: str, path: Path) -> dict:
    try:
        value = json.loads(group.attrs[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {name} metadata in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} metadata must be a mapping in {path}")
    return value


def _text_attribute(group, name: str, path: Path) -> str:
    if name not in group.attrs:
        raise ValueError(f"Missing {name} metadata in {path}")
    value = group.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} metadata must be text in {path}")
    return value


def _check_self_contained(handle: h5py.File, path: Path) -> None:
    """Check link metadata before dereferencing any externally backed object."""
    seen = {h5py.h5o.get_info(handle["/"].id).addr}

    def visit(group):
        for name in group:
            link = group.get(name, getlink=True)
            location = group.name.rstrip("/") + "/" + name
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"HDF5 external/soft links are not allowed: {path}:{location}")
            obj = group[name]
            address = h5py.h5o.get_info(obj.id).addr
            if address in seen:
                raise ValueError(f"Aliased HDF5 objects are not allowed: {path}:{location}")
            seen.add(address)
            if isinstance(obj, h5py.Group):
                visit(obj)
            elif obj.is_virtual or obj.external:
                raise ValueError(f"HDF5 virtual/external storage is not allowed: {path}:{location}")

    visit(handle["/"])


def _check_task_file(path: Path, suite: str, task: str) -> None:
    with h5py.File(path, "r") as handle:
        _check_self_contained(handle, path)
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError(f"Missing LIBERO data group in {path}")
        data = handle["data"]
        if set(data) != {f"demo_{index}" for index in range(50)}:
            raise ValueError(f"Expected exactly demo_0 through demo_49 in {path}")
        if any(not isinstance(data[name], h5py.Group) for name in data):
            raise ValueError(f"Every LIBERO demonstration must be a group in {path}")
        if data.attrs.get("num_demos") != 50:
            raise ValueError(f"Expected num_demos=50 metadata in {path}")
        bddl = PurePosixPath(_text_attribute(data, "bddl_file_name", path).replace("\\", "/"))
        if bddl.parts[-2:] != (suite, task + ".bddl"):
            raise ValueError(f"BDDL task metadata does not match {suite}/{task} in {path}")
        info = _json_attribute(data, "problem_info", path)
        if info.get("language_instruction") != standard_task_language(task):
            raise ValueError(f"Task language metadata does not match {suite}/{task} in {path}")
        if info.get("domain_name") != "robosuite":
            raise ValueError(f"Expected robosuite domain metadata in {path}")
        environment = _text_attribute(data, "env_name", path)
        env_args = _json_attribute(data, "env_args", path)
        if (info.get("problem_name") != environment.lower()
                or env_args.get("problem_name") != info.get("problem_name")
                or env_args.get("env_name") != environment):
            raise ValueError(f"Inconsistent LIBERO environment metadata in {path}")
        # bddl_file_name is the canonical released task identity. env_args'
        # bddl_file retains the original collection-time names in the release.


def validate_standard_libero_source(config) -> dict[str, int]:
    """Require the exact official task set, 50 demos per task and local data.

    JSON manifests and other non-HDF5 sidecars are allowed. Image augmentation
    and added affordance labels do not change the source-task identity.
    """
    if not config.online_vjepa:
        raise ValueError("Standard LIBERO training requires online_vjepa=true and raw demonstrations")
    root = Path(config.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO dataset root not found: {root}")
    for suite in STANDARD_SUITES:
        directory = root / suite
        if directory.is_symlink():
            raise ValueError(f"LIBERO task directories must not be symlinks: {directory}")
    expected = {Path(suite) / (task + "_demo.hdf5"): (suite, task)
                for suite, tasks in STANDARD_TASKS.items() for task in tasks}
    paths = sorted(root.rglob("*.hdf5"))
    actual = {path.relative_to(root) for path in paths}
    if actual != set(expected):
        missing = sorted(str(path) for path in set(expected) - actual)
        unexpected = sorted(str(path) for path in actual - set(expected))
        raise ValueError(f"Expected the exact 40 standard LIBERO task files; "
                         f"missing={missing[:3]}, unexpected={unexpected[:3]}")
    seen_files = set()
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"LIBERO task files must not use symlinks or external paths: {path}")
        if not path.is_file():
            raise ValueError(f"LIBERO task path is not a regular file: {path}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if stat.st_nlink != 1 or identity in seen_files:
            raise ValueError(f"Duplicate or hard-linked LIBERO task file: {path}")
        seen_files.add(identity)
        suite, task = expected[path.relative_to(root)]
        _check_task_file(path, suite, task)
    return {suite: len(tasks) for suite, tasks in STANDARD_TASKS.items()}
