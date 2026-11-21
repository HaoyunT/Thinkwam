"""CPU-only identity and completeness checks for closed-loop evaluations.

A manifest binds checkpoint *contents*, model configuration, protocol and the
complete case schedule. Completed failures are immutable results too. Retrying
an interrupted case is allowed only within that same manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    """Hash once at initialization, refusing a file changed during the read."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def file_signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _integer(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {label}: {value!r}")


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing evaluation manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != object_sha256(body):
        raise ValueError("Evaluation manifest digest mismatch")
    checkpoint = manifest["checkpoint"]
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("sha256", "")):
        raise ValueError("Invalid checkpoint SHA256")
    _integer(checkpoint["step"], "checkpoint step")
    _integer(checkpoint["size_bytes"], "checkpoint size", 1)
    if type(manifest.get("official")) is not bool:
        raise ValueError("Manifest must declare official as a boolean")
    world_size = manifest["protocol"]["world_size"]
    _integer(world_size, "world_size", 1)
    seen = set()
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("Evaluation needs a nonempty case schedule")
    for case in manifest["cases"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            raise ValueError(f"Invalid case_id: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"Duplicate planned case: {case_id}")
        seen.add(case_id)
        for key in ("env_seed", "policy_seed"):
            _integer(case[key], key, 0, 2**32 - 1)
        _integer(case["rank"], "case rank", 0, world_size - 1)
        _integer(case["max_steps"], "max_steps", 1)
    return manifest


def make_manifest(checkpoint_path, checkpoint_step, model_config, protocol, cases, *, official=True):
    checkpoint_path = Path(checkpoint_path)
    manifest = {
        "schema_version": 1, "official": official,
        "checkpoint": {"sha256": file_sha256(checkpoint_path),
                       "size_bytes": checkpoint_path.stat().st_size, "step": checkpoint_step},
        "model_config": model_config, "protocol": protocol, "cases": cases,
    }
    # Normalize dataclass tuples to their JSON representation before comparing.
    manifest = json.loads(canonical_json(manifest))
    manifest["manifest_id"] = object_sha256(manifest)
    return validate_manifest(manifest)


def atomic_json(path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bind_manifest(root, manifest):
    """Atomically bind a directory; concurrent ranks must agree before rollout."""
    validate_manifest(manifest)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "manifest.json"
    if target.exists():
        previous = validate_manifest(json.loads(target.read_text()))
        if canonical_json(previous) != canonical_json(manifest):
            raise ValueError("Output directory belongs to a different checkpoint/protocol/config; use a new directory")
        return
    if any(root.rglob("result.json")) or any(root.rglob("summary.json")):
        if target.exists():
            # Another rank may have published the matching manifest while we
            # checked for legacy files.
            return bind_manifest(root, manifest)
        raise ValueError("Existing evaluation results have no verified manifest; use a new directory")
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        try:
            os.link(temporary, target)  # publish a complete file without replacing another rank's manifest
        except FileExistsError:
            previous = validate_manifest(json.loads(target.read_text()))
            if canonical_json(previous) != canonical_json(manifest):
                raise ValueError("Concurrent evaluators disagree on checkpoint/protocol/config")
    finally:
        os.unlink(temporary)


def result_identity(manifest):
    return {"manifest_id": manifest["manifest_id"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint_step": manifest["checkpoint"]["step"]}


def validate_case_result(row, case, manifest):
    if not isinstance(row, dict):
        raise ValueError("Result must be an object")
    for key, expected in {**case, **result_identity(manifest), "status": "completed"}.items():
        if key not in row or canonical_json(row[key]) != canonical_json(expected):
            raise ValueError(f"Case {case['case_id']} has mismatched {key}")
    if type(row.get("success")) is not bool:
        raise ValueError(f"Case {case['case_id']} success must be boolean")
    _integer(row.get("steps"), "executed steps", 0, case["max_steps"])
    if manifest["protocol"]["benchmark"] == "robocasa24":
        if type(row.get("initial_success")) is not bool:
            raise ValueError("initial_success must be boolean")
        if row["initial_success"] and (not row["success"] or row["steps"] != 0):
            raise ValueError("Initially successful RoboCasa case must finish successfully at step zero")
    return row


def validate_rows(rows, cases, manifest, *, complete=True):
    expected = {case["case_id"]: case for case in cases}
    found = {}
    for row in rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if case_id not in expected:
            raise ValueError(f"Unknown or unassigned result case: {case_id!r}")
        if case_id in found:
            raise ValueError(f"Duplicate result case: {case_id}")
        found[case_id] = validate_case_result(row, expected[case_id], manifest)
    if complete and set(found) != set(expected):
        raise ValueError(f"Incomplete evaluation: {len(set(expected) - set(found))} cases missing")
    return [found[case["case_id"]] for case in cases if case["case_id"] in found]


def read_case_results(root, cases, manifest, *, complete=False):
    rows = []
    for path in sorted(Path(root).glob("*/result.json")):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("case_id") != path.parent.name:
            raise ValueError(f"Result path and case_id disagree: {path}")
        rows.append(row)
    return validate_rows(rows, cases, manifest, complete=complete)


# These fields change locations or execution/logging, not the trained objective,
# data protocol, architecture, optimizer schedule, or effective batch size.
OPERATIONAL_CONFIG_FIELDS = frozenset({
    "vjepa_checkpoint", "qwen_path", "dataset_root", "output_dir", "p0_checkpoint",
    "robotwin_frame_cache", "robotwin_manifest", "robotwin_stats_path", "robocasa_split_manifest",
    "num_workers", "dataloader_prefetch_factor", "dataloader_persistent_workers",
    "dataloader_pin_memory", "dataloader_worker_threads", "compile_vjepa", "vjepa_compile_mode",
    "compile_action_dit", "action_dit_compile_mode", "qwen_attn_implementation",
    "gradient_checkpointing", "checkpoint_save_limit", "eval_every", "eval_batches",
    "tensorboard", "tensorboard_histogram_every", "open_loop_eval", "save_best_checkpoint",
})


def validate_checkpoint_metadata(checkpoint, reference_config, *, official=True, final_step=40000):
    """Pure-data validation, also used by CPU tests without loading model code."""
    values = checkpoint.get("config")
    if not isinstance(values, dict) or set(values) != set(reference_config):
        found = set(values) if isinstance(values, dict) else set()
        raise ValueError("Checkpoint needs the complete current embedded config; "
                         f"missing={sorted(set(reference_config) - found)}, "
                         f"unknown={sorted(found - set(reference_config))}")
    _integer(checkpoint.get("step"), "checkpoint step")
    if official and checkpoint["step"] != final_step:
        raise ValueError(f"Formal evaluation requires the final step-{final_step} checkpoint")
    if official:
        mismatched = [key for key, expected in reference_config.items()
                      if key not in OPERATIONAL_CONFIG_FIELDS
                      and canonical_json(values[key]) != canonical_json(expected)]
        if mismatched:
            raise ValueError(f"Checkpoint differs from the paper training recipe: {sorted(mismatched)}")
    return values


def validate_paper_checkpoint_metadata(checkpoint, benchmark, *, debug=False):
    """Load the current config schema only in the model/evaluation process."""
    from dataclasses import asdict
    import yaml
    from thinkwam_p1.config import P1Config

    recipe = {"libero": "libero_p1.yaml", "libero-plus": "libero_p1.yaml",
              "robocasa24": "robocasa24_p1.yaml"}.get(benchmark)
    if recipe is None:
        raise ValueError(f"Unsupported paper checkpoint benchmark: {benchmark}")
    path = Path(__file__).resolve().parents[1] / "configs" / "paper" / recipe
    reference_values = yaml.safe_load(path.read_text())
    for key in ("jepa_horizons", "jepa_token_counts"):
        reference_values[key] = tuple(reference_values[key])
    reference = asdict(P1Config(**reference_values))
    values = dict(validate_checkpoint_metadata(checkpoint, reference, official=not debug))
    for key in ("jepa_horizons", "jepa_token_counts"):
        values[key] = tuple(values[key])
    # __post_init__ validates domains even for a debug-only checkpoint.
    return P1Config(**values)
