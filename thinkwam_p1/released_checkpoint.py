"""Explicit, pinned compatibility for the three published inference checkpoints.

Paper-recipe validation remains separate. Never rewrite the stored training
step or silently accept arbitrary historical configurations/partial states.
"""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path


def release_profile(benchmark):
    benchmark = {"libero-plus": "libero", "robotwin2": "robotwin"}.get(benchmark, benchmark)
    data = json.loads((Path(__file__).with_name("release_data") / "profiles.json").read_text())
    if benchmark not in data["profiles"]:
        raise ValueError(f"No released checkpoint profile for {benchmark!r}")
    return data["profiles"][benchmark]


def validate_released_metadata(checkpoint, benchmark):
    """Match the actual embedded metadata before mapping inference fields."""
    from .evaluation_protocol import canonical_json
    profile = release_profile(benchmark)
    if type(checkpoint.get("step")) is not int or checkpoint["step"] != profile["step"]:
        raise ValueError("Released checkpoint step does not match the pinned profile")
    if canonical_json(checkpoint.get("config")) != canonical_json(profile["config"]):
        raise ValueError("Released checkpoint config does not match the pinned profile")
    values = checkpoint["config"]
    # LIBERO retains an old phase head, but its inference contribution is zero.
    # A checkpoint with active phase fusion needs its original implementation.
    if values.get("phase_enabled", True) and values["phase_memory_max_scale"] != 0:
        raise ValueError("Active legacy phase fusion is not supported")
    return profile


def released_config(checkpoint, benchmark):
    from .config import P1Config
    validate_released_metadata(checkpoint, benchmark)
    known = {field.name for field in fields(P1Config)}
    # Full published metadata is retained in manifests. Only the inference
    # dataclass is projected; fields outside it are not executed.
    values = {key: value for key, value in checkpoint["config"].items() if key in known}
    for key in ("jepa_horizons", "jepa_token_counts"):
        values[key] = tuple(values[key])
    return P1Config(**values)


def load_released_state(model, checkpoint, benchmark):
    """Remove only the eight known, inactive LIBERO phase tensors, then strict-load."""
    profile = validate_released_metadata(checkpoint, benchmark)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or len(state) != profile["tensor_count"]:
        raise ValueError("Released checkpoint state has an unexpected tensor count")
    state = dict(state)
    values = profile["config"]
    removed = []
    if values.get("phase_enabled", True):
        width, memory, classes = values["jepa_context_dim"], values["memory_dim"], values["phase_num_classes"]
        shapes = {
            "phase_head.0.weight": (width,), "phase_head.0.bias": (width,),
            "phase_head.1.weight": (classes, width), "phase_head.1.bias": (classes,),
            "phase_to_memory.0.weight": (memory, width), "phase_to_memory.0.bias": (memory,),
            "phase_to_memory.2.weight": (memory, memory), "phase_to_memory.2.bias": (memory,),
        }
        for key, shape in shapes.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError(f"Unexpected inactive phase tensor: {key}")
            del state[key]
            removed.append(key)
    model.load_state_dict(state, strict=True)
    return removed


def validate_released_manifest(manifest):
    """Verify release identity while keeping full benchmark completeness checks."""
    protocol = manifest["protocol"]
    mode = protocol.get("checkpoint_profile", "paper")
    if mode == "paper":
        return
    if mode != "released" or protocol.get("paper_recipe_match") is not False:
        raise ValueError("Invalid checkpoint profile/provenance")
    profile = validate_released_metadata(
        {"step": manifest["checkpoint"]["step"], "config": manifest["model_config"]},
        protocol["benchmark"],
    )
    for key in ("sha256", "size_bytes"):
        if manifest["checkpoint"][key] != profile[key]:
            raise ValueError(f"Released checkpoint {key} differs from the published artifact")
