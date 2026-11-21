"""Require a complete final paper-P0 checkpoint before starting matching P1."""

from collections.abc import Mapping
from pathlib import Path

import torch
import yaml

from thinkwam_p0.config import P0Config
from .checkpoint import _RUNTIME_FIELDS, _normalize


_SHARED_MODULES = frozenset({
    "qwen", "memory", "predictor", "qwen_action_to_context", "memory_to_action", "action_dit",
})
_ARCHITECTURE_FIELDS = (
    "vjepa_checkpoint", "vjepa_dim", "vjepa_image_size", "vjepa_patch_size", "vjepa_views",
    "qwen_path", "qwen_dim", "qwen_max_length", "qwen_action_tokens", "qwen_lora",
    "qwen_lora_rank", "qwen_lora_alpha", "qwen_lora_dropout", "qwen_lora_target_modules",
    "qwen_visual_tokens", "qwen_hidden_only", "visual_token_count", "text_max_length",
    "action_context_dim", "conditioning_mlp_hidden", "jepa_horizons", "jepa_token_counts",
    "jepa_context_dim", "jepa_observation_tokens", "jepa_layers", "jepa_heads",
    "memory_dim", "memory_slots", "memory_observation_tokens", "memory_action_history",
    "memory_layers", "memory_heads", "action_dim", "proprio_dim", "action_horizon",
    "action_dit_hidden", "action_dit_layers", "action_dit_heads", "action_dit_inner_dim",
    "action_dit_future_tokens",
)


def paper_p0_config(dataset: str) -> P0Config:
    if dataset not in {"libero", "robocasa24"}:
        raise ValueError("Only LIBERO and RoboCasa24 P1 use a P0 warm start")
    path = Path(__file__).resolve().parents[2] / "configs" / "paper" / f"{dataset}_p0.yaml"
    values = yaml.safe_load(path.read_text())
    for key in ("jepa_horizons", "jepa_token_counts"):
        values[key] = tuple(values[key])
    return P0Config(**values)


def validate_p0_warm_start(state, expected: P0Config) -> None:
    """Validate saved provenance/recipe metadata, without changing weights."""
    if not isinstance(state, Mapping):
        raise ValueError("P0 warm start requires a checkpoint mapping")
    if not isinstance(state.get("config"), Mapping) or not isinstance(state.get("model"), Mapping):
        raise ValueError("P0 warm start requires saved config and model state")
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected.max_steps:
        raise ValueError(f"P0 warm start requires the final {expected.max_steps}-step checkpoint")
    expected_values = vars(expected)
    saved = state["config"]
    changed = sorted(
        key for key in (set(saved) | set(expected_values)) - _RUNTIME_FIELDS
        if key not in saved or key not in expected_values
        or _normalize(saved[key]) != _normalize(expected_values[key])
    )
    if changed:
        raise ValueError("P0 warm-start recipe differs from the matching paper P0: " + ", ".join(changed))


def restore_shared_p0_weights(model, state) -> dict[str, int]:
    """Load all shared modules atomically after checking names and shapes.

    Frozen V-JEPA weights are loaded separately from the configured release;
    P0's checkpoint writer omits them. P1-only modules retain initialization.
    """
    source = state["model"]
    target = model.state_dict()
    required = {key for key in target if key.split(".", 1)[0] in _SHARED_MODULES}
    if not required:
        raise ValueError("P1 model has no shared P0 parameters")
    missing = sorted(required - set(source))
    unexpected = sorted(key for key in source if key not in required and not key.startswith("vjepa."))
    mismatched = sorted(key for key in required & set(source)
                        if not isinstance(source[key], torch.Tensor)
                        or tuple(source[key].shape) != tuple(target[key].shape))
    if missing or unexpected or mismatched:
        raise ValueError(f"Incomplete P0 warm start: missing={missing[:5]}, "
                         f"unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}")
    model.load_state_dict({key: source[key] for key in required}, strict=False)
    return {"loaded": len(required), "new_p1_tensors": sum(
        key not in required and not key.startswith("vjepa.") for key in target
    )}


def load_p0_warm_start(model, checkpoint_path: str, config) -> dict[str, int]:
    expected = paper_p0_config(config.dataset)
    incompatible = [key for key in _ARCHITECTURE_FIELDS
                    if _normalize(getattr(config, key)) != _normalize(getattr(expected, key))]
    if incompatible:
        raise ValueError("P1 shared architecture differs from its paper P0: " + ", ".join(incompatible))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_p0_warm_start(state, expected)
    return restore_shared_p0_weights(model, state)
