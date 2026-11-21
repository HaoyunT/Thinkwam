"""Regression fixtures extracted from the published .pt ZIP metadata (no weights).

Construct the actual full Qwen/ViT/P1 architectures on meta, then strict-load
every published key/shape. This checks compatibility, not benchmark success.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from thinkwam_p1.released_checkpoint import (
    release_profile, released_config, load_released_state,
    validate_released_metadata, validate_released_manifest,
)

FIXTURES = Path(__file__).with_name("fixtures")


def checkpoint_for(benchmark):
    profile = release_profile(benchmark)
    schema = json.loads((FIXTURES / f"released_{benchmark}_tensors.json").read_text())
    types = {"FloatStorage": torch.float32, "BFloat16Storage": torch.bfloat16,
             "LongStorage": torch.int64}
    state = {key: torch.empty(value["shape"], device="meta", dtype=types[value["dtype"]])
             for key, value in schema.items()}
    return dict(config=profile["config"], step=profile["step"], model=state)


@pytest.mark.parametrize("benchmark", ["libero", "robotwin", "robocasa24"])
def test_all_published_tensor_names_and_shapes_load_strictly(benchmark):
    from transformers import AutoModelForCausalLM, Qwen2Config
    from accelerate import init_empty_weights
    from thinkwam_p1.models.observation_encoder import FrozenObservationEncoder
    from thinkwam_p1.models.p1 import P1Model
    checkpoint = checkpoint_for(benchmark)
    config = released_config(checkpoint, benchmark)
    config.qwen_attn_implementation = "sdpa"
    qwen_config = Qwen2Config(**json.loads((FIXTURES / "qwen2_5_0_5b_config.json").read_text()))
    # No network, no pretrained values, no 4 GB CPU/GPU allocation.
    def qwen_from_config(*args, **kwargs):
        return AutoModelForCausalLM.from_config(qwen_config, attn_implementation="sdpa")
    # Keep tiny derived buffers on CPU (ViT computes scalar drop-path values).
    with init_empty_weights(), patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=qwen_from_config):
        model = P1Model(config, vjepa_encoder=FrozenObservationEncoder(None))
    removed = load_released_state(model, checkpoint, benchmark)
    assert len(removed) == (8 if benchmark == "libero" else 0)
    assert len(model.state_dict()) == len(checkpoint["model"]) - len(removed)
    broken = dict(checkpoint, model=dict(checkpoint["model"]))
    key = "memory_to_action.weight"
    broken["model"][key] = torch.empty(1, device="meta")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_released_state(model, broken, benchmark)


@pytest.mark.parametrize("benchmark", ["libero", "robotwin", "robocasa24"])
def test_metadata_drift_is_rejected_and_never_relabelled(benchmark):
    checkpoint = checkpoint_for(benchmark)
    original = deepcopy(checkpoint["config"])
    config = released_config(checkpoint, benchmark)
    assert checkpoint["config"] == original
    assert config.qwen_visual_tokens == original["qwen_visual_tokens"]
    for changed in [dict(checkpoint, step=40000),
                    dict(checkpoint, config={**original, "unknown_feature": True}),
                    dict(checkpoint, config={**original, "phase_memory_max_scale": 1.0})]:
        with pytest.raises(ValueError):
            validate_released_metadata(changed, benchmark)


def test_release_manifest_requires_exact_artifact_identity():
    profile = release_profile("libero")
    manifest = {"checkpoint": {key: profile[key] for key in ("step", "sha256", "size_bytes")},
                "model_config": profile["config"],
                "protocol": {"benchmark": "libero", "checkpoint_profile": "released", "paper_recipe_match": False}}
    validate_released_manifest(manifest)
    for key, value in [("sha256", "0" * 64), ("step", 40000), ("size_bytes", 1)]:
        changed = deepcopy(manifest)
        changed["checkpoint"][key] = value
        with pytest.raises(ValueError):
            validate_released_manifest(changed)


def test_robotwin_release_plan_keeps_step_and_distinct_protocol():
    from scripts.robotwin_eval.manifest import build_plan, validate_plan, digest
    profile = release_profile("robotwin")
    identity = dict(checkpoint="/tmp/policy.pt", checkpoint_sha256=profile["sha256"],
                    checkpoint_config_sha256=digest(profile["config"]))
    assets = {name: dict(path=f"/tmp/{name}.json", sha256="a" * 64)
              for name in ("stats", "frame_manifest")}
    plan = build_plan(identity, "/tmp/simulator", assets=assets, checkpoint_profile="released", qwen_path="/tmp/qwen")
    validate_plan(plan)
    assert plan["checkpoint_step"] == 155000
    assert plan["paper_recipe_match"] is False
    assert plan["planned"] == 4000
    changed = dict(plan, checkpoint_step=90000)
    with pytest.raises(ValueError):
        validate_plan(changed)
