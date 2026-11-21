#!/usr/bin/env python3
"""Compare online preprocessing with an explicitly supplied LIBERO feature cache.

Requires a non-empty config.feature_cache and the known rank0 first-shard
layout. The online paper recipes leave feature_cache empty, so this diagnostic
is not a default training/evaluation preflight or proof of checkpoint lineage.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinkwam_p1.libero_eval_inputs import encode_obs, ensure_libero_config, get_proprio, load_config, training_contract
from thinkwam_p0.data.libero_inputs import prepare_raw_action, uniform_pool
from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    contract = training_contract(config)
    ensure_libero_config()
    from robosuite.utils.transform_utils import axisangle2quat

    root = Path(config.dataset_root)
    cache = Path(config.feature_cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    first = manifest["records"][0]
    # The merged rank0 shard begins at the first sorted file / demo / timestep.
    if not first["file"].startswith("rank0_shard_00000"):
        raise ValueError("Audit requires the known rank0 first-shard provenance")
    with np.load(cache / first["file"]) as data:
        stored = {key: data[key][0] for key in data.files}
    hdf5 = sorted(root.rglob("*.hdf5"))[0]
    with h5py.File(hdf5, "r") as data:
        demo_name = sorted(data["data"].keys())[0]
        demo = data["data"][demo_name]
        ee = demo["obs/ee_states"][0]
        obs = {
            "agentview_image": demo["obs/agentview_rgb"][0],
            "robot0_eye_in_hand_image": demo["obs/eye_in_hand_rgb"][0],
            "robot0_eef_pos": ee[:3],
            "robot0_eef_quat": axisangle2quat(ee[3:].copy()),
            "robot0_gripper_qpos": demo["obs/gripper_states"][0],
        }
        actions = demo["actions"][:config.action_horizon].astype(np.float32)
        language = json.loads(data["data"].attrs["problem_info"])["language_instruction"]
    np.testing.assert_array_equal(actions, stored["actions"])
    np.testing.assert_array_equal(stored["action_history"], np.zeros_like(stored["action_history"]))
    proprio = get_proprio(obs).numpy()
    np.testing.assert_allclose(proprio, stored["proprio"], rtol=1e-5, atol=1e-6)
    for action in actions:
        np.testing.assert_array_equal(prepare_raw_action(action), action)

    tokenizer = AutoTokenizer.from_pretrained(config.qwen_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = tokenizer(language, max_length=contract["text_length"], truncation=True, padding="max_length", return_tensors="np")
    np.testing.assert_array_equal(text["input_ids"][0], stored["text_input_ids"])
    np.testing.assert_array_equal(text["attention_mask"][0].astype(bool), stored["attention_mask"])
    encoder = FrozenVJEPA21(config.vjepa_checkpoint).to("cuda")
    online = encode_obs(encoder, obs, "cuda", contract["visual_tokens"])[0].cpu().numpy()
    assert online.shape == stored["obs_features"].shape == (contract["visual_tokens"], config.vjepa_dim)
    reference = stored["obs_features"].astype(np.float32)
    token_cosine = np.sum(online * reference, axis=-1) / (np.linalg.norm(online, axis=-1) * np.linalg.norm(reference, axis=-1))
    relative_error = float(np.linalg.norm(online - reference) / np.linalg.norm(reference))
    # Encoding batch size may change CUDA/bfloat16 reduction order slightly.
    assert token_cosine.min() > 0.995 and relative_error < 0.05, (token_cosine.min(), relative_error)
    np.testing.assert_array_equal(uniform_pool(np.arange(1152)[None, :, None], 64).ravel(), np.linspace(0, 1151, 64).round().astype(int))
    result = {
        "passed": True, "hdf5": str(hdf5), "demo": demo_name, "t": 0,
        "cache_shard": first["file"], "contract": contract,
        "feature_cosine_mean": float(token_cosine.mean()),
        "feature_cosine_min": float(token_cosine.min()),
        "feature_relative_l2_error": relative_error,
        "proprio_max_abs_error": float(np.abs(proprio - stored["proprio"]).max()),
        "text_ids_and_mask_exact": True, "actions_and_history_exact": True,
        "raw_demo_actions_unchanged_by_postprocessing": True,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
