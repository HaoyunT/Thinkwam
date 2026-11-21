#!/usr/bin/env python3
"""Run one disjoint shard of the RoboCasa24 evaluation for a P1 checkpoint.

The policy sees the three 224 px RoboCasa cameras (left agent, right agent,
wrist; resized to 384 and ImageNet-normalized), the instruction padded or
truncated to 128 Qwen tokens, a zero proprioception placeholder, and the
8 most recent executed actions. It predicts a 16 x 7 chunk in [-1, 1] with
two 4-step Euler passes, executes all 16 actions after inverting the released
low/high action statistics, and replans. Every completed case, including a
failure, is preserved. Only interrupted cases may resume under the same manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.robocasa24_eval.protocol import (  # noqa: E402
    CAMERA_NAMES, CHUNK, REPLAN_STEPS, SETTLE_STEPS, planned_cases, validate_evaluation_manifest,
)
from thinkwam_p1.evaluation_protocol import (  # noqa: E402
    atomic_json, bind_manifest, file_sha256, file_signature, make_manifest, read_case_results,
    result_identity, validate_case_result, validate_paper_checkpoint_metadata,
)

os.environ.setdefault("MUJOCO_GL", "egl")


save_json = atomic_json


def set_policy_seed(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_action(normalized, low, high, np):
    raw = (0.5 * (np.clip(normalized, -1.0, 1.0) + 1.0) * (high - low) + low).astype(np.float32)
    raw[-1] = 1.0 if raw[-1] >= 0 else -1.0
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-profile", choices=("paper", "released"), default="paper")
    parser.add_argument("--output", required=True)
    parser.add_argument("--norm-stats", default="/data/thinkwam/robocasa24_deva/norm_stats.json")
    parser.add_argument("--controller", default=str(Path(__file__).with_name("controller_osc_pose.json")))
    parser.add_argument("--vjepa-checkpoint", default="")
    parser.add_argument("--qwen-path", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug", action="store_true", help="Run a single preflight case; never a formal summary")
    parser.add_argument("--case-id", default="", help="Fixed-schedule case for --debug (default: first case)")
    args = parser.parse_args()
    if args.case_id and not args.debug:
        raise ValueError("--case-id requires --debug")
    if not 0 <= args.rank < args.ranks:
        raise ValueError("rank must be in [0, ranks)")

    import numpy as np
    import torch
    import robosuite
    import robosuite.macros as macros
    from robocasa.utils import dataset_registry  # noqa: F401
    from transformers import AutoTokenizer

    from thinkwam_p0.data.collate import _preprocess_images
    from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21
    from thinkwam_p1.config import P1Config
    from thinkwam_p1.models.p1 import P1Model

    macros.IMAGE_CONVENTION = "opengl"
    device = torch.device(args.device)
    checkpoint_stat = file_signature(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=args.checkpoint_profile == "released", mmap=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    if args.checkpoint_profile == "released":
        from thinkwam_p1.released_checkpoint import released_config
        original_config = released_config(checkpoint, "robocasa24")
    else:
        original_config = validate_paper_checkpoint_metadata(checkpoint, "robocasa24", debug=args.debug)
    overrides = {"compile_vjepa": False, "compile_action_dit": False, "qwen_attn_implementation": "sdpa"}
    if args.vjepa_checkpoint:
        overrides["vjepa_checkpoint"] = args.vjepa_checkpoint
    if args.qwen_path:
        overrides["qwen_path"] = args.qwen_path
    config = P1Config(**{**asdict(original_config), **overrides})
    expected = {"dataset": "robocasa24", "online_vjepa": True, "vjepa_views": 3,
                "action_dim": 7, "action_horizon": CHUNK, "flow_steps": 4}
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"checkpoint config {key}={getattr(config, key)!r}, expected {value!r}")
    controllers = json.loads(Path(args.controller).read_text())
    stats = json.loads(Path(args.norm_stats).read_text())
    low = np.asarray(stats["action_low"], dtype=np.float32)
    high = np.asarray(stats["action_high"], dtype=np.float32)
    if low.shape != (7,) or high.shape != (7,) or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
        raise ValueError("invalid action normalization bounds")

    output_root = Path(args.output)
    cases = planned_cases(args.ranks)
    if args.debug:
        wanted = args.case_id or cases[0]["case_id"]
        cases = [case for case in cases if case["case_id"] == wanted]
        if not cases:
            raise ValueError(f"Unknown debug case: {wanted}")
    protocol = {"benchmark": "robocasa24", "version": 1, "world_size": args.ranks,
                "action_horizon": CHUNK, "replan_steps": REPLAN_STEPS, "flow_steps": config.flow_steps,
                "settle_steps": SETTLE_STEPS, "camera_names": list(CAMERA_NAMES),
                "object_split": "B", "success_predicate": "native env._check_success",
                "attempts_per_case": 1, "evaluation_overrides": overrides,
                "norm_stats_sha256": file_sha256(args.norm_stats),
                "controller_sha256": file_sha256(args.controller),
                "render_backend": os.environ.get("MUJOCO_GL"),
                "camera_resolution": 224, "control_freq": 20, "image_convention": "opengl",
                "checkpoint_profile": args.checkpoint_profile,
                "paper_recipe_match": args.checkpoint_profile == "paper" and not args.debug}
    manifest = make_manifest(args.checkpoint, checkpoint_step, checkpoint["config"], protocol, cases, official=not args.debug)
    if checkpoint_stat != file_signature(args.checkpoint):
        raise ValueError("Checkpoint changed between loading and manifest hashing")
    validate_evaluation_manifest(manifest)
    bind_manifest(output_root, manifest)
    completed = {row["case_id"]: row for row in read_case_results(output_root / "episodes", manifest["cases"], manifest)}
    encoder = FrozenVJEPA21(None, config.vjepa_image_size, config.vjepa_patch_size)
    model = P1Model(config, qwen_backend="qwen", vjepa_encoder=encoder).to(device)
    if args.checkpoint_profile == "released":
        from thinkwam_p1.released_checkpoint import load_released_state
        load_released_state(model, checkpoint, "robocasa24")
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assigned = [case for case in manifest["cases"] if case["rank"] == args.rank]
    for case in assigned:
        output = output_root / "episodes" / case["case_id"]
        result_path = output / "result.json"
        if case["case_id"] in completed:
            continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        env = None
        steps = 0
        try:
            env = robosuite.make(
                env_name=case["task"], robots="PandaMobile", controller_configs=controllers,
                camera_names=list(CAMERA_NAMES), camera_widths=224, camera_heights=224,
                has_renderer=False, has_offscreen_renderer=True, ignore_done=False,
                horizon=case["max_steps"] + 10, control_freq=20,
                use_object_obs=True, use_camera_obs=True, camera_depths=False,
                seed=case["env_seed"], obj_instance_split="B",
                generative_textures=None, randomize_cameras=False,
                layout_and_style_ids=(tuple(case["layout_style"]),),
                translucent_robot=False,
            )
            obs = env.reset()
            for _ in range(SETTLE_STEPS):
                settle = np.zeros(env.action_dim, dtype=np.float32)
                settle[-1] = -1.0
                obs, _, _, _ = env.step(settle)
            initial_success = bool(env._check_success())
            text = tokenizer(env.get_ep_meta()["lang"], max_length=config.text_max_length,
                             truncation=True, padding="max_length", return_tensors="pt")
            ids = text["input_ids"].to(device)
            mask = text["attention_mask"].to(device).bool()
            history = torch.zeros(1, config.memory_action_history, 7, device=device)
            # RoboCasa24 has no proprioception input; a zero placeholder is used.
            proprio = torch.zeros(1, config.proprio_dim, device=device)
            set_policy_seed(case["policy_seed"], torch, np)
            success, done = initial_success, False
            while steps < case["max_steps"] and not success and not done:
                frames = [np.flipud(obs[f"{name}_image"]) for name in CAMERA_NAMES]
                images = _preprocess_images([np.stack(frames)], image_size=config.vjepa_image_size).to(device)
                with torch.inference_mode(), model.autocast_context():
                    features = model.vjepa_adapter(encoder(images).float())
                    chunk = model.predict(features, history, proprio=proprio,
                                          text_input_ids=ids, attention_mask=mask,
                                          steps=config.flow_steps,
                                          global_step=checkpoint_step)[0].float().cpu().numpy()
                if chunk.shape != (CHUNK, 7) or not np.isfinite(chunk).all():
                    raise ValueError(f"invalid action chunk: {chunk.shape}")
                for normalized in chunk[: min(REPLAN_STEPS, case["max_steps"] - steps)]:
                    raw = inverse_action(normalized, low, high, np)
                    command = np.concatenate([raw, [0, 0, 0, 0, -1]]).astype(np.float32)
                    executed = np.clip(2 * (raw - low) / (high - low) - 1, -1, 1).astype(np.float32)
                    obs, _, done, _ = env.step(command)
                    steps += 1
                    success = bool(env._check_success())
                    history = torch.cat(
                        [history[:, 1:], torch.as_tensor(executed, device=device).view(1, 1, 7)], dim=1
                    )
                    if success or done:
                        break
            result = {
                **case, **result_identity(manifest), "status": "completed", "success": success,
                "initial_success": initial_success, "steps": steps,
                "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
                "replan_steps": REPLAN_STEPS, "flow_steps": config.flow_steps,
                "wall_seconds": time.monotonic() - started,
            }
            validate_case_result(result, case, manifest)
            save_json(result_path, result, overwrite=False)
            print(json.dumps({"case_id": case["case_id"], "success": success, "steps": steps}), flush=True)
        except Exception:
            save_json(output / "error.json", {**case, **result_identity(manifest), "status": "error", "steps": steps,
                                              "traceback": traceback.format_exc()})
            raise
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
