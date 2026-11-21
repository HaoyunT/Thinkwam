#!/usr/bin/env python3
"""Closed-loop evaluator for a P1 checkpoint on LIBERO or LIBERO-Plus.

The P1 trainer consumes raw RGB images and applies the V-JEPA residual adapter
inside ``P1Model._encode_images``.  This evaluator mirrors that boundary for
closed-loop rollouts: it computes the two camera V-JEPA tokens online, applies
the trained adapter, and then calls the P1 two-pass action policy.

Each evaluator process owns one CUDA device and receives a disjoint slice of
the benchmark jobs through ``--rank``/``--world-size``.  The launch wrapper
starts independent ranks for one selected benchmark. Completed cases, including
failures, are resumed only after manifest and result validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

# The benchmark checkout must be selected before this module is imported.
# The wrapper sets THINKWAM_LIBERO_ROOT separately for the standard and Plus
# subprocesses, which keeps the two incompatible task registries isolated.
LIBERO_ROOT = Path(os.environ.get("THINKWAM_LIBERO_ROOT", "/data/thinkwam/src/LIBERO"))
ROBOSUITE_COMPAT = Path("/data/thinkwam/compat/robosuite140")
if ROBOSUITE_COMPAT.is_dir():
    sys.path.insert(0, str(ROBOSUITE_COMPAT))
if LIBERO_ROOT.is_dir():
    sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkwam_p1.libero_eval_inputs import (  # noqa: E402
    LIBERO_SUITES,
    ensure_libero_config,
    get_proprio,
    max_steps_for_suite,
    training_contract,
    video_frame,
)
from thinkwam_p0.data.libero_inputs import online_cache_features, prepare_raw_action  # noqa: E402
from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21  # noqa: E402
from thinkwam_p1.config import P1Config  # noqa: E402
from thinkwam_p1.models.p1 import P1Model  # noqa: E402


from thinkwam_p1.evaluation_protocol import (  # noqa: E402
    atomic_json, bind_manifest, file_sha256, file_signature, make_manifest, read_case_results,
    result_identity, validate_case_result, validate_paper_checkpoint_metadata,
)
from thinkwam_p1.libero_evaluation import (  # noqa: E402
    PLUS_CATEGORY_COUNTS, validate_libero_manifest,
)

PLUS_CATEGORIES = tuple(PLUS_CATEGORY_COUNTS)

CATEGORY_ALIASES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "sensor": "Sensor Noise",
    "noise": "Sensor Noise",
    "objects": "Objects Layout",
    "layout": "Objects Layout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-profile", choices=("paper", "released"), default="paper")
    parser.add_argument("--qwen-path", default="", help="Local Qwen2.5-0.5B model/tokenizer directory")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), required=True)
    parser.add_argument("--suite", default="all", help="all or a comma-separated suite list")
    parser.add_argument(
        "--task-spec",
        default="",
        help="Optional suite-specific task selection, e.g. libero_goal:3,5,6;libero_10:5,6",
    )
    parser.add_argument("--plus-categories", default="all")
    parser.add_argument("--tasks", type=int, default=-1, help="-1 means all tasks per suite")
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Initial states per LIBERO task. LIBERO-Plus always uses initial state 0 of every variant.",
    )
    # The paper executes the first 14 actions of each 16-step chunk.
    parser.add_argument("--replan-steps", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-root", default="", help="Shared output root used by all ranks")
    parser.add_argument("--debug", action="store_true", help="Allow a subset/preflight; never write a formal summary")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one task/one episode with one action cycle for a functional check.",
    )
    return parser.parse_args()



def load_checkpoint_and_config(args: argparse.Namespace):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    args.checkpoint_stat = file_signature(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=args.checkpoint_profile == "released", mmap=True)
    if args.checkpoint_profile == "released":
        from thinkwam_p1.released_checkpoint import released_config
        config = released_config(checkpoint, args.benchmark)
    else:
        config = validate_paper_checkpoint_metadata(checkpoint, args.benchmark, debug=args.debug or args.smoke)
    # Relocate assets without altering the embedded checkpoint metadata.
    if args.qwen_path:
        config.qwen_path = args.qwen_path
    config.qwen_attn_implementation = args.attention
    config.compile_vjepa = config.compile_action_dit = config.gradient_checkpointing = False
    if not config.online_vjepa:
        raise ValueError(
            "This evaluator mirrors the current raw-image P1 contract and requires online_vjepa=true."
        )
    if config.dataset != "libero":
        raise ValueError(f"Expected a LIBERO P1 checkpoint, got dataset={config.dataset!r}")
    return checkpoint_path, checkpoint, config


def resolve_categories(spec: str) -> set[str] | None:
    value = str(spec or "all").strip()
    if value.lower() in {"all", "*"}:
        return None
    selected: set[str] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        canonical = CATEGORY_ALIASES.get(raw.lower(), raw)
        if canonical not in PLUS_CATEGORIES:
            raise ValueError(
                f"Unknown LIBERO-Plus category {raw!r}; use {PLUS_CATEGORIES} or all."
            )
        selected.add(canonical)
    return selected or None


def load_plus_classification() -> dict[str, dict[str, dict[str, Any]]]:
    path = LIBERO_ROOT / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus classification file not found: {path}")
    payload = json.loads(path.read_text())
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for suite, rows in payload.items():
        if suite not in LIBERO_SUITES or not isinstance(rows, list):
            raise ValueError(f"Invalid LIBERO-Plus classification suite: {suite}")
        result[suite] = {}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                    or not row["name"] or row.get("category") not in PLUS_CATEGORIES):
                raise ValueError(f"Unclassified or invalid LIBERO-Plus registry entry in {suite}")
            if row["name"] in result[suite]:
                raise ValueError(f"Duplicate classification entry: {suite}/{row['name']}")
            result[suite][row["name"]] = row
    return result


def load_model(
    checkpoint: dict[str, Any],
    config: P1Config,
    device: torch.device,
    checkpoint_profile: str = "paper",
    benchmark: str = "libero",
) -> tuple[P1Model, FrozenVJEPA21, int]:
    encoder = FrozenVJEPA21(
        None,  # The complete frozen encoder is already in the policy state.
        config.vjepa_image_size,
        config.vjepa_patch_size,
    ).to(device)
    model = P1Model(config, qwen_backend="qwen", vjepa_encoder=encoder).to(device)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    if checkpoint_profile == "released":
        from thinkwam_p1.released_checkpoint import load_released_state
        load_released_state(model, checkpoint, benchmark)
        missing, unexpected = [], []
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
    # A P1 checkpoint and this evaluator use the same V-JEPA module contract.
    # Refuse silent shape/name drift; a partial load would make the benchmark
    # result uninterpretable.
    if missing or unexpected:
        raise RuntimeError(
            f"P1 checkpoint/model mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval()
    encoder.eval()
    return model, encoder, int(checkpoint.get("step", 0))


def adapted_features(model: P1Model, encoder: FrozenVJEPA21, obs: dict, device, token_count: int):
    """Encode the two RGB views and apply the trained P1 V-JEPA adapter."""
    features = online_cache_features(encoder, obs, device, token_count).float()
    # P1 training calls this adapter in _encode_images before any Qwen/memory
    # branch. P1Model.predict accepts features, so the evaluator must preserve
    # that boundary explicitly.
    return model.vjepa_adapter(features)


def selected_suites(suite_spec: str) -> tuple[str, ...]:
    if suite_spec == "all":
        return LIBERO_SUITES
    suites = tuple(item.strip() for item in suite_spec.split(",") if item.strip())
    if not suites or len(suites) != len(set(suites)) or any(suite not in LIBERO_SUITES for suite in suites):
        raise ValueError("--suite must name distinct standard LIBERO suites")
    return suites


def parse_task_spec(spec: str) -> dict[str, list[int]]:
    task_spec: dict[str, list[int]] = {}
    for item in str(spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --task-spec item {item!r}; expected suite:id,id")
        suite, ids = item.split(":", 1)
        parsed = [int(value.strip()) for value in ids.split(",") if value.strip()]
        if not suite.strip() or not parsed:
            raise ValueError(f"Invalid --task-spec item {item!r}; expected suite:id,id")
        if suite.strip() not in LIBERO_SUITES or suite.strip() in task_spec or len(parsed) != len(set(parsed)):
            raise ValueError("Task selection contains an unknown suite or duplicate task")
        task_spec[suite.strip()] = parsed
    return task_spec


def build_jobs(args: argparse.Namespace, get_benchmark):
    task_spec = parse_task_spec(args.task_spec)
    suites = tuple(task_spec) if task_spec else selected_suites(args.suite)
    classification = load_plus_classification() if args.benchmark == "libero-plus" else {}
    categories = resolve_categories(args.plus_categories) if args.benchmark == "libero-plus" else None
    if args.benchmark == "libero" and str(args.plus_categories).strip().lower() not in {"", "all", "*"}:
        raise ValueError("--plus-categories only applies to --benchmark libero-plus")

    jobs: list[tuple[str, int, int]] = []
    task_counts: dict[str, int] = {}
    benchmarks = {}
    for suite in suites:
        benchmark = get_benchmark(suite)(0)
        benchmarks[suite] = benchmark
        if suite in task_spec:
            task_ids = task_spec[suite]
            invalid = [
                task_id
                for task_id in task_ids
                if task_id < 0 or task_id >= benchmark.get_num_tasks()
            ]
            if invalid:
                raise ValueError(
                    f"Invalid task ids for {suite}: {invalid}; "
                    f"benchmark has {benchmark.get_num_tasks()} tasks"
                )
        else:
            num_tasks = benchmark.get_num_tasks() if args.tasks < 0 else min(args.tasks, benchmark.get_num_tasks())
            task_ids = list(range(num_tasks))
        selected = []
        for task_id in task_ids:
            task = benchmark.get_task(task_id)
            if args.benchmark == "libero-plus":
                metadata = classification.get(suite, {}).get(task.name, {})
                if metadata.get("category") not in PLUS_CATEGORIES:
                    raise ValueError(f"Unclassified LIBERO-Plus variant: {suite}/{task.name}")
                if categories is not None and metadata["category"] not in categories:
                    continue
            selected.append(task_id)
            init_states = benchmark.get_task_init_states(task_id)
            # LIBERO uses the first N initial states per task; every LIBERO-Plus
            # registry variant is evaluated once at initial state 0.
            episodes = 1 if args.benchmark == "libero-plus" else args.episodes
            if len(init_states) < episodes:
                raise ValueError(f"Insufficient initial states for {suite}/{task.name}: {len(init_states)} < {episodes}")
            for episode in range(episodes):
                jobs.append((suite, task_id, episode))
        if args.benchmark == "libero-plus" and not (args.debug or args.smoke):
            names = {benchmark.get_task(i).name for i in range(benchmark.get_num_tasks())}
            if len(names) != benchmark.get_num_tasks() or names != set(classification.get(suite, {})):
                raise ValueError(f"Registry and classification variants disagree for {suite}")
        task_counts[suite] = len(selected)
    return suites, benchmarks, jobs, task_counts


def build_case_schedule(args, benchmarks, jobs, classification, get_libero_path):
    cases, task_inputs = [], {}
    for index, (suite, task_id, episode) in enumerate(jobs):
        benchmark = benchmarks[suite]
        task = benchmark.get_task(task_id)
        key = (suite, task_id)
        if key not in task_inputs:
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            task_inputs[key] = (file_sha256(bddl), benchmark.get_task_init_states(task_id))
        bddl_sha256, initial_states = task_inputs[key]
        state = np.ascontiguousarray(initial_states[episode])
        digest = hashlib.sha256()
        digest.update(str(state.dtype).encode())
        digest.update(str(state.shape).encode())
        digest.update(state.tobytes())
        case = {"case_id": f"{suite}_task{task_id:05d}_ep{episode:03d}",
                "suite": suite, "task_id": task_id, "task_name": task.name,
                "language": task.language, "episode": episode,
                "env_seed": args.env_seed, "policy_seed": (args.seed + index) % 2**32,
                "max_steps": args.max_steps or max_steps_for_suite(suite),
                "rank": index % args.world_size, "bddl_sha256": bddl_sha256,
                "initial_state_sha256": digest.hexdigest()}
        if args.benchmark == "libero-plus":
            case["category"] = classification[suite][task.name]["category"]
        cases.append(case)
    return cases


def run(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path | None = None,
    checkpoint: dict[str, Any] | None = None,
    config: P1Config | None = None,
    tokenizer=None,
) -> dict[str, Any]:
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank must be in [0, world_size), got {args.rank}/{args.world_size}")
    if args.episodes <= 0 or args.replan_steps <= 0 or args.wait_steps < 0:
        raise ValueError("episodes/replan-steps must be positive and wait-steps non-negative")

    debug = args.debug or args.smoke
    if not debug and (args.suite != "all" or args.task_spec or args.tasks != -1
                      or args.plus_categories != "all" or args.episodes != 50
                      or args.replan_steps != 14 or args.max_steps != 0 or args.wait_steps != 10):
        raise ValueError("Formal evaluation requires the full paper protocol; use --debug for a preflight/subset")
    if checkpoint_path is None or checkpoint is None or config is None:
        checkpoint_path, checkpoint, config = load_checkpoint_and_config(args)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_path,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    contract = training_contract(config)
    checkpoint_step = int(checkpoint.get("step", 0))
    if not debug:
        for key, expected in {"action_horizon": 16, "action_dim": 7, "flow_steps": 4, "vjepa_views": 2}.items():
            if getattr(config, key) != expected:
                raise ValueError(f"Paper LIBERO evaluation requires {key}={expected}")

    ensure_libero_config()
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite import macros
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("LIBERO dependencies are required for closed-loop evaluation") from exc

    suites, benchmarks, jobs, task_counts = build_jobs(args, get_benchmark)
    classification = load_plus_classification() if args.benchmark == "libero-plus" else {}
    if args.smoke:
        jobs = jobs[:1]
        args.replan_steps = 1
        args.max_steps = 1
    cases = build_case_schedule(args, benchmarks, jobs, classification, get_libero_path)
    camera_resolution = int(contract["camera_resolution"])
    macros.IMAGE_CONVENTION = contract["image_convention"]
    protocol = {
        "benchmark": args.benchmark, "version": 1, "world_size": args.world_size,
        "replan_steps": args.replan_steps, "flow_steps": config.flow_steps,
        "action_horizon": config.action_horizon, "wait_steps": args.wait_steps,
        "seed": args.seed, "env_seed": args.env_seed, "input_contract": contract,
        "success_predicate": "native env.check_success", "attempts_per_case": 1,
        "policy_seed_schedule": "seed + canonical case index, modulo 2**32",
        "evaluation_overrides": {"qwen_path": config.qwen_path, "attention": args.attention},
        "checkpoint_profile": args.checkpoint_profile,
        "paper_recipe_match": args.checkpoint_profile == "paper" and not debug,
        "render_backend": os.environ.get("MUJOCO_GL"),
    }
    manifest = make_manifest(checkpoint_path, checkpoint_step, checkpoint["config"], protocol, cases, official=not debug)
    if getattr(args, "checkpoint_stat", file_signature(checkpoint_path)) != file_signature(checkpoint_path):
        raise ValueError("Checkpoint changed between loading and manifest hashing")
    validate_libero_manifest(manifest)
    output = Path(args.output).expanduser()
    bind_manifest(Path(args.manifest_root).expanduser() if args.manifest_root else output, manifest)
    bind_manifest(output, manifest)
    assigned_cases = [case for case in manifest["cases"] if case["rank"] == args.rank]
    completed = {row["case_id"]: row for row in read_case_results(output / "episodes", assigned_cases, manifest)}
    atomic_json(output / "evaluation_config.json", {**protocol, **result_identity(manifest), "official": not debug})
    model, encoder, checkpoint_step = load_model(checkpoint, config, device, args.checkpoint_profile, args.benchmark)
    del checkpoint

    successes = 0
    episodes = 0
    rows: list[dict[str, Any]] = []
    suite_counts = {
        suite: {"successes": 0, "episodes": 0}
        for suite in suites
    }

    for case in assigned_cases:
        suite, task_id, episode = case["suite"], case["task_id"], case["episode"]
        if case["case_id"] in completed:
            row = completed[case["case_id"]]
            rows.append(row)
            successes += int(row["success"])
            episodes += 1
            suite_counts[suite]["successes"] += int(row["success"])
            suite_counts[suite]["episodes"] += 1
            continue
        np.random.seed(case["policy_seed"])
        random.seed(case["policy_seed"])
        torch.manual_seed(case["policy_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(case["policy_seed"])
        benchmark = benchmarks[suite]
        task = benchmark.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=camera_resolution,
            camera_widths=camera_resolution,
        )
        replay_images = []
        success = False
        step_count = 0
        clipped_values = 0
        try:
            env.seed(case["env_seed"])
            init_states = benchmark.get_task_init_states(task_id)
            env.reset()
            obs = env.set_init_state(init_states[episode])
            for _ in range(args.wait_steps):
                dummy = np.zeros(7, dtype=np.float32)
                dummy[-1] = -1.0
                obs, _, _, _ = env.step(dummy)
            if args.save_videos and not args.no_videos:
                replay_images.append(video_frame(obs))

            text_length = int(config.text_max_length)
            text = tokenizer(
                task.language,
                max_length=text_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids = text["input_ids"].to(device)
            mask = text["attention_mask"].to(device).bool()
            history = torch.zeros(
                1,
                config.memory_action_history,
                config.action_dim,
                device=device,
            )
            limit = args.max_steps or max_steps_for_suite(suite)
            while step_count < limit and not success:
                features = adapted_features(
                    model,
                    encoder,
                    obs,
                    device,
                    config.visual_token_count,
                )
                proprio = get_proprio(obs, config.proprio_dim).to(device)[None]
                actions = model.predict(
                    features,
                    history,
                    proprio=proprio,
                    text_input_ids=ids,
                    attention_mask=mask,
                    steps=config.flow_steps,
                    global_step=checkpoint_step,
                )[0].detach().float().cpu().numpy()
                execute = min(args.replan_steps, len(actions), limit - step_count)
                for action in actions[:execute]:
                    action = np.asarray(action, dtype=np.float32).copy()
                    clipped_values += int((np.abs(action[:6]) > 1).sum())
                    action = prepare_raw_action(action)
                    obs, _, _, _ = env.step(action)
                    if args.save_videos and not args.no_videos:
                        replay_images.append(video_frame(obs))
                    history = torch.cat(
                        [
                            history[:, 1:],
                            torch.from_numpy(action).to(device).view(1, 1, -1),
                        ],
                        dim=1,
                    )
                    step_count += 1
                    # Success is decided by the native task predicates only.
                    if bool(env.check_success()):
                        success = True
                        break
        finally:
            env.close()

        video_path = None
        if args.save_videos and not args.no_videos and replay_images:
            import imageio.v2 as imageio

            video_path = output / f"{suite}_task{task_id:04d}_episode{episode:03d}_success_{int(success)}.mp4"
        successes += int(success)
        episodes += 1
        suite_counts[suite]["successes"] += int(success)
        suite_counts[suite]["episodes"] += 1
        row = {
            **case, **result_identity(manifest), "status": "completed",
            "suite": suite,
            "task_id": task_id,
            "task": task.language,
            "episode": episode,
            "success": bool(success),
            "steps": step_count,
            "video": str(video_path) if video_path else None,
            "video_frames": len(replay_images),
            "arm_clip_fraction": clipped_values / max(1, step_count * 6),
        }
        validate_case_result(row, case, manifest)
        atomic_json(output / "episodes" / case["case_id"] / "result.json", row, overwrite=False)
        # A video encoding failure must not discard a completed rollout.
        if video_path is not None:
            imageio.mimsave(video_path, replay_images, fps=20)
        rows.append(row)
        print(
            f"benchmark={args.benchmark} rank={args.rank}/{args.world_size} "
            f"suite={suite} task={task_id} episode={episode} "
            f"success={success} steps={step_count}",
            flush=True,
        )

    suite_summaries = [
        {
            "suite": suite,
            "successes": suite_counts[suite]["successes"],
            "episodes": suite_counts[suite]["episodes"],
            "success_rate": suite_counts[suite]["successes"]
            / max(1, suite_counts[suite]["episodes"]),
            "tasks": task_counts[suite],
        }
        for suite in suites
    ]
    result = {
        **result_identity(manifest), "official": not debug, "protocol": protocol,
        "benchmark": args.benchmark,
        "suite": args.suite,
        "suites": suite_summaries,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / max(1, episodes),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "world_size": args.world_size,
        "rank": args.rank,
        "evaluation_config": protocol,
        "rows": rows,
    }
    # Subsets and single-case checks never create a formal summary artifact.
    atomic_json(output / ("debug_results.json" if debug else "summary.json"), result)
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "rank": args.rank,
                "successes": successes,
                "episodes": episodes,
                "success_rate": result["success_rate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    checkpoint_path, checkpoint, config = load_checkpoint_and_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    run(
        args,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
