"""Strict LIBERO/Plus schedule validation and aggregation (no simulator needed)."""
from collections import Counter
import re

from .evaluation_protocol import canonical_json, validate_manifest, validate_rows

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = dict(zip(LIBERO_SUITES, (220, 280, 300, 520)))
PLUS_CATEGORY_COUNTS = {
    "Camera Viewpoints": 1599, "Robot Initial States": 1550,
    "Language Instructions": 1537, "Light Conditions": 1142,
    "Background Textures": 1076, "Sensor Noise": 1601, "Objects Layout": 1525,
}


def validate_libero_manifest(manifest, *, require_official=False):
    validate_manifest(manifest)
    protocol, cases = manifest["protocol"], manifest["cases"]
    from .released_checkpoint import validate_released_manifest
    validate_released_manifest(manifest)
    benchmark = protocol["benchmark"]
    if benchmark not in {"libero", "libero-plus"}:
        raise ValueError("Expected a LIBERO evaluation manifest")
    if require_official and not manifest["official"]:
        raise ValueError("Debug/preflight results cannot produce an official summary")
    for key in ("seed", "env_seed"):
        if type(protocol.get(key)) is not int or not 0 <= protocol[key] < 2**32:
            raise ValueError(f"Invalid {key}")
    if manifest["official"]:
        if protocol.get("checkpoint_profile", "paper") == "paper" and manifest["checkpoint"]["step"] != 40000:
            raise ValueError("Formal LIBERO evaluation requires the final step-40000 checkpoint")
        for key, expected in {"replan_steps": 14, "flow_steps": 4,
                              "action_horizon": 16, "wait_steps": 10}.items():
            if type(protocol.get(key)) is not int or protocol[key] != expected:
                raise ValueError(f"Paper evaluation requires {key}={expected}")
    keys, variants, task_metadata = set(), set(), {}
    for index, case in enumerate(cases):
        suite, task_id, episode = case["suite"], case["task_id"], case["episode"]
        if suite not in LIBERO_SUITES or type(task_id) is not int or task_id < 0:
            raise ValueError("Invalid LIBERO suite/task id")
        if type(episode) is not int or not 0 <= episode < (1 if benchmark == "libero-plus" else 50):
            raise ValueError("Invalid LIBERO initial-state index")
        key = (suite, task_id, episode)
        if key in keys:
            raise ValueError(f"Duplicate LIBERO case key: {key}")
        keys.add(key)
        if case["case_id"] != f"{suite}_task{task_id:05d}_ep{episode:03d}":
            raise ValueError("LIBERO case_id does not match task/episode")
        if (case["env_seed"] != protocol["env_seed"] or
                case["policy_seed"] != (protocol["seed"] + index) % 2**32 or
                case["rank"] != index % protocol["world_size"]):
            raise ValueError("LIBERO case seed/rank differs from the fixed schedule")
        if manifest["official"] and case["max_steps"] != MAX_STEPS[suite]:
            raise ValueError("LIBERO case has a nonstandard step limit")
        for name in ("bddl_sha256", "initial_state_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", case.get(name, "")):
                raise ValueError(f"Missing registry input identity: {name}")
        if not isinstance(case.get("task_name"), str) or not case["task_name"]:
            raise ValueError("Missing registry variant name")
        task_key = (suite, task_id)
        metadata = (case["task_name"], case["bddl_sha256"], case.get("language"))
        if task_key in task_metadata and task_metadata[task_key] != metadata:
            raise ValueError("Registry task identity changes between initial states")
        task_metadata[task_key] = metadata
        if benchmark == "libero-plus":
            if case.get("category") not in PLUS_CATEGORY_COUNTS:
                raise ValueError(f"Unknown LIBERO-Plus category: {case.get('category')!r}")
            variant = (suite, case["task_name"])
            if variant in variants:
                raise ValueError(f"Duplicate LIBERO-Plus registry variant: {variant}")
            variants.add(variant)
    if manifest["official"]:
        if benchmark == "libero":
            expected = {(suite, task, episode) for suite in LIBERO_SUITES
                        for task in range(10) for episode in range(50)}
            if keys != expected:
                raise ValueError("Official LIBERO evaluation requires exactly 40 tasks x 50 initial states")
            for suite in LIBERO_SUITES:
                names = {case["task_name"] for case in cases if case["suite"] == suite}
                if len(names) != 10:
                    raise ValueError("Standard LIBERO must contain ten distinct tasks per suite")
        else:
            counts = Counter(case["category"] for case in cases)
            if dict(counts) != PLUS_CATEGORY_COUNTS or len(cases) != 10030:
                raise ValueError(f"LIBERO-Plus needs all 10030 variants and seven category counts: {dict(counts)}")
            if {case["suite"] for case in cases} != set(LIBERO_SUITES):
                raise ValueError("LIBERO-Plus requires all four base suites")
            for suite in LIBERO_SUITES:
                ids = {case["task_id"] for case in cases if case["suite"] == suite}
                if ids != set(range(len(ids))):
                    raise ValueError("LIBERO-Plus registry task ids must be complete and contiguous")
    return manifest


def aggregate_libero(manifest, rows):
    validate_libero_manifest(manifest, require_official=True)
    rows = validate_rows(rows, manifest["cases"], manifest)
    suites = []
    for suite in LIBERO_SUITES:
        selected = [row for row in rows if row["suite"] == suite]
        successes = sum(row["success"] for row in selected)
        suites.append({"suite": suite, "successes": successes, "episodes": len(selected),
                       "success_rate": successes / len(selected),
                       "tasks": len({row["task_id"] for row in selected})})
    successes = sum(row["success"] for row in rows)
    result = {"official": True, "manifest_id": manifest["manifest_id"],
              "checkpoint_sha256": manifest["checkpoint"]["sha256"],
              "checkpoint_step": manifest["checkpoint"]["step"],
              "benchmark": manifest["protocol"]["benchmark"], "suite": "all",
              "protocol": manifest["protocol"], "suites": suites, "rows": rows,
              "successes": successes, "episodes": len(rows), "success_rate": successes / len(rows),
              "world_size": manifest["protocol"]["world_size"]}
    if result["benchmark"] == "libero-plus":
        categories = {}
        for category, expected in PLUS_CATEGORY_COUNTS.items():
            selected = [row for row in rows if row["category"] == category]
            if len(selected) != expected:
                raise ValueError(f"Incomplete category: {category}")
            successes = sum(row["success"] for row in selected)
            categories[category] = {"episodes": expected, "successes": successes,
                                    "success_rate": successes / expected}
        result["categories"] = categories
        result["category_macro_success_rate"] = sum(value["success_rate"] for value in categories.values()) / 7
    return result


def merge_rank_results(manifest, summaries):
    validate_libero_manifest(manifest, require_official=True)
    world_size = manifest["protocol"]["world_size"]
    ranks, rows = set(), []
    for summary in summaries:
        rank = summary.get("rank")
        if type(rank) is not int or not 0 <= rank < world_size or rank in ranks:
            raise ValueError("Duplicate or invalid evaluator rank")
        ranks.add(rank)
        if (summary.get("manifest_id") != manifest["manifest_id"] or
                summary.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"] or
                canonical_json(summary.get("checkpoint_step")) != canonical_json(manifest["checkpoint"]["step"]) or
                canonical_json(summary.get("world_size")) != canonical_json(world_size) or summary.get("official") is not True or
                canonical_json(summary.get("protocol")) != canonical_json(manifest["protocol"])):
            raise ValueError("Evaluator shards disagree on checkpoint/protocol/config")
        assigned = [case for case in manifest["cases"] if case["rank"] == rank]
        rows.extend(validate_rows(summary.get("rows", []), assigned, manifest))
    if ranks != set(range(world_size)):
        raise ValueError("One or more evaluator ranks are missing")
    return aggregate_libero(manifest, rows)
