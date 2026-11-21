"""CPU fixtures exercise the same manifest/resume/merge gates as the CLIs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from thinkwam_p1.evaluation_protocol import (
    atomic_json, bind_manifest, make_manifest, object_sha256, read_case_results,
    result_identity, validate_case_result, validate_manifest, validate_rows,
    validate_checkpoint_metadata,
)
from thinkwam_p1.libero_evaluation import (
    LIBERO_SUITES, MAX_STEPS, PLUS_CATEGORY_COUNTS, aggregate_libero,
    merge_rank_results, validate_libero_manifest,
)
from scripts.robocasa24_eval.protocol import (
    CAMERA_NAMES, planned_cases, validate_evaluation_manifest,
)
from scripts.robocasa24_eval.collect import collect


def resign(manifest):
    manifest.pop("manifest_id", None)
    manifest["manifest_id"] = object_sha256(manifest)
    return manifest


def manifest_for(tmp_path, benchmark="libero", *, official=True):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for CPU validation")
    protocol = {"benchmark": benchmark, "world_size": 2, "seed": 7, "env_seed": 0,
                "replan_steps": 14, "flow_steps": 4, "action_horizon": 16, "wait_steps": 10}
    cases = []
    if benchmark == "robocasa24":
        protocol.update(replan_steps=16, settle_steps=10, camera_names=list(CAMERA_NAMES),
                        object_split="B", success_predicate="native env._check_success",
                        norm_stats_sha256="a" * 64, controller_sha256="b" * 64)
        cases = planned_cases(2)
    else:
        if benchmark == "libero":
            specs = [(suite, task, episode, None) for suite in LIBERO_SUITES
                     for task in range(10) for episode in range(50)]
        else:
            specs, next_ids = [], {suite: 0 for suite in LIBERO_SUITES}
            for category, count in PLUS_CATEGORY_COUNTS.items():
                for index in range(count):
                    suite = LIBERO_SUITES[len(specs) % 4]
                    task = next_ids[suite]
                    next_ids[suite] += 1
                    specs.append((suite, task, 0, category))
        for index, (suite, task, episode, category) in enumerate(specs):
            case = {"case_id": f"{suite}_task{task:05d}_ep{episode:03d}",
                    "suite": suite, "task_id": task, "task_name": f"task_{task}",
                    "episode": episode, "env_seed": 0, "policy_seed": 7 + index,
                    "rank": index % 2, "max_steps": MAX_STEPS[suite],
                    "bddl_sha256": "a" * 64, "initial_state_sha256": "b" * 64}
            if category:
                case["category"] = category
            cases.append(case)
    return make_manifest(checkpoint, 40000, {"dataset": benchmark}, protocol, cases, official=official)


def rows_for(manifest):
    return [{**case, **result_identity(manifest), "status": "completed", "steps": 1,
             "success": index % 2 == 0, "initial_success": False}
            for index, case in enumerate(manifest["cases"])]


@pytest.mark.parametrize("benchmark", ["libero", "libero-plus", "robocasa24"])
def test_published_checkpoint_keeps_complete_evaluation_schedule(tmp_path, benchmark):
    from thinkwam_p1.released_checkpoint import release_profile
    manifest = manifest_for(tmp_path, benchmark)
    profile = release_profile(benchmark)
    manifest["checkpoint"] = {key: profile[key] for key in ("step", "sha256", "size_bytes")}
    manifest["model_config"] = profile["config"]
    manifest["protocol"].update(checkpoint_profile="released", paper_recipe_match=False)
    resign(manifest)
    validate = validate_evaluation_manifest if benchmark == "robocasa24" else validate_libero_manifest
    validate(manifest, require_official=True)
    incomplete = deepcopy(manifest)
    incomplete["cases"].pop()
    resign(incomplete)
    with pytest.raises(ValueError):
        validate(incomplete, require_official=True)


def shards_for(manifest):
    rows = rows_for(manifest)
    return [{**result_identity(manifest), "official": True, "protocol": manifest["protocol"],
             "rank": rank, "world_size": 2, "successes": 999999, "episodes": 999999,
             "rows": [row for row in rows if row["rank"] == rank]} for rank in range(2)]


def test_checkpoint_content_config_and_protocol_bind_resume(tmp_path):
    manifest = manifest_for(tmp_path)
    output = tmp_path / "output"
    bind_manifest(output, manifest)
    bind_manifest(output, deepcopy(manifest))
    for field in ("checkpoint", "model_config", "protocol"):
        changed = deepcopy(manifest)
        if field == "checkpoint":
            changed[field]["sha256"] = "c" * 64
        elif field == "model_config":
            changed[field]["new_option"] = True
        else:
            changed[field]["seed"] = 8
        resign(changed)
        with pytest.raises(ValueError, match="different checkpoint/protocol/config"):
            bind_manifest(output, changed)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"different contents at the SAME checkpoint path")
    changed = make_manifest(checkpoint, 40000, manifest["model_config"], manifest["protocol"], manifest["cases"])
    assert changed["checkpoint"]["sha256"] != manifest["checkpoint"]["sha256"]
    with pytest.raises(ValueError):
        bind_manifest(output, changed)


def test_unbound_old_results_cannot_be_adopted(tmp_path):
    manifest = manifest_for(tmp_path)
    root = tmp_path / "old"
    atomic_json(root / "episodes" / "old_case" / "result.json", {"success": True})
    with pytest.raises(ValueError, match="no verified manifest"):
        bind_manifest(root, manifest)


@pytest.mark.parametrize("field,value", [
    ("success", 1), ("success", "false"), ("status", "error"), ("env_seed", 99),
    ("policy_seed", 99), ("checkpoint_sha256", "c" * 64), ("checkpoint_step", 10),
    ("manifest_id", "c" * 64), ("episode", 50), ("task_id", -1),
    ("steps", -1), ("steps", 100000), ("steps", True),
])
def test_invalid_case_fields_rejected(tmp_path, field, value):
    manifest = manifest_for(tmp_path)
    row = rows_for(manifest)[0]
    row[field] = value
    with pytest.raises(ValueError):
        validate_case_result(row, manifest["cases"][0], manifest)


def test_missing_duplicate_and_unknown_cases_rejected(tmp_path):
    manifest = manifest_for(tmp_path)
    rows = rows_for(manifest)
    for invalid in (rows[:-1], rows + [rows[0]], [{**rows[0], "case_id": "unknown"}] + rows[1:]):
        with pytest.raises(ValueError):
            validate_rows(invalid, manifest["cases"], manifest)


def test_completed_failure_is_preserved_during_resume(tmp_path):
    manifest = manifest_for(tmp_path)
    row = rows_for(manifest)[1]
    assert row["success"] is False
    atomic_json(tmp_path / "episodes" / row["case_id"] / "result.json", row)
    resumed = read_case_results(tmp_path / "episodes", manifest["cases"], manifest)
    assert resumed == [row]
    row["policy_seed"] += 1
    atomic_json(tmp_path / "episodes" / row["case_id"] / "result.json", row)
    with pytest.raises(ValueError):
        read_case_results(tmp_path / "episodes", manifest["cases"], manifest)


def test_completed_result_cannot_be_overwritten(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"success": False}, overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_json(path, {"success": True}, overwrite=False)
    assert json.loads(path.read_text()) == {"success": False}


def test_libero_path_config_cannot_silently_switch_checkout(tmp_path, monkeypatch):
    from thinkwam_p1 import libero_eval_inputs as inputs
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.setenv("THINKWAM_LIBERO_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(inputs, "LIBERO_ROOT", tmp_path / "standard")
    inputs.ensure_libero_config()
    inputs.ensure_libero_config()
    monkeypatch.setattr(inputs, "LIBERO_ROOT", tmp_path / "plus")
    with pytest.raises(ValueError, match="different checkout"):
        inputs.ensure_libero_config()


def test_libero_merge_recomputes_counts_from_complete_rows(tmp_path):
    manifest = manifest_for(tmp_path)
    result = merge_rank_results(manifest, shards_for(manifest))
    assert result["episodes"] == 2000 and result["successes"] == 1000
    assert result["success_rate"] == 0.5
    assert [suite["episodes"] for suite in result["suites"]] == [500] * 4


@pytest.mark.parametrize("mutation", ["checkpoint", "protocol", "duplicate_rank", "missing_rank", "missing_case", "wrong_rank_case"])
def test_libero_merge_rejects_mixed_or_incomplete_shards(tmp_path, mutation):
    manifest = manifest_for(tmp_path)
    shards = shards_for(manifest)
    if mutation == "checkpoint":
        shards[1]["checkpoint_sha256"] = "c" * 64
    elif mutation == "protocol":
        shards[1]["protocol"] = {**shards[1]["protocol"], "replan_steps": 15}
    elif mutation == "duplicate_rank":
        shards[1]["rank"] = 0
    elif mutation == "missing_rank":
        shards.pop()
    elif mutation == "missing_case":
        shards[1]["rows"].pop()
    else:
        shards[1]["rows"][0] = shards[0]["rows"][0]
    with pytest.raises(ValueError):
        merge_rank_results(manifest, shards)


def test_plus_requires_full_registry_and_seven_categories(tmp_path):
    manifest = manifest_for(tmp_path, "libero-plus")
    result = aggregate_libero(manifest, rows_for(manifest))
    assert result["episodes"] == 10030
    assert {key: value["episodes"] for key, value in result["categories"].items()} == PLUS_CATEGORY_COUNTS
    assert result["category_macro_success_rate"] == pytest.approx(
        sum(value["success_rate"] for value in result["categories"].values()) / 7)
    for mutation in ("unknown_category", "missing_variant", "duplicate_variant", "nonzero_initial_state"):
        changed = deepcopy(manifest)
        if mutation == "unknown_category":
            changed["cases"][0]["category"] = None
        elif mutation == "missing_variant":
            changed["cases"].pop()
        elif mutation == "duplicate_variant":
            changed["cases"][4]["task_name"] = changed["cases"][0]["task_name"]
        else:
            changed["cases"][0]["episode"] = 1
        resign(changed)
        with pytest.raises(ValueError):
            validate_libero_manifest(changed)


def test_debug_cannot_become_a_formal_summary(tmp_path):
    manifest = manifest_for(tmp_path, official=False)
    with pytest.raises(ValueError, match="Debug/preflight"):
        aggregate_libero(manifest, rows_for(manifest))
    rc = manifest_for(tmp_path, "robocasa24", official=False)
    with pytest.raises(ValueError, match="Debug/preflight"):
        validate_evaluation_manifest(rc, require_official=True)


def test_robocasa_collect_counts_all_1200_including_failures(tmp_path):
    manifest = manifest_for(tmp_path, "robocasa24")
    bind_manifest(tmp_path / "output", manifest)
    for row in rows_for(manifest):
        atomic_json(tmp_path / "output" / "episodes" / row["case_id"] / "result.json", row)
    result = collect(tmp_path / "output")
    assert result["episodes"] == 1200 and result["successes"] == 600
    assert all(task["episodes"] == 50 for task in result["tasks"].values())
    last = manifest["cases"][-1]
    path = tmp_path / "output" / "episodes" / last["case_id"] / "result.json"
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        collect(tmp_path / "output")


def test_robocasa_rejects_changed_schedule_or_seed(tmp_path):
    manifest = manifest_for(tmp_path, "robocasa24")
    for field, value in (("env_seed", -1), ("layout_style", [999, 999]), ("episode_idx", 50)):
        changed = deepcopy(manifest)
        changed["cases"][0][field] = value
        resign(changed)
        with pytest.raises(ValueError):
            validate_evaluation_manifest(changed)


def test_manifest_digest_detects_edit(tmp_path):
    manifest = manifest_for(tmp_path)
    manifest["model_config"]["dataset"] = "different"
    with pytest.raises(ValueError, match="digest"):
        validate_manifest(manifest)


def test_checkpoint_schema_budget_and_scientific_recipe():
    reference = {"dataset": "libero", "max_steps": 40000, "action_horizon": 16,
                 "jepa_token_counts": [288, 144, 72], "first_action_loss_weight": 1.2,
                 "video_residual_loss_weight": 0.2,
                 "batch_size": 24, "dataset_root": "/data/demos", "compile_vjepa": True}
    checkpoint = {"config": deepcopy(reference), "step": 40000}
    validate_checkpoint_metadata(checkpoint, reference)
    checkpoint["config"]["dataset_root"] = "/other/location"
    checkpoint["config"]["compile_vjepa"] = False
    validate_checkpoint_metadata(checkpoint, reference)
    for field, value in (("max_steps", 75000), ("action_horizon", 20),
                         ("jepa_token_counts", [144]), ("first_action_loss_weight", 2.0),
                         ("video_residual_loss_weight", 0.8), ("video_residual_loss_weight", 1.0),
                         ("batch_size", 20)):
        changed = deepcopy(checkpoint)
        changed["config"][field] = value
        with pytest.raises(ValueError, match="paper training recipe"):
            validate_checkpoint_metadata(changed, reference)
    checkpoint["step"] = 10000
    with pytest.raises(ValueError, match="final step"):
        validate_checkpoint_metadata(checkpoint, reference)
    validate_checkpoint_metadata(checkpoint, reference, official=False)
    for values in ({}, {**reference, "legacy_mixture": True}, {key: value for key, value in reference.items() if key != "batch_size"}):
        with pytest.raises(ValueError, match="complete current embedded config"):
            validate_checkpoint_metadata({"config": values, "step": 10000}, reference, official=False)
