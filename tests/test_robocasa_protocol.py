"""CPU metadata fixtures for the formal RoboCasa24 data boundary."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from thinkwam_p1.robocasa_protocol import (
    EPISODES, TASKS, VIEW_NAMES, read_source_json, source_path,
    validate_action_statistics, validate_robocasa_source, validate_robocasa_split,
)


@pytest.fixture(scope="module")
def paper_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("robocasa_paper_source")
    for folder in ("videos", "metas", "affordance"):
        (root / folder).mkdir()
    (root / "dataset_info.json").write_text(json.dumps({
        "views": list(VIEW_NAMES), "tile": [[VIEW_NAMES[0], VIEW_NAMES[1]], [VIEW_NAMES[2], None]],
    }))
    (root / "norm_stats.json").write_text(json.dumps({
        "action_low": [-1.0] * 7, "action_high": [1.0] * 7,
        "norm_method": "minmax_symmetric", "action_dim": 7, "use_delta_actions": False,
        "action_horizon": 33,
    }))
    actions = {f"demo_{i}": [[0.0] * 7] for i in range(1199)}
    (root / "action.json").write_text(json.dumps(actions))
    mapping, start = {}, 0
    for i, task in enumerate(sorted(TASKS)):
        count = 50 if i < 23 else 49
        mapping[task] = {"start_idx": start, "end_idx": start + count - 1, "count": count}
        start += count
    (root / "data_mapping.json").write_text(json.dumps(mapping))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{view: np.zeros((1, 224, 224), dtype=np.float16) for view in VIEW_NAMES})
    labels = buffer.getvalue()
    for episode in actions:
        (root / f"videos/{episode}.mp4").write_bytes(b"metadata fixture: decoder tested separately")
        (root / f"metas/{episode}.txt").write_text("open the drawer")
        (root / f"affordance/{episode}.npz").write_bytes(labels)
    return root


def config(root, *, p1=False):
    values = dict(dataset="robocasa24", dataset_root=str(root), online_vjepa=True, seed=7, eval_fraction=0.1)
    if p1:
        values["robocasa24_require_affordance"] = True
    return SimpleNamespace(**values)


@contextmanager
def changed(path, value=None):
    existed = path.exists()
    previous = path.read_bytes() if existed else None
    try:
        if value is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(value if isinstance(value, bytes) else value.encode())
        yield
    finally:
        path.unlink(missing_ok=True)
        if existed:
            path.write_bytes(previous)


def test_formal_source_accepts_complete_export_and_p0_needs_no_labels(paper_source):
    assert validate_robocasa_source(config(paper_source))["episodes"] == 1199
    report = validate_robocasa_source(config(paper_source, p1=True))
    assert report["tasks"] == 24 and report["windows"] == 1199 and report["require_affordance"]
    with changed(paper_source / "affordance/demo_0.npz"):
        validate_robocasa_source(config(paper_source))
        with pytest.raises(ValueError, match="affordance episode set"):
            validate_robocasa_source(config(paper_source, p1=True))


def test_formal_source_rejects_consistently_reduced_episode_set(paper_source):
    actions = json.loads((paper_source / "action.json").read_text())
    actions.pop("demo_0")
    with changed(paper_source / "action.json", json.dumps(actions)):
        with changed(paper_source / "videos/demo_0.mp4"), changed(paper_source / "metas/demo_0.txt"):
            with pytest.raises(ValueError, match="canonical demo_0 through demo_1198"):
                validate_robocasa_source(config(paper_source))


@pytest.mark.parametrize("fault", ["missing_task", "extra_task", "overlap", "gap", "wrong_count"])
def test_formal_source_rejects_invalid_task_mapping(paper_source, fault):
    path = paper_source / "data_mapping.json"
    mapping = json.loads(path.read_text())
    first, second = list(mapping)[:2]
    if fault == "missing_task":
        mapping.pop(first)
    elif fault == "extra_task":
        mapping["unknown_task"] = mapping[first]
    elif fault == "overlap":
        mapping[second] = dict(mapping[first])
    elif fault == "gap":
        mapping[first]["end_idx"] -= 1
        mapping[first]["count"] -= 1
    else:
        mapping[first]["count"] = 1
    with changed(path, json.dumps(mapping)), pytest.raises(ValueError):
        validate_robocasa_source(config(paper_source))


@pytest.mark.parametrize("folder,extension", [("videos", ".mp4"), ("metas", ".txt"), ("affordance", ".npz")])
def test_formal_source_rejects_missing_and_extra_assets(paper_source, folder, extension):
    with changed(paper_source / folder / f"demo_0{extension}"), pytest.raises(ValueError, match="episode set"):
        validate_robocasa_source(config(paper_source, p1=True))
    with changed(paper_source / folder / f"demo_1199{extension}", b"unexpected"), pytest.raises(ValueError, match="episode set"):
        validate_robocasa_source(config(paper_source, p1=True))


def test_empty_language_size_manifest_and_duplicate_json_keys_fail(paper_source):
    with changed(paper_source / "metas/demo_0.txt", "  \n"), pytest.raises(ValueError, match="instruction is empty"):
        validate_robocasa_source(config(paper_source))
    with changed(paper_source / ".required_files_manifest.json", json.dumps({"videos/demo_0.mp4": {"size": 99999}})):
        with pytest.raises(ValueError, match="incorrect-sized"):
            validate_robocasa_source(config(paper_source))
    with changed(paper_source / "norm_stats.json", '{"action_low": [], "action_low": []}'):
        with pytest.raises(ValueError, match="Duplicate"):
            read_source_json(paper_source, "norm_stats.json")


@pytest.mark.parametrize("fault", ["nan", "infinity", "reversed", "equal", "wrong_dim", "wrong_method", "delta", "missing_method"])
def test_bad_action_statistics_are_rejected(paper_source, fault):
    values = json.loads((paper_source / "norm_stats.json").read_text())
    if fault in {"nan", "infinity", "reversed", "equal"}:
        values["action_low"][0] = {"nan": float("nan"), "infinity": float("inf"), "reversed": 2, "equal": 1}[fault]
    elif fault == "wrong_dim":
        values["action_high"].pop()
    elif fault == "wrong_method":
        values["norm_method"] = "zscore"
    elif fault == "delta":
        values["use_delta_actions"] = True
    else:
        values.pop("norm_method")
    with pytest.raises(ValueError):
        validate_action_statistics(values, require_protocol=True)


def test_preflight_validates_action_values_and_label_headers(paper_source):
    actions = json.loads((paper_source / "action.json").read_text())
    actions["demo_0"][0][0] = 9.0
    with changed(paper_source / "action.json", json.dumps(actions)), pytest.raises(ValueError, match="normalization bounds"):
        validate_robocasa_source(config(paper_source))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{view: np.zeros((2, 224, 224), dtype=np.float16) for view in VIEW_NAMES})
    with changed(paper_source / "affordance/demo_0.npz", buffer.getvalue()), pytest.raises(ValueError, match="shape/dtype"):
        validate_robocasa_source(config(paper_source, p1=True))


def test_source_rejects_external_symlink_and_manifest_traversal(paper_source, tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"external source")
    target = paper_source / "videos/demo_0.mp4"
    with changed(target):
        try:
            target.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"Symlinks unavailable: {exc}")
        with pytest.raises(ValueError, match="escapes"):
            validate_robocasa_source(config(paper_source))
    for relative in ("../outside", str(outside)):
        with pytest.raises(ValueError, match="contained"):
            source_path(paper_source, relative)
    with changed(paper_source / ".required_files_manifest.json", json.dumps({"../outside": {"size": 1}})):
        with pytest.raises(ValueError, match="contained"):
            validate_robocasa_source(config(paper_source))


def test_p1_cannot_disable_required_affordance(paper_source):
    value = config(paper_source, p1=True)
    value.robocasa24_require_affordance = False
    with pytest.raises(ValueError, match="P1 requires"):
        validate_robocasa_source(value)


@pytest.mark.parametrize("online", [False, None, "true", 1])
def test_source_guard_rejects_offline_or_missing_online_contract(paper_source, online):
    value = config(paper_source)
    value.feature_cache = "/arbitrary/latent/cache"
    if online is None:
        del value.online_vjepa
    else:
        value.online_vjepa = online
    with pytest.raises(ValueError, match="online_vjepa=true"):
        validate_robocasa_source(value)


def test_actual_preflight_cli_uses_shared_full_source_guard(paper_source, tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(vars(config(paper_source))))
    result = subprocess.run([sys.executable, str(root / "scripts/check_robocasa24_files.py"), "--config", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["episodes"] == 1199
    with changed(paper_source / "videos/demo_0.mp4"):
        result = subprocess.run([sys.executable, str(root / "scripts/check_robocasa24_files.py"), "--config", str(path)], capture_output=True, text=True)
        assert result.returncode != 0 and "episode set" in result.stderr


def test_fixed_split_is_shared_by_p0_and_p1_and_rejects_alternatives(tmp_path):
    pytest.importorskip("torch")
    from thinkwam_p0.data.splits import episode_holdout_indices
    dataset = SimpleNamespace(episode_lengths={episode: 1 for episode in EPISODES},
                              index=[(episode, 0) for episode in sorted(EPISODES)])
    train, val = episode_holdout_indices(dataset.index, eval_fraction=0.1, seed=7)
    report = validate_robocasa_split(dataset, train, val, config(tmp_path))
    assert report == validate_robocasa_split(dataset, train, val, config(tmp_path, p1=True))
    assert (len(report["train_episodes"]), len(report["val_episodes"])) == (1080, 119)
    assert report["train_sha256"] == "be11d5701c35c8e02db2574fa4727ed5018913ed0255f997643348d5b931c23e"
    assert report["val_sha256"] == "45aad378d635a9f9cfa5a24afc936754a9f7c2563e9f8cbe41112d0281f81dfb"
    wrong_train, wrong_val = list(train), list(val)
    wrong_train[0], wrong_val[0] = wrong_val[0], wrong_train[0]
    for left, right in ((wrong_train, wrong_val), (train[:-1], val), (train + [train[0]], val), (train, train[:119])):
        with pytest.raises(ValueError):
            validate_robocasa_split(dataset, left, right, config(tmp_path))
    altered = config(tmp_path)
    altered.seed = 8
    with pytest.raises(ValueError, match="seed=7"):
        validate_robocasa_split(dataset, train, val, altered)
    dataset.index.append(dataset.index[0])
    with pytest.raises(ValueError, match="window index"):
        validate_robocasa_split(dataset, train, val, config(tmp_path))
