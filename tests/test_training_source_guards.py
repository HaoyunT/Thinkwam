"""The P1 entry must validate each benchmark before choosing any data reader."""

from pathlib import Path

import pytest

from scripts import train_p1
from thinkwam_p1 import robocasa_protocol, robotwin_protocol
from thinkwam_p1.data import robocasa24, robotwin_online
from thinkwam_p1.data.synthetic import SyntheticP1Dataset


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("libero", "robocasa24", "robotwin2.0")
GUARDS = {
    "libero": (train_p1, "validate_standard_libero_source"),
    "robocasa24": (robocasa_protocol, "validate_robocasa_source"),
    "robotwin2.0": (robotwin_protocol, "validate_robotwin_training_config"),
}


class SourceRejected(ValueError):
    pass


def small_config(benchmark, online):
    config = train_p1.load_config(str(ROOT / "tests/fixtures/p1_smoke.yaml"))
    config.dataset = benchmark
    config.online_vjepa = online
    config.dataset_root = "/must-not-read/raw-demonstrations"
    config.feature_cache = "/must-not-read/unverified-feature-cache.npz"
    return config


def prohibit_real_readers(monkeypatch):
    constructed = []
    for module, name in (
        (train_p1, "P1LiberoDataset"),
        (train_p1, "FeatureCacheDataset"),
        (robocasa24, "RoboCasa24Dataset"),
        (robotwin_online, "OnlineRobotWinDataset"),
    ):
        def forbidden(*args, _name=name, **kwargs):
            constructed.append(_name)
            raise AssertionError(f"Data reader {_name} ran before source approval")

        monkeypatch.setattr(module, name, forbidden)
    return constructed


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("online", [True, False], ids=["raw-images", "feature-cache-attempt"])
def test_each_benchmark_guard_runs_before_raw_or_feature_cache_reader(monkeypatch, benchmark, online):
    config = small_config(benchmark, online)
    constructed = prohibit_real_readers(monkeypatch)
    checked = []

    def reject_source(actual_config):
        checked.append(actual_config)
        raise SourceRejected(f"Rejected {benchmark} source")

    owner, name = GUARDS[benchmark]
    monkeypatch.setattr(owner, name, reject_source)
    with pytest.raises(SourceRejected, match=benchmark):
        train_p1.build_dataset(config, smoke=False)
    assert checked == [config]
    assert constructed == []


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("online", [True, False], ids=["raw-images", "feature-cache-attempt"])
def test_smoke_uses_only_synthetic_data_without_source_access(monkeypatch, benchmark, online):
    config = small_config(benchmark, online)
    constructed = prohibit_real_readers(monkeypatch)

    def forbidden_guard(*args, **kwargs):
        raise AssertionError("Smoke must not inspect a real benchmark source")

    for owner, name in GUARDS.values():
        monkeypatch.setattr(owner, name, forbidden_guard)
    dataset = train_p1.build_dataset(config, smoke=True)
    assert isinstance(dataset, SyntheticP1Dataset)
    assert len(dataset) == max(8, config.batch_size * 2)
    assert dataset[0]["obs_features"].shape == (config.visual_token_count, config.vjepa_dim)
    assert constructed == []
