#!/usr/bin/env python3
"""Validate the complete RoboCasa24 paper source before training."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinkwam_p1.robocasa_protocol import validate_robocasa_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(values, dict):
        raise ValueError("Training config must be a YAML mapping")
    config = SimpleNamespace(**values)
    print(json.dumps(validate_robocasa_source(config), indent=2))
    if values.get("future_target_cache"):
        from thinkwam_p0.config import P0Config
        from thinkwam_p0.data.future_cache import FutureTargetCache
        cache = FutureTargetCache(
            values["future_target_cache"], dataset_root=config.dataset_root,
            checkpoint=values.get("vjepa_checkpoint", P0Config.vjepa_checkpoint),
            horizons=values.get("jepa_horizons", (7, 15, 31)),
            token_counts=values.get("jepa_token_counts", (288, 144, 72)),
            action_horizon=values.get("action_horizon", 16),
            feature_dim=values.get("vjepa_dim", 1024),
        )
        print(json.dumps({"future_target_cache": cache.validate_files(config.dataset_root)}, indent=2))


if __name__ == "__main__":
    main()
