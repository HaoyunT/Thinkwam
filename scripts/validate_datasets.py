#!/usr/bin/env python3
"""Check the paper training sources without starting training or changing data."""

import argparse
from pathlib import Path
import json
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_config(path: Path) -> dict:
    from thinkwam_p0.config import P0Config
    from thinkwam_p1.config import P1Config

    values = yaml.safe_load(path.read_text())
    for key in ("jepa_horizons", "jepa_token_counts"):
        if key in values:
            values[key] = tuple(values[key])
    config = (P1Config if "p1_stage" in values else P0Config)(**values)
    if config.dataset == "libero":
        from thinkwam_p1.libero_protocol import validate_standard_libero_source

        counts = validate_standard_libero_source(config)
        report = {"source_root": str(Path(config.dataset_root).resolve()),
                  "suite_tasks": counts, "demonstrations": sum(counts.values()) * 50}
    elif config.dataset == "robotwin2.0":
        from thinkwam_p1.robotwin_protocol import (
            validate_robotwin_training_config, validate_clean_rgb_cache,
        )

        validate_robotwin_training_config(config)
        evidence = validate_clean_rgb_cache(
            config.dataset_root, frame_cache=config.robotwin_frame_cache,
            manifest_name=config.robotwin_manifest, image_size=config.vjepa_image_size,
            expected_views=config.vjepa_views, stats_path=config.robotwin_stats_path,
        )
        report = {"source_root": str(evidence["source_root"]),
                  "source_files": len(evidence["source_files"]),
                  "demonstrations": len(evidence["labels"]),
                  "source_identity_sha256": evidence["source_identity_sha256"],
                  "manifest_sha256": evidence["manifest_sha256"],
                  "rgb_verification": "Each view is verified against Clean source pixels before use"}
    elif config.dataset == "robocasa24":
        from thinkwam_p1.robocasa_protocol import validate_robocasa_source

        report = validate_robocasa_source(config)
    else:
        raise ValueError(f"No appendix training source for {config.dataset}")
    return {"dataset": config.dataset, "config": str(path.resolve()), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, action="append",
                        help="Paper YAML to check; may be repeated. Defaults to all three P1 recipes.")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "configs" / "paper"
    configs = args.config or [directory / name for name in
                             ("libero_p1.yaml", "robotwin2.yaml", "robocasa24_p1.yaml")]
    for path in configs:
        print(json.dumps(check_config(path), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
