#!/usr/bin/env python3
"""Check optional RoboTwin contact-export coverage for standalone cache diagnostics.

The current paper RoboTwin recipe disables affordance guidance and does not
call this utility. Its variant selector is separate from Clean-only training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thinkwam_p0.data.robotwin import ROBOTWIN_CAMERAS, RobotWinIndex, load_affordance_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/data/thinkwam/datasets/robotwin2.0-lerobot-v3.0")
    parser.add_argument("--variant", choices=("clean", "random", "full"), default="full")
    parser.add_argument("--affordance-root", required=True)
    parser.add_argument("--cameras", default="cam_high,cam_left_wrist,cam_right_wrist")
    args = parser.parse_args()
    aliases = {camera.rsplit(".", 1)[-1]: camera for camera in ROBOTWIN_CAMERAS}
    cameras = tuple(aliases.get(item.strip(), item.strip()) for item in args.cameras.split(",") if item.strip())
    if not cameras or any(camera not in ROBOTWIN_CAMERAS for camera in cameras):
        raise ValueError(f"select RoboTwin cameras from {ROBOTWIN_CAMERAS}")
    index = RobotWinIndex(args.dataset_root, variant=args.variant)
    covered = 0
    bad_length = []
    missing = []
    for episode in index.episodes:
        maps = load_affordance_episode(args.affordance_root, episode, cameras)
        if maps is None:
            missing.append(f"{episode.variant}/{episode.episode_index}")
            continue
        lengths = {camera: int(value.shape[0]) for camera, value in maps.items()}
        if any(length < episode.length for length in lengths.values()):
            bad_length.append({
                "episode": f"{episode.variant}/{episode.episode_index}",
                "expected": episode.length,
                "actual": lengths,
            })
        else:
            covered += 1
    result = {
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "affordance_root": str(Path(args.affordance_root).expanduser()),
        "variants": sorted({episode.variant for episode in index.episodes}),
        "episodes": len(index.episodes),
        "covered_episodes": covered,
        "coverage": covered / max(1, len(index.episodes)),
        "missing_count": len(missing),
        "bad_length_count": len(bad_length),
        "cameras": list(cameras),
    }
    if missing:
        result["missing_examples"] = missing[:20]
    if bad_length:
        result["bad_length_examples"] = bad_length[:20]
    print(json.dumps(result, indent=2, sort_keys=True))
    if missing or bad_length:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
