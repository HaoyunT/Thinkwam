#!/usr/bin/env python3
"""Export RoboTwin contact records to per-camera affordance ``.npz`` files.

Input JSONL is produced during simulation.  One line per frame must contain
``frame_index``, ``contacts_world`` and ``cameras`` with each camera's
``height``, ``width``, ``intrinsic_cv`` and ``extrinsic_cv``.  The exporter
keeps frame order, fills missing frame indices with zero maps, and writes one
``<camera>.npz`` containing an array of shape ``[T,H,W]``.

This contact-export utility is separate from the paper RoboTwin training path,
which disables affordance guidance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from any working directory (``python scripts/...``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from thinkwam_p1.data.robotwin_affordance import frame_to_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "--input-jsonl", dest="input", required=True, type=Path,
        help="per-frame JSONL from RoboTwin simulation",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="expected episode length; missing frames are written as zeros",
    )
    parser.add_argument("--sigma", type=float, default=6.0)
    args = parser.parse_args()

    records = []
    with args.input.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                record["_line"] = line_number
                records.append(record)
    if not records:
        raise ValueError("input JSONL contains no frames")
    records.sort(key=lambda item: int(item["frame_index"]))
    observed_last = int(records[-1]["frame_index"])
    if args.num_frames is not None and args.num_frames <= observed_last:
        raise ValueError(f"--num-frames={args.num_frames} must exceed last frame index {observed_last}")
    frame_count = args.num_frames if args.num_frames is not None else observed_last + 1
    expected = list(range(frame_count))
    by_index = {int(item["frame_index"]): item for item in records}
    if len(by_index) != len(records):
        raise ValueError("duplicate frame_index in input JSONL")

    first_maps = frame_to_heatmaps(records[0], sigma=args.sigma)
    arrays = {
        camera: np.zeros((len(expected),) + heatmap.shape, dtype=np.float16)
        for camera, heatmap in first_maps.items()
    }
    for frame_index in expected:
        record = by_index.get(frame_index)
        if record is None:
            continue
        maps = frame_to_heatmaps(record, sigma=args.sigma)
        for camera, heatmap in maps.items():
            if camera not in arrays or arrays[camera].shape[1:] != heatmap.shape:
                raise ValueError(f"camera shape changed at frame {frame_index}: {camera} {heatmap.shape}")
            arrays[camera][frame_index] = np.clip(heatmap, 0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for camera, array in arrays.items():
        np.savez_compressed(args.output_dir / f"{camera}.npz", **{camera: array})
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frames": len(expected),
                "observed_frames": len(records),
                "missing_frames": len(expected) - len(records),
                "cameras": sorted(arrays),
                "sigma": args.sigma,
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps({"frames": len(expected), "cameras": sorted(arrays), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
