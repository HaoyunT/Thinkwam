#!/usr/bin/env python3
"""Validate and merge a complete manifest-bound LIBERO/Plus evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinkwam_p1.evaluation_protocol import atomic_json, canonical_json, validate_manifest
from thinkwam_p1.libero_evaluation import merge_rank_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = validate_manifest(json.loads((output / "manifest.json").read_text()))
    parts = sorted(Path(args.parts).glob("rank*/summary.json"))
    summaries = []
    for path in parts:
        shard_manifest = validate_manifest(json.loads((path.parent / "manifest.json").read_text()))
        if canonical_json(shard_manifest) != canonical_json(manifest):
            raise ValueError(f"Shard manifest differs from output manifest: {path.parent}")
        summary = json.loads(path.read_text())
        if path.parent.name != f"rank{summary.get('rank')}":
            raise ValueError(f"Shard path and rank disagree: {path}")
        summaries.append(summary)
    result = merge_rank_results(manifest, summaries)
    atomic_json(output / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("successes", "episodes", "success_rate", "world_size")}, indent=2))


if __name__ == "__main__":
    main()
