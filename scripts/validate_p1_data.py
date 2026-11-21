#!/usr/bin/env python3
"""Check P1 contact-heatmap coverage in LIBERO demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--key", default="affordance")
    args = parser.parse_args()
    files = sorted(Path(args.root).rglob("*.hdf5"))
    records = []
    for path in files:
        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                continue
            for demo_name in data:
                demo = data[demo_name]
                found = None
                if args.key in demo:
                    found = f"data/{demo_name}/{args.key}"
                elif "obs" in demo and args.key in demo["obs"]:
                    found = f"data/{demo_name}/obs/{args.key}"
                records.append({"file": str(path), "demo": demo_name, "path": found})
    summary = {
        "files": len(files),
        "demos": len(records),
        "with_affordance": sum(item["path"] is not None for item in records),
        "key": args.key,
    }
    print(json.dumps(summary, indent=2))
    if records and summary["with_affordance"] != summary["demos"]:
        raise SystemExit("P1 affordance labels are missing for one or more demos")
    if not records:
        raise SystemExit("no HDF5 demos found")


if __name__ == "__main__":
    main()

