#!/usr/bin/env python3
"""Move per-rank V-JEPA shards into one output and combine their record lists.

Metadata comes from the first part; this utility does not validate cross-rank
metadata equality or certify the resulting cache's source provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parts = Path(args.parts)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = sorted(parts.glob("rank*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed rank manifests under {parts}")
    records = []
    source_manifests = []
    for manifest_path in manifests:
        rank_dir = manifest_path.parent
        rank = rank_dir.name
        manifest = json.loads(manifest_path.read_text())
        source_manifests.append(manifest)
        for record in manifest["records"]:
            source = rank_dir / record["file"]
            target_name = f"{rank}_{record['file']}"
            target = output / target_name
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
            records.append({"file": target_name, "length": int(record["length"])})
    merged = dict(source_manifests[0])
    merged["records"] = records
    merged["shard_size"] = max(int(manifest.get("shard_size", 512)) for manifest in source_manifests)
    merged["world_size"] = len(source_manifests)
    merged.pop("rank", None)
    tmp = output / "manifest.json.tmp"
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(output / "manifest.json")
    print(f"merged {len(records)} shards into {output}")


if __name__ == "__main__":
    main()
