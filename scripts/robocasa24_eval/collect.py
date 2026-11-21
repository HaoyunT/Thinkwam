"""Summarize a verified, complete RoboCasa24 manifest (24 x 50 cases)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.robocasa24_eval.protocol import TASK_MAX_STEPS, validate_evaluation_manifest
from thinkwam_p1.evaluation_protocol import atomic_json, read_case_results, result_identity


def collect(root):
    root = Path(root)
    manifest = validate_evaluation_manifest(json.loads((root / "manifest.json").read_text()), require_official=True)
    rows = read_case_results(root / "episodes", manifest["cases"], manifest, complete=True)
    tasks = {}
    for task in TASK_MAX_STEPS:
        selected = [row for row in rows if row["task"] == task]
        if len(selected) != 50:
            raise ValueError(f"Expected 50 completed cases for {task}")
        successes = sum(row["success"] for row in selected)
        tasks[task] = {"successes": successes, "episodes": len(selected), "success_rate": successes / len(selected)}
    successes = sum(row["success"] for row in rows)
    return {**result_identity(manifest), "official": True, "protocol": manifest["protocol"],
            "episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
            "initial_success_episodes": sum(row["initial_success"] for row in rows), "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = collect(args.output)
    atomic_json(Path(args.output) / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("episodes", "successes", "success_rate")}, indent=2))


if __name__ == "__main__":
    main()
