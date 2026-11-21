"""Merge per-rank RoboTwin episodes into Clean, Random, and their average."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.robotwin_eval.manifest import identity, load_plan, read_records, write_json


def aggregate(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    rows = read_records(root, plan, complete=True)
    conditions = {}
    tasks = {task: {} for task in plan['tasks']}
    for condition in plan['conditions']:
        for task in plan['tasks']:
            selected = [rows[(condition, task, episode)] for episode in range(100)]
            tasks[task][condition] = dict(successes=sum(r['success'] for r in selected), episodes=100)
        successes = sum(tasks[task][condition]['successes'] for task in plan['tasks'])
        conditions[condition] = dict(successes=successes, episodes=2000, success_rate=successes/2000)
    summary = dict(
        **identity(plan), checkpoint=plan['checkpoint'], conditions=conditions,
        clean_random_average=sum(c['success_rate'] for c in conditions.values())/2,
        tasks=tasks, checkpoint_profile=plan.get('checkpoint_profile', 'paper'),
        paper_recipe_match=plan.get('checkpoint_profile', 'paper') == 'paper',
    )
    write_json(root / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    summary = aggregate(args.root)
    print(json.dumps({k: summary[k] for k in ('conditions', 'clean_random_average')}, indent=2))


if __name__ == '__main__':
    main()
