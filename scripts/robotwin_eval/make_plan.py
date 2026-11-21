"""Write the RoboTwin 2.0 evaluation plan used by sim_worker.py.

Protocol: the fixed 20-task subset of JEPA-WAM Appendix D, 100 episodes per
task under each of the Clean and Random conditions (2,000 records per
condition), replanning every 32 actions of a 32-step chunk, success from the
native ``env.eval_success``. There are no success-based retries.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.robotwin_eval.manifest import (
    ROBOTWIN20_TASKS, CONDITIONS, build_plan, checkpoint_assets, ensure_plan,
    inspect_checkpoint,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='evaluation output directory')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-profile', choices=['paper', 'released'], default='paper')
    p.add_argument('--checkpoint-step', type=int, choices=[90000, 155000], default=None,
                   help='Optional assertion; never rewrites the stored step')
    p.add_argument('--qwen-path', default='')
    p.add_argument('--stats', default='', help='Relocated training normalization JSON')
    p.add_argument('--frame-manifest', default='', help='Relocated three-camera frame manifest')
    p.add_argument('--simulator', type=Path, required=True, help='RoboTwin 2.0 checkout')
    p.add_argument('--episodes', type=int, choices=[100], default=100)
    p.add_argument('--replan-steps', type=int, choices=[32], default=32)
    p.add_argument('--world-size', type=int, default=4)
    args = p.parse_args()
    expected_step = 155000 if args.checkpoint_profile == 'released' else 90000
    if args.checkpoint_step is not None and args.checkpoint_step != expected_step:
        p.error('--checkpoint-step disagrees with --checkpoint-profile')
    args.root = args.root.resolve()
    args.simulator = args.simulator.resolve(strict=True)
    if not args.simulator.is_dir():
        p.error('--simulator must name a RoboTwin checkout directory')
    checkpoint_identity, values = inspect_checkpoint(args.checkpoint, args.checkpoint_profile)
    asset_values = dict(values)
    if args.stats:
        asset_values['robotwin_stats_path'] = args.stats
    if args.frame_manifest:
        asset_values['robotwin_frame_cache'] = str(Path(args.frame_manifest).parent)
        asset_values['robotwin_manifest'] = Path(args.frame_manifest).name
    plan = build_plan(checkpoint_identity, args.simulator, args.world_size, checkpoint_assets(asset_values),
                      checkpoint_profile=args.checkpoint_profile, qwen_path=args.qwen_path)
    if args.checkpoint_step is not None and args.checkpoint_step != plan['checkpoint_step']:
        raise ValueError('--checkpoint-step differs from the verified checkpoint')
    ensure_plan(args.root, plan)
    link = args.root/'simulator'
    if link.exists() or link.is_symlink():
        if link.resolve() != args.simulator:
            raise ValueError('existing simulator link differs from the immutable plan')
    else:
        os.symlink(args.simulator, link, target_is_directory=True)
    print(json.dumps({k: plan[k] for k in ('planned', 'replan_steps', 'episodes_per_task')}))


if __name__ == '__main__':
    main()
