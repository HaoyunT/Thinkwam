"""Immutable identities and strict case accounting for the paper RoboTwin run.

Only checkpoint inspection imports torch. Validation and aggregation are CPU-only
and do not import the simulator or model. A launch hashes the checkpoint once;
policy processes reuse its verified file-stat receipt and check loaded metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROBOTWIN20_TASKS = (
    'adjust_bottle', 'beat_block_hammer', 'click_alarmclock', 'click_bell',
    'dump_bin_bigbin', 'grab_roller', 'handover_mic', 'lift_pot',
    'place_bread_basket', 'place_bread_skillet', 'place_burger_fries',
    'place_cans_plasticbox', 'place_empty_cup', 'place_object_basket',
    'place_shoe', 'press_stapler', 'shake_bottle_horizontally', 'shake_bottle',
    'stack_bowls_three', 'stack_bowls_two',
)
CONDITIONS = ('demo_clean', 'demo_randomized')
SCHEMA_VERSION = 1
PROTOCOL = 'thinkwam-robotwin-paper-90k-h32-r32-f4-v1'
RELEASE_PROTOCOL = 'thinkwam-robotwin-released-155k-h32-r32-f4-v1'
IDENTITY_FIELDS = (
    'schema_version', 'protocol', 'plan_sha256', 'case_manifest_sha256',
    'checkpoint_sha256', 'checkpoint_config_sha256', 'checkpoint_step',
    'action_horizon', 'replan_steps', 'flow_steps',
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def expected_cases():
    return [(condition, task, episode) for condition in CONDITIONS
            for task in ROBOTWIN20_TASKS for episode in range(100)]


def identity(plan):
    return {key: plan[key] for key in IDENTITY_FIELDS}


def validate_identity(value, plan):
    for key, expected in identity(plan).items():
        if key not in value or type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError(f'identity mismatch or missing {key}')


def paper_checkpoint_config(values):
    """Reject old/extra model features and changed scientific recipe values.

    Full current dataclass metadata is the training checkpoint contract. Only
    relocation and operational logging/loader fields may differ from the recipe.
    Unknown fields are never silently dropped before constructing the model.
    """
    from dataclasses import asdict, fields
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from thinkwam_p1.config import P1Config
    known = {field.name for field in fields(P1Config)}
    if not isinstance(values, dict) or set(values) != known:
        actual = set(values) if isinstance(values, dict) else set()
        raise ValueError(f'checkpoint config schema mismatch: missing={sorted(known-actual)}, '
                         f'unknown={sorted(actual-known)}; legacy training features are unsupported')
    recipe_path = Path(__file__).resolve().parents[2] / 'configs/paper/robotwin2.yaml'
    recipe = yaml.safe_load(recipe_path.read_text(encoding='utf-8'))
    for name in ('jepa_horizons', 'jepa_token_counts'):
        recipe[name] = tuple(recipe[name])
    expected = asdict(P1Config(**recipe))
    operational = {
        'dataset_root', 'robotwin_frame_cache', 'robotwin_stats_path',
        'vjepa_checkpoint', 'qwen_path', 'output_dir', 'num_workers',
        'dataloader_prefetch_factor', 'dataloader_persistent_workers',
        'dataloader_pin_memory', 'dataloader_worker_threads', 'compile_vjepa',
        'vjepa_compile_mode', 'compile_action_dit', 'action_dit_compile_mode',
        'checkpoint_save_limit', 'eval_every', 'eval_batches', 'tensorboard',
        'tensorboard_histogram_every', 'open_loop_eval', 'save_best_checkpoint',
    }
    different = [key for key in known - operational
                 if canonical(values[key]) != canonical(expected[key])]
    if different:
        raise ValueError(f'checkpoint differs from paper RoboTwin training recipe: {sorted(different)}')
    return values


def inspect_checkpoint(path, checkpoint_profile='paper'):
    import torch
    path = Path(path).resolve(strict=True)
    before = file_stat(path)
    checkpoint_hash = sha256_file(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=checkpoint_profile == 'released', mmap=True)
    if checkpoint_profile == 'released':
        from thinkwam_p1.released_checkpoint import validate_released_metadata
        profile = validate_released_metadata(checkpoint, 'robotwin')
        if checkpoint_hash != profile['sha256'] or before['size'] != profile['size_bytes']:
            raise ValueError('checkpoint differs from the published RoboTwin artifact')
        values = checkpoint['config']
    elif checkpoint_profile == 'paper':
        if type(checkpoint.get('step')) is not int or checkpoint['step'] != 90000:
            raise ValueError('paper evaluation requires the final step-90000 checkpoint')
        values = paper_checkpoint_config(checkpoint.get('config'))
    else:
        raise ValueError('Unknown checkpoint profile')
    config_hash = digest(values)
    if file_stat(path) != before:
        raise ValueError('checkpoint changed while computing its identity')
    return dict(checkpoint=str(path), checkpoint_sha256=checkpoint_hash,
                checkpoint_config_sha256=config_hash), values


def build_plan(checkpoint_identity, simulator, world_size=4, assets=None, *,
               checkpoint_profile='paper', qwen_path=''):
    if checkpoint_profile not in {'paper', 'released'}:
        raise ValueError('Unknown checkpoint profile')
    if type(world_size) is not int or world_size < 1 or world_size > 40:
        raise ValueError('world_size must be an integer in [1, 40]')
    plan = dict(
        schema_version=SCHEMA_VERSION, protocol=PROTOCOL, **checkpoint_identity,
        checkpoint_step=90000, simulator=str(Path(simulator).resolve()),
        embodiment='aloha-agilex', tasks=list(ROBOTWIN20_TASKS), conditions=list(CONDITIONS),
        episodes_per_task=100, planned=4000, planned_per_condition=2000,
        replan_steps=32, action_horizon=32, flow_steps=4,
        instruction_type='unseen', scene_seed_start=100000, policy_seed=0,
        world_size=world_size, expert_check=True, action_type='qpos',
        success_source='native env.eval_success', retries='none',
        policy_randomness='SHA256(policy_seed,condition,task,episode,chunk)',
        task_source='JEPA-WAM Appendix D (20-task subset)',
        report=['clean success rate', 'randomized success rate', '(clean + randomized) / 2'],
        jobs=[[condition, task] for condition in CONDITIONS for task in ROBOTWIN20_TASKS],
        case_manifest_sha256=digest(expected_cases()), assets=assets or {},
    )
    if checkpoint_profile == 'released':
        from thinkwam_p1.released_checkpoint import release_profile
        profile = release_profile('robotwin')
        if (plan['checkpoint_sha256'] != profile['sha256'] or
                plan['checkpoint_config_sha256'] != digest(profile['config'])):
            raise ValueError('released plan must bind the published checkpoint and config')
        plan.update(protocol=RELEASE_PROTOCOL, checkpoint_step=profile['step'],
                    checkpoint_profile='released', paper_recipe_match=False)
    if qwen_path:
        plan['qwen_path'] = str(Path(qwen_path).expanduser().resolve())
    plan['plan_sha256'] = digest(plan)
    return plan


def checkpoint_assets(values):
    result = {}
    paths = {'stats': Path(values['robotwin_stats_path']),
             'frame_manifest': Path(values['robotwin_frame_cache']) / values['robotwin_manifest']}
    for name, path in paths.items():
        path = path.resolve(strict=True)
        result[name] = dict(path=str(path), sha256=sha256_file(path))
    return result


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError('plan must be an object')
    for key in ('checkpoint_sha256', 'checkpoint_config_sha256', 'plan_sha256'):
        value = plan.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing or invalid {key}; legacy identity-free plans are unsupported')
    if not isinstance(plan.get('checkpoint'), str) or not Path(plan['checkpoint']).is_absolute():
        raise ValueError('checkpoint path must be absolute')
    if not isinstance(plan.get('simulator'), str) or not Path(plan['simulator']).is_absolute():
        raise ValueError('simulator path must be absolute')
    expected = build_plan({key: plan[key] for key in (
        'checkpoint', 'checkpoint_sha256', 'checkpoint_config_sha256')},
        plan['simulator'], plan.get('world_size'), plan.get('assets'),
        checkpoint_profile=plan.get('checkpoint_profile', 'paper'), qwen_path=plan.get('qwen_path', ''))
    if canonical(plan) != canonical(expected):
        raise ValueError('plan schema, protocol constants, or manifest hash mismatch')
    if set(plan['assets']) != {'stats', 'frame_manifest'}:
        raise ValueError('plan must bind normalization statistics and frame manifest')
    for asset in plan['assets'].values():
        if (set(asset) != {'path', 'sha256'} or not Path(asset['path']).is_absolute()
                or len(asset['sha256']) != 64):
            raise ValueError('invalid asset identity')
    return plan


def load_plan(root):
    return validate_plan(read_json(Path(root).resolve() / 'plan.json'))


def validate_record(row, plan):
    if not isinstance(row, dict):
        raise ValueError('episode record must be an object')
    validate_identity(row, plan)
    required = set(IDENTITY_FIELDS) | {
        'record_type', 'condition', 'task', 'episode', 'scene_seed', 'instruction',
        'success', 'steps', 'step_limit', 'seconds', 'completed_at', 'initial_rgb_sha256',
    }
    if set(row) != required:
        raise ValueError(f'episode schema mismatch: missing={sorted(required-set(row))}, '
                         f'unknown={sorted(set(row)-required)}')
    if row.get('record_type') != 'formal_episode':
        raise ValueError('only formal_episode records may contribute to the paper score')
    if type(row.get('episode')) is not int or not 0 <= row['episode'] < 100:
        raise ValueError('invalid episode index')
    if row.get('condition') not in CONDITIONS or row.get('task') not in ROBOTWIN20_TASKS:
        raise ValueError('unexpected condition/task')
    if type(row.get('success')) is not bool:
        raise ValueError('success must be a JSON boolean')
    for name in ('scene_seed', 'steps', 'step_limit'):
        if type(row.get(name)) is not int:
            raise ValueError(f'{name} must be an integer')
    if row['scene_seed'] < plan['scene_seed_start'] or not 0 <= row['steps'] <= row['step_limit']:
        raise ValueError('invalid seed or action count')
    if row['step_limit'] <= 0 or (not row['success'] and row['steps'] != row['step_limit']):
        raise ValueError('unsuccessful formal rollout must reach the native step limit')
    if not isinstance(row.get('instruction'), str) or not row['instruction']:
        raise ValueError('missing episode instruction')
    for name in ('seconds', 'completed_at'):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {name}')
    rgb_hash = row.get('initial_rgb_sha256')
    if not isinstance(rgb_hash, str) or len(rgb_hash) != 64:
        raise ValueError('missing initial observation hash')
    return row['condition'], row['task'], row['episode']


def read_records(root, plan, complete=False, rank=None):
    rows = {}
    pattern = 'parts/rank*/episodes.jsonl' if rank is None else f'parts/rank{rank}/episodes.jsonl'
    for log in sorted(Path(root).resolve().glob(pattern)):
        rank_name = log.parent.name.removeprefix('rank')
        if not rank_name.isdecimal() or not 0 <= int(rank_name) < plan['world_size']:
            raise ValueError(f'unexpected worker directory {log.parent.name}')
        rank = int(rank_name)
        jobs = {tuple(job) for job in plan['jobs'][rank::plan['world_size']]}
        for number, line in enumerate(log.read_text(encoding='utf-8').splitlines(), 1):
            try:
                row = json.loads(line)
                key = validate_record(row, plan)
                if key in rows:
                    raise ValueError(f'duplicate episode {key}; success-based selection is forbidden')
                if key[:2] not in jobs:
                    raise ValueError(f'episode is in the wrong worker shard: {key}')
                rows[key] = row
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{log}:{number}: {exc}') from exc
    for condition, task in plan['jobs']:
        prior = sorted((row for key, row in rows.items() if key[:2] == (condition, task)),
                       key=lambda row: row['episode'])
        if [row['episode'] for row in prior] != list(range(len(prior))):
            raise ValueError(f'non-contiguous resumed cases for {condition}/{task}')
        if any(b['scene_seed'] <= a['scene_seed'] for a, b in zip(prior, prior[1:])):
            raise ValueError(f'reused or reordered scene seeds for {condition}/{task}')
    expected = set(expected_cases())
    actual = set(rows)
    if actual - expected or (complete and actual != expected):
        raise ValueError(f'case set mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    return rows


def ensure_plan(root, plan):
    """An existing output is resumable only under its original full identity."""
    root = Path(root).resolve()
    validate_plan(plan)
    if (root / 'plan.json').exists():
        previous = load_plan(root)
        if canonical(previous) != canonical(plan):
            raise ValueError('output already has a different plan/checkpoint; use a new evaluation root')
        read_records(root, plan)
        return False
    if root.exists() and any(root.iterdir()):
        raise ValueError('output is not empty and has no verifiable plan; legacy results cannot be resumed')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'plan.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    return True


def verify_assets(plan):
    for name, asset in plan['assets'].items():
        if sha256_file(asset['path']) != asset['sha256']:
            raise ValueError(f'{name} content differs from the plan')


def verify_checkpoint_once(root, receipt_path):
    plan = load_plan(root)
    read_records(root, plan)
    before = file_stat(plan['checkpoint'])
    if sha256_file(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('checkpoint content differs from the plan')
    if before != file_stat(plan['checkpoint']):
        raise ValueError('checkpoint changed during verification')
    verify_assets(plan)
    receipt = dict(**identity(plan), checkpoint_stat=before)
    write_json(receipt_path, receipt)
    return plan


def validate_receipt(receipt_path, plan):
    receipt = read_json(receipt_path)
    validate_identity(receipt, plan)
    if receipt.get('checkpoint_stat') != file_stat(plan['checkpoint']):
        raise ValueError('checkpoint changed since launch verification; re-run launch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--receipt', type=Path, required=True)
    verify.add_argument('--world-size', type=int, required=True)
    ready = sub.add_parser('ready')
    ready.add_argument('--root', type=Path, required=True)
    ready.add_argument('--file', type=Path, required=True)
    ready.add_argument('--pid', type=int, required=True)
    args = p.parse_args()
    if args.command == 'verify':
        plan = load_plan(args.root)
        if args.world_size != plan['world_size']:
            raise ValueError('GPU worker count differs from immutable evaluation plan')
        verify_checkpoint_once(args.root, args.receipt)
        print(args.root.resolve())
    else:
        try:
            value = read_json(args.file)
            validate_identity(value, load_plan(args.root))
            if value.get('pid') != args.pid or value.get('loaded_strictly') is not True:
                raise ValueError('stale policy readiness file')
        except (OSError, ValueError, KeyError):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
