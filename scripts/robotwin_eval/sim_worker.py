"""Official RoboTwin task rollouts with append-only per-episode results."""
from bootstrap import bootstrap
bootstrap()
import argparse
import copy
import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import urllib.request
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.robotwin_eval.manifest import (
    identity, load_plan, read_records, validate_identity, validate_record,
)


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())



def run_expert_check(env, episode, seed, cfg):
    """Official eval_policy.py semantics: expert exceptions reject the scene seed.

    This wrapper never surrounds the model rollout or modifies its success result.
    """
    info = None
    error = None
    try:
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **copy.deepcopy(cfg))
        info = env.play_once()
        accepted = bool(env.plan_success and env.check_success())
    except Exception as exc:
        accepted = False
        error = {'exception_type': type(exc).__name__, 'exception': str(exc),
                 'traceback': traceback.format_exc()}
    finally:
        if hasattr(env, 'scene'):
            env.close_env()
    return accepted, info, error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=4)
    p.add_argument('--port', type=int, default=18400)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--preflight-condition', choices=['demo_clean', 'demo_randomized'], default='demo_clean')
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--job-index', type=int, default=-1)
    args = p.parse_args()
    args.root = args.root.resolve()
    plan = load_plan(args.root)
    if args.world_size != plan['world_size'] or not 0 <= args.rank < args.world_size:
        p.error('rank/world-size must match the immutable plan')
    if args.stop_after < 0 or args.job_index < -1 or args.job_index >= len(plan['jobs']):
        p.error('invalid stop-after or job-index')
    out = args.root / (f'preflight/{args.preflight_condition}' if args.preflight else f'parts/rank{args.rank}')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Read our shard only after taking its lock. Other workers may append to
    # their own shards now; launch and aggregate validate the complete set.
    all_rows = {} if args.preflight else read_records(args.root, plan, rank=args.rank)
    sim = (args.root/'simulator').resolve(strict=True)
    if str(sim) != plan['simulator']:
        raise ValueError('simulator link differs from the immutable plan')
    if not args.preflight:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/', timeout=30) as response:
            ready = json.load(response)
        validate_identity(ready, plan)
        if ready.get('loaded_strictly') is not True:
            raise ValueError('policy was not loaded strictly')
    os.chdir(sim)
    sys.path[:0] = [str(sim), str(sim/'description/utils')]
    print('stage=importing_simulator', flush=True)
    import torch
    from bootstrap import configure_torch
    configure_torch(torch)
    from generate_episode_instructions import generate_episode_descriptions
    camera = yaml.safe_load((sim/'task_config/_camera_config.yml').read_text())
    robot = sim/'assets/embodiments/aloha-agilex'
    robot_config = yaml.safe_load((robot/'config.yml').read_text())
    common = dict(policy_name='thinkwam',
                    ckpt_setting=plan['checkpoint'], eval_mode=True, render_freq=0,
                    eval_video_log=False, collect_data=False, save_data=False,
                    save_path=str(out/'data'), dual_arm_embodied=True,
                    left_robot_file=str(robot), right_robot_file=str(robot),
                    left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                    head_camera_h=camera['D435']['h'], head_camera_w=camera['D435']['w'])
    jobs = plan['jobs'][args.rank::args.world_size]
    rank_planned = len(jobs) * plan['episodes_per_task']
    if args.job_index >= 0:
        assert args.job_index % args.world_size == args.rank
        jobs = [plan['jobs'][args.job_index]]
    if args.preflight:
        jobs = [(args.preflight_condition,plan['tasks'][0])]
    log = out/'episodes.jsonl'
    assigned_jobs = {tuple(job) for job in plan['jobs'][args.rank::args.world_size]}
    rows = [] if args.preflight else [row for key, row in all_rows.items() if key[:2] in assigned_jobs]
    for condition,task in jobs:
        cfg = yaml.safe_load((sim/f'task_config/{condition}.yml').read_text())
        cfg.update(common, task_config=condition, task_name=task)
        cls = getattr(importlib.import_module('envs.'+task), task)
        prior = sorted((r for r in rows if r['task']==task and r['condition']==condition), key=lambda r:r['episode'])
        assert [r['episode'] for r in prior] == list(range(len(prior)))
        seed = prior[-1]['scene_seed']+1 if prior else plan['scene_seed_start']
        episode = len(prior)
        env = cls()
        while episode < (1 if args.preflight else plan['episodes_per_task']):
            start = time.monotonic()
            atomic(out/'progress.json', dict(stage='expert_check', task=task, episode=episode,
                                           scene_seed=seed,condition=condition,updated_at=time.time()))
            print(f'stage=expert_check task={task} episode={episode} seed={seed}', flush=True)
            random.seed(seed)
            accepted, info, expert_error = run_expert_check(env, episode, seed, cfg)
            if not accepted:
                rejection = dict(condition=condition, task=task, episode=episode, seed=seed,
                                 stage='expert_check', reason='expert_exception' if expert_error else 'expert_unsuccessful',
                                 rejected_at=time.time())
                if expert_error:
                    rejection.update(expert_error)
                append(out/'rejected_seeds.jsonl', rejection)
                print('expert_seed_rejected '+json.dumps(rejection), flush=True)
                if args.preflight and expert_error and expert_error['exception_type'] in ('FileNotFoundError','UnicodeDecodeError','ModuleNotFoundError','ImportError','AttributeError'):
                    raise RuntimeError('Preflight infrastructure failure: '+expert_error['exception'])
                seed += 1
                if seed-plan['scene_seed_start'] > 10000:
                    raise RuntimeError('Too many rejected environment seeds')
                continue
            atomic(out/'progress.json', dict(stage='reset_for_policy',task=task,episode=episode,
                                           scene_seed=seed,updated_at=time.time()))
            try:
                random.seed(seed)
                env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **copy.deepcopy(cfg))
                descriptions = generate_episode_descriptions(task, [info['info']], plan['episodes_per_task'])
                assert descriptions and descriptions[0][plan['instruction_type']]
                instruction = str(np.random.choice(descriptions[0][plan['instruction_type']]))
                env.set_instruction(instruction)
                obs = env.get_obs()
                images = np.stack([obs['observation'][k]['rgb'] for k in ['head_camera','left_camera','right_camera']])
                state = np.asarray(obs['joint_action']['vector'], np.float32)
                assert state.shape == (14,) and np.isfinite(state).all()
                assert images.ndim == 4 and images.shape[0] == 3 and images.shape[-1] == 3
                assert all(float(view.std()) > 1 for view in images), 'Blank camera rendering'
                if args.preflight:
                    from PIL import Image
                    for i,name in enumerate(['head','left','right']):
                        Image.fromarray(images[i]).save(out/f'{name}.png')
                    np.savez(out/'observation.npz',images=images,state=state,instruction=instruction,
                             history=np.empty((0,14),np.float32),policy_seed=np.int64(seed))
                    atomic(out/'preflight_passed.json',dict(task=task,scene_seed=seed,
                           image_shape=list(images.shape),state=state.tolist(),
                           instruction=instruction,step_limit=env.step_lim,
                           expert_success=True,checkpoint_step=plan['checkpoint_step'],condition=condition,at=time.time()))
                    print('stage=preflight_passed',flush=True)
                    return
                history = []
                chunk = 0
                initial_sha = hashlib.sha256(images.tobytes()).hexdigest()
                while env.take_action_cnt < env.step_lim and not env.eval_success:
                    if chunk:
                        obs = env.get_obs()
                        images = np.stack([obs['observation'][k]['rgb'] for k in ['head_camera','left_camera','right_camera']])
                        state = np.asarray(obs['joint_action']['vector'],np.float32)
                    policy_seed = int.from_bytes(hashlib.sha256(
                        f"{plan['policy_seed']}:{condition}:{task}:{episode}:{chunk}".encode()).digest()[:4],'little')
                    atomic(out/'progress.json',dict(stage='policy_inference',task=task,episode=episode,
                           scene_seed=seed,actions=int(env.take_action_cnt),updated_at=time.time()))
                    buf = io.BytesIO()
                    np.savez(buf,images=images,state=state,history=np.asarray(history[-16:],np.float32).reshape(-1,14),
                             instruction=instruction,policy_seed=np.int64(policy_seed))
                    req = urllib.request.Request(f'http://127.0.0.1:{args.port}/infer',data=buf.getvalue(),
                                                  headers={'Content-Type':'application/octet-stream'})
                    with urllib.request.urlopen(req, timeout=600) as response:
                        with np.load(io.BytesIO(response.read()), allow_pickle=False) as data:
                            validate_identity({key: data[key].item() for key in identity(plan)}, plan)
                            actions = data['actions'].copy()
                    assert actions.shape == (32,14) and np.isfinite(actions).all()
                    atomic(out/'progress.json',dict(stage='simulating',task=task,episode=episode,
                           scene_seed=seed,actions=int(env.take_action_cnt),updated_at=time.time()))
                    for action in actions[:plan['replan_steps']]:
                        if env.take_action_cnt >= env.step_lim or env.eval_success:
                            break
                        previous = env.take_action_cnt
                        env.take_action(action, action_type='qpos')
                        assert env.take_action_cnt == previous+1
                        history.append(action.copy())
                    chunk += 1
                    print(json.dumps(dict(task=task,episode=episode,chunk=chunk,
                          actions=int(env.take_action_cnt),eval_success=bool(env.eval_success))),flush=True)
                success = bool(env.eval_success)
                row = dict(**identity(plan), record_type='formal_episode',
                           condition=condition,task=task,episode=episode,scene_seed=seed,instruction=instruction,
                           success=success,steps=int(env.take_action_cnt),step_limit=int(env.step_lim),
                           seconds=time.monotonic()-start,
                           initial_rgb_sha256=initial_sha,completed_at=time.time())
                validate_record(row, plan)
                append(log,row)
                rows.append(row)
                atomic(out/'summary.json',dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                       planned=rank_planned,rows=rows))
                print('episode_completed '+json.dumps(row),flush=True)
                episode += 1
                seed += 1
                if args.stop_after and len(rows)>=args.stop_after:
                    return
            finally:
                if hasattr(env,'scene'):
                    env.close_env(clear_cache=((episode+1) % cfg['clear_cache_freq']==0))
    if len(rows) == rank_planned:
        atomic(out/'completed.json',dict(episodes=len(rows),successes=sum(r['success'] for r in rows),at=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(),flush=True)
        raise
