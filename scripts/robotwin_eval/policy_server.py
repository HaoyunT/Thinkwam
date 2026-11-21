"""Checkpoint-exact ThinkWAM P1 inference for the RoboTwin simulator."""
from bootstrap import bootstrap
bootstrap()
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import subprocess
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from bootstrap import configure_torch
configure_torch(torch)
import imageio_ffmpeg
from transformers import AutoTokenizer
from thinkwam_p0.data.collate import _preprocess_images
from thinkwam_p1.config import P1Config
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.models.observation_encoder import FrozenObservationEncoder
from scripts.robotwin_eval.manifest import (
    digest, identity, load_plan, paper_checkpoint_config, validate_receipt, verify_assets, write_json,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True,
                   help='checkpoint verification receipt produced by launch.sh')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    args = p.parse_args()
    args.root = args.root.resolve()
    plan = load_plan(args.root)
    validate_receipt(args.verification, plan)
    verify_assets(plan)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print('stage=loading_checkpoint_metadata', flush=True)
    checkpoint = torch.load(plan['checkpoint'], map_location='cpu',
                            weights_only=plan.get('checkpoint_profile') == 'released', mmap=True)
    validate_receipt(args.verification, plan)
    step = checkpoint['step']
    if type(step) is not int or step != plan['checkpoint_step']:
        raise ValueError('loaded checkpoint step differs from the paper plan')
    if digest(checkpoint['config']) != plan['checkpoint_config_sha256']:
        raise ValueError('loaded checkpoint config differs from the paper plan')
    released = plan.get('checkpoint_profile') == 'released'
    if released:
        from thinkwam_p1.released_checkpoint import released_config
        values = asdict(released_config(checkpoint, 'robotwin'))
    else:
        values = dict(paper_checkpoint_config(checkpoint['config']))
    for key in ['jepa_horizons', 'jepa_token_counts']:
        values[key] = tuple(values[key])
    values.update(compile_vjepa=False, compile_action_dit=False, gradient_checkpointing=False,
                  qwen_attn_implementation='sdpa')
    if plan.get('qwen_path'):
        values['qwen_path'] = plan['qwen_path']
    values['robotwin_stats_path'] = plan['assets']['stats']['path']
    values['robotwin_frame_cache'] = str(Path(plan['assets']['frame_manifest']['path']).parent)
    values['robotwin_manifest'] = Path(plan['assets']['frame_manifest']['path']).name
    config = P1Config(**values)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print('stage=constructing_model', flush=True)
    # The complete frozen encoder is in the training state dict. Avoid loading
    # the standalone V-JEPA checkpoint a second time.
    encoder = FrozenObservationEncoder(None, config.vjepa_image_size, config.vjepa_patch_size)
    model = P1Model(config, qwen_backend='qwen', vjepa_encoder=encoder)
    print('stage=loading_model_state', flush=True)
    if released:
        from thinkwam_p1.released_checkpoint import load_released_state
        load_released_state(model, checkpoint, 'robotwin')
    else:
        model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stats_path = Path(config.robotwin_stats_path)
    stats = json.loads(stats_path.read_text())
    act = stats.get('action', {}).get('default', stats.get('action', {}))
    state = stats.get('state', {}).get('default', stats.get('observation.state', {}))
    am = np.asarray(act.get('global_mean', act.get('mean')), np.float32)
    ast = np.maximum(np.asarray(act.get('global_std', act.get('std')), np.float32), 1e-6)
    sm = np.asarray(state.get('global_mean', state.get('mean')), np.float32)
    sst = np.maximum(np.asarray(state.get('global_std', state.get('std')), np.float32), 1e-6)
    assert all(v.shape == (14,) for v in [am, ast, sm, sst])
    manifest = json.loads((Path(config.robotwin_frame_cache) / config.robotwin_manifest).read_text())
    assert manifest['cameras'] == ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
    validate_receipt(args.verification, plan)
    ready = dict(**identity(plan), checkpoint=plan['checkpoint'], pid=os.getpid(),
                 gpu=os.environ['CUDA_VISIBLE_DEVICES'], cameras=manifest['cameras'],
                 stats=str(stats_path), stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                 normalization='training global z-score; raw absolute qpos actions',
                 loaded_strictly=True, ready_at=time.time())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            body = json.dumps(ready).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            try:
                start = time.monotonic()
                length = int(self.headers['Content-Length'])
                assert 0 < length < 20 * 1024**2
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                    images = data['images'].copy()
                    proprio = data['state'].astype(np.float32)
                    history_raw = data['history'].astype(np.float32)
                    instruction = str(data['instruction'].item())
                    policy_seed = int(data['policy_seed'].item())
                assert images.ndim == 4 and images.shape[0] == 3 and images.shape[-1] == 3 and images.dtype == np.uint8
                assert proprio.shape == (14,) and history_raw.ndim == 2 and history_raw.shape[1] == 14
                assert np.isfinite(proprio).all() and np.isfinite(history_raw).all()
                if images.shape[1:3] != (384, 384):
                    h, w = images.shape[1:3]
                    resized = subprocess.run([
                        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                        '-video_size', f'{w}x{h}', '-i', 'pipe:0', '-frames:v', '3',
                        '-filter_threads', '1', '-vf', 'scale=384:384', '-threads', '1',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                        input=images.tobytes(), capture_output=True, timeout=30, check=True)
                    images = np.frombuffer(resized.stdout, np.uint8).reshape(3,384,384,3).copy()
                # Zero pad in normalized coordinates exactly like the dataset.
                history = np.zeros((config.memory_action_history, 14), np.float32)
                n = min(len(history_raw), len(history))
                if n:
                    history[-n:] = (history_raw[-n:] - am) / ast
                torch.manual_seed(policy_seed)
                tokens = tokenizer(instruction, max_length=config.text_max_length,
                                   truncation=True, padding='max_length', return_tensors='pt')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    image_tensor = _preprocess_images([images], config.vjepa_image_size).to('cuda')
                    features = model.vjepa_adapter(model.vjepa(image_tensor).float())
                    prediction = model.predict(
                        features, torch.from_numpy(history)[None].to('cuda'),
                        proprio=torch.from_numpy((proprio-sm)/sst)[None].to('cuda'),
                        text_input_ids=tokens['input_ids'].to('cuda'),
                        attention_mask=tokens['attention_mask'].to('cuda').bool(),
                        steps=config.flow_steps, global_step=step,
                    )[0].float().cpu().numpy()
                assert prediction.shape == (config.action_horizon, 14) and np.isfinite(prediction).all()
                actions = prediction * ast + am
                buffer = io.BytesIO()
                np.savez(buffer, actions=actions, **identity(plan))
                body = buffer.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                print(json.dumps({'inference_seconds': time.monotonic()-start, 'checkpoint_step':step,
                                  'action_shape':list(actions.shape)}), flush=True)
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(error.encode())
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    write_json(args.output / 'policy_ready.json', ready)
    print('stage=policy_ready ' + json.dumps(ready), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
