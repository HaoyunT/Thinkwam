#!/usr/bin/env python3
"""Legacy queue helper for optional frozen future-target caches.

This wrapper omits the builder's required --output argument and interprets an
empty future_target_cache as the working directory. Use
build_robocasa24_future_cache.py with an explicit --output instead.
"""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def idle_gpus():
    def query(argument):
        result = subprocess.run(['nvidia-smi', argument, '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    busy = {row[0].strip() for row in query('--query-compute-apps=gpu_uuid') if row}
    return [row[1].strip() for row in query('--query-gpu=index,uuid,utilization.gpu,memory.free')
            if row[1].strip() not in busy and float(row[2]) <= 5 and float(row[3]) >= 18000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.batch_size < 1:
        parser.error('poll interval and batch size must be positive')
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    output = Path(config['future_target_cache'])
    output.mkdir(parents=True, exist_ok=True)

    def status(state, **values):
        data = dict(at=time.time(), pid=os.getpid(), state=state, **values)
        temporary = output / 'queue_status.json.tmp'
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(output / 'queue_status.json')
        print(json.dumps(data), flush=True)

    with (output / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if (output / 'manifest.json').is_file():
                status('cache_already_complete')
                return
            available = idle_gpus()
            if available:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=available[0])
                command = [sys.executable, str(Path(__file__).with_name('build_robocasa24_future_cache.py')),
                           '--config', str(config_path), '--device', 'cuda:0',
                           '--batch-size', str(args.batch_size)]
                status('building', gpu_uuid=available[0], command=command)
                result = subprocess.run(command, env=env)
                status('complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
                raise SystemExit(result.returncode)
            status('waiting_for_idle_gpu', eligible_gpus=[])
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
