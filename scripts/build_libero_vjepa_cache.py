#!/usr/bin/env python3
"""Build standalone LIBERO feature-cache records from HDF5 images.

LIBERO's HDF5 files already contain ``agentview_rgb`` and
``eye_in_hand_rgb``.  This avoids a simulator replay for cache construction;
each record uses the two images at ``t`` and at ``t+7/+15/+31`` and stores the
frozen V-JEPA tokens, action chunk, history, proprioception and Qwen token ids.
This utility retains a 20-action, 64-observation-token cache format and drops
terminal windows without all requested futures. The current paper P0/P1 source
guards require online RGB instead; this cache is a separate diagnostic format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21
from thinkwam_p0.data.libero_inputs import make_image_batch, encode_pairs, uniform_pool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data/thinkwam/datasets/LIBERO")
    p.add_argument("--output", default="/data/thinkwam/datasets/thinkwam/p0_vjepa_cache/libero")
    p.add_argument("--vjepa-checkpoint", default="/data/thinkwam/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt")
    p.add_argument("--qwen-path", default="/data/thinkwam/models/Qwen2.5-0.5B")
    p.add_argument("--max-records", type=int, default=0, help="0 means all valid frames")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--text-max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64, help="V-JEPA encoding batch")
    p.add_argument("--shard-size", type=int, default=512, help="records per output NPZ shard")
    p.add_argument("--compress", action="store_true", help="use CPU zlib compression (slower, smaller files)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    root, output = Path(args.dataset_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    encoder = FrozenVJEPA21(args.vjepa_checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    horizons = (7, 15, 31)
    action_horizon = 20
    records = []
    written = 0
    pending = []
    shard_samples = []
    shard_index = 0

    def flush_shard():
        nonlocal shard_index, shard_samples
        if not shard_samples:
            return
        keys = shard_samples[0].keys()
        arrays = {key: np.stack([sample[key] for sample in shard_samples], axis=0) for key in keys}
        filename = f"shard_{shard_index:05d}.npz"
        if args.compress:
            np.savez_compressed(output / filename, **arrays)
        else:
            np.savez(output / filename, **arrays)
        records.append({"file": filename, "length": len(shard_samples)})
        shard_index += 1
        print(f"wrote {filename} ({len(shard_samples)} records)", flush=True)
        shard_samples = []

    def flush_pending():
        nonlocal written, pending
        if not pending:
            return
        obs_tokens = encode_pairs(encoder, [(item["agent"], item["wrist"]) for item in pending], device)
        future_tokens = {
            horizon: encode_pairs(encoder, [(item["future"][horizon][0], item["future"][horizon][1]) for item in pending], device)
            for horizon in horizons
        }
        for index, item in enumerate(pending):
            arrays = {
                # The full encoder emits 1,152 tokens (two 24x24 views).  A
                # 64-token uniform spatial sample defines this utility's
                # compact cache, not the current paper observation budget.
                "obs_features": uniform_pool(obs_tokens[index:index + 1], 64)[0].astype(np.float16),
                "actions": item["actions"],
                "action_history": item["action_history"],
                "text_input_ids": item["text_input_ids"],
                "attention_mask": item["attention_mask"],
                "proprio": item["proprio"],
            }
            for horizon in horizons:
                target_count = {7: 64, 15: 32, 31: 16}[horizon]
                arrays[f"future_t{horizon}"] = uniform_pool(
                    future_tokens[horizon][index:index + 1], target_count
                )[0].astype(np.float16)
            shard_samples.append(arrays)
            written += 1
            if len(shard_samples) >= args.shard_size:
                flush_shard()
        pending = []
        print(f"cached {written} records", flush=True)
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    files = sorted(root.rglob("*.hdf5"))[args.rank :: args.world_size]
    if not files:
        raise FileNotFoundError(f"No LIBERO HDF5 files under {root}")
    for hdf5_path in files:
        with h5py.File(hdf5_path, "r") as h5:
            data = h5["data"]
            try:
                language = json.loads(data.attrs["problem_info"])["language_instruction"]
            except Exception:
                language = hdf5_path.stem.replace("_demo", "").replace("_", " ")
            tokenized = tokenizer(language, max_length=args.text_max_length, truncation=True, padding="max_length", return_tensors="np")
            text_ids = tokenized["input_ids"][0].astype(np.int64)
            attention = tokenized["attention_mask"][0].astype(np.bool_)
            for demo_name in sorted(data.keys()):
                demo = data[demo_name]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                obs = demo["obs"]
                agent = np.asarray(obs["agentview_rgb"])
                wrist = np.asarray(obs["eye_in_hand_rgb"])
                ee = np.asarray(obs["ee_states"], dtype=np.float32)
                grip = np.asarray(obs["gripper_states"], dtype=np.float32)
                n = min(len(actions), len(agent), len(wrist))
                for t in range(n):
                    if t + max(horizons) >= n or t + action_horizon > n:
                        continue
                    history = np.zeros((8, actions.shape[-1]), dtype=np.float32)
                    start = max(0, t - 8)
                    history_len = t - start
                    if history_len:
                        history[-history_len:] = actions[start:t]
                    pending.append({
                        "agent": agent[t],
                        "wrist": wrist[t],
                        "future": {horizon: (agent[t + horizon], wrist[t + horizon]) for horizon in horizons},
                        "actions": actions[t : t + action_horizon],
                        "action_history": history,
                        "text_input_ids": text_ids,
                        "attention_mask": attention,
                        "proprio": np.concatenate([ee[t], grip[t]], axis=0),
                    })
                    if len(pending) >= args.batch_size:
                        flush_pending()
                    if args.max_records and written + len(pending) >= args.max_records:
                        break
                if args.max_records and written + len(pending) >= args.max_records:
                    break
        if args.max_records and written + len(pending) >= args.max_records:
            break
    flush_pending()
    flush_shard()
    if args.max_records and written > args.max_records:
        # This can only happen when the final encoding batch crossed the limit.
        written = args.max_records
        records = records[:written]
    if not records:
        raise RuntimeError("No valid records; verify HDF5 observation/action arrays")
    manifest = {
        "dataset": "libero",
        "rank": args.rank,
        "world_size": args.world_size,
        "records": records,
        "shard_size": args.shard_size,
        "horizons": list(horizons),
        "token_source": "Qwen tokenizer input ids",
        "vjepa_checkpoint": args.vjepa_checkpoint,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {written} records to {output}", flush=True)


if __name__ == "__main__":
    main()
