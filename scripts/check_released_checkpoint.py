#!/usr/bin/env python3
"""Verify a published checkpoint; optionally construct and strict-load the model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkwam_p1.evaluation_protocol import file_sha256, file_signature
from thinkwam_p1.released_checkpoint import release_profile, released_config, load_released_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['libero', 'libero-plus', 'robotwin', 'robocasa24'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--qwen-path', default='')
    p.add_argument('--load-model', action='store_true')
    args = p.parse_args()
    profile = release_profile(args.benchmark)
    before = file_signature(args.checkpoint)
    if args.checkpoint.stat().st_size != profile['size_bytes'] or file_sha256(args.checkpoint) != profile['sha256']:
        p.error('checkpoint content differs from the pinned published artifact')
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    config = released_config(checkpoint, args.benchmark)
    removed = []
    if args.load_model:
        if not args.qwen_path:
            p.error('--load-model requires --qwen-path')
        from thinkwam_p1.models.observation_encoder import FrozenObservationEncoder
        from thinkwam_p1.models.p1 import P1Model
        config.qwen_path = args.qwen_path
        config.qwen_attn_implementation = 'sdpa'
        config.compile_vjepa = config.compile_action_dit = False
        config.gradient_checkpointing = False
        model = P1Model(config, vjepa_encoder=FrozenObservationEncoder(None)).eval()
        removed = load_released_state(model, checkpoint, args.benchmark)
    if file_signature(args.checkpoint) != before:
        raise ValueError('checkpoint changed while verifying')
    print(json.dumps(dict(benchmark=args.benchmark, step=checkpoint['step'], sha256=profile['sha256'],
                         tensor_count=len(checkpoint['model']), metadata_verified=True,
                         strict_model_load=args.load_model, inactive_phase_tensors=removed,
                         checkpoint_profile='released', paper_recipe_match=False), indent=2))


if __name__ == '__main__':
    main()
