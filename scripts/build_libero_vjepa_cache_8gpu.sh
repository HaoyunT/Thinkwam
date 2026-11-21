#!/usr/bin/env bash
set -euo pipefail

# Standalone latent-cache diagnostics; current paper training requires online RGB.

ROOT="${1:-/data/thinkwam/datasets/LIBERO}"
OUTPUT="${2:-/data/thinkwam/datasets/thinkwam/p0_vjepa_cache/libero}"
PARTS="${OUTPUT}_parts"
PYTHON="/data/thinkwam/envs/thinkwam/bin/python"
REPO="/data/thinkwam/Thinkwam"
WORLD_SIZE=8

mkdir -p "$PARTS"
for rank in $(seq 0 7); do
  mkdir -p "$PARTS/rank${rank}"
  CUDA_VISIBLE_DEVICES="$rank" setsid nohup "$PYTHON" "$REPO/scripts/build_libero_vjepa_cache.py" \
    --dataset-root "$ROOT" \
    --output "$PARTS/rank${rank}" \
    --device cuda:0 \
    --batch-size 256 \
    --shard-size 512 \
    --rank "$rank" \
    --world-size "$WORLD_SIZE" \
    </dev/null >"$PARTS/rank${rank}.log" 2>&1 &
  echo "rank=${rank} pid=$!"
done

wait
"$PYTHON" "$REPO/scripts/merge_vjepa_cache_parts.py" --parts "$PARTS" --output "$OUTPUT"
