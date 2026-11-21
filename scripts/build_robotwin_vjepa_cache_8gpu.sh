#!/usr/bin/env bash
set -euo pipefail

# Build standalone RoboTwin latent-cache diagnostics without changing source data.
# This is not the paper's Clean RGB training path. Each rank writes its own output.
ROOT="${1:-/data/thinkwam/datasets/robotwin2.0-lerobot-v3.0}"
VARIANT="${2:-clean}"
OUTPUT="${3:-/data/thinkwam/datasets/thinkwam/p0_vjepa_cache/robotwin2_clean}"
PARTS="${OUTPUT}_parts"
PYTHON="/data/thinkwam/envs/thinkwam/bin/python"
REPO="/data/thinkwam/Thinkwam"
WORLD_SIZE="${WORLD_SIZE:-8}"

mkdir -p "$PARTS"
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  mkdir -p "$PARTS/rank${rank}"
  CUDA_VISIBLE_DEVICES="$rank" setsid nohup "$PYTHON" "$REPO/scripts/build_robotwin_vjepa_cache.py" \
    --dataset-root "$ROOT" \
    --variant "$VARIANT" \
    --output "$PARTS/rank${rank}" \
    --device cuda:0 \
    --batch-size 32 \
    --shard-size 512 \
    --rank "$rank" \
    --world-size "$WORLD_SIZE" \
    </dev/null >"$PARTS/rank${rank}.log" 2>&1 &
  echo "rank=${rank} pid=$!"
done

wait
"$PYTHON" "$REPO/scripts/merge_vjepa_cache_parts.py" --parts "$PARTS" --output "$OUTPUT"
