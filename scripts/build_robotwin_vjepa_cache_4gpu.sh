#!/usr/bin/env bash
set -euo pipefail

# Build standalone latent-cache diagnostics on the selected GPUs (default 4-7).
# This is not the paper's Clean RGB preparation or a RoboTwin P0 stage.
# Per-rank builders write separate outputs; merging starts after successful exits.

ROOT="${1:-/data/thinkwam/datasets/robotwin2.0-lerobot-v3.0}"
VARIANT="${2:-clean}"
OUTPUT="${3:-/data/thinkwam/datasets/thinkwam/p0_vjepa_cache/robotwin2_clean}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
WORLD_SIZE="${WORLD_SIZE:-4}"
BUILDER_BATCH_SIZE="${BUILDER_BATCH_SIZE:-32}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
PYTHON="${PYTHON:-/data/thinkwam/envs/thinkwam/bin/python}"
REPO="${REPO:-/data/thinkwam/Thinkwam}"
PARTS="${OUTPUT}_parts"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "GPU_IDS must contain WORLD_SIZE entries: GPU_IDS=$GPU_IDS WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi

mkdir -p "$PARTS"
PIDS=()
for rank in "${!GPUS[@]}"; do
  gpu="${GPUS[$rank]}"
  rank_dir="$PARTS/rank${rank}"
  mkdir -p "$rank_dir"
  log="$PARTS/rank${rank}.log"
  echo "starting rank=$rank physical_gpu=$gpu log=$log"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON" "$REPO/scripts/build_robotwin_vjepa_cache.py" \
        --dataset-root "$ROOT" \
        --variant "$VARIANT" \
        --output "$rank_dir" \
        --device cuda:0 \
        --batch-size "$BUILDER_BATCH_SIZE" \
        --shard-size "$SHARD_SIZE" \
        --rank "$rank" \
        --world-size "$WORLD_SIZE" \
        >"$log" 2>&1 &
  PIDS+=("$!")
  echo "rank=$rank pid=${PIDS[-1]}"
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more cache ranks failed; inspect $PARTS/rank*.log" >&2
  exit "$status"
fi

"$PYTHON" "$REPO/scripts/merge_vjepa_cache_parts.py" \
  --parts "$PARTS" \
  --output "$OUTPUT"
echo "cache_ready=$OUTPUT"
