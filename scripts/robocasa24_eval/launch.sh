#!/usr/bin/env bash
set -euo pipefail

# RoboCasa24 evaluation: 24 tasks x 50 rollouts, chunk 16 / replan 16, one
# attempt per case. Workers are spread round-robin over the listed GPUs.
CHECKPOINT="${1:?checkpoint path required}"
OUTPUT="${2:?output directory required}"
shift 2
PYTHON="${PYTHON:-/data/thinkwam/envs/libero_eval/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=(${GPUS:-0 1 2 3})
RANKS="${RANKS:-8}"

mkdir -p "$OUTPUT/logs"
pids=()
for rank in $(seq 0 $((RANKS - 1))); do
  gpu="${GPUS[$((rank % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
    PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" nohup "$PYTHON" "$HERE/worker.py" \
    --checkpoint "$CHECKPOINT" --output "$OUTPUT" --rank "$rank" --ranks "$RANKS" "$@" \
    >"$OUTPUT/logs/rank${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [ "$status" -ne 0 ]; then
  echo "one or more RoboCasa24 workers failed" >&2
  exit "$status"
fi
exec "$PYTHON" "$HERE/collect.py" --output "$OUTPUT"
