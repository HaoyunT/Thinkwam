#!/usr/bin/env bash
set -euo pipefail

# Paper protocol: LIBERO = 40 tasks x first 50 initial states (2,000 records);
# LIBERO-Plus = all 10,030 registry variants at initial state 0. The same P1
# checkpoint is used for both, with replan 14 and four Euler steps per pass.
BENCHMARK="${1:?libero|libero-plus required}"
CHECKPOINT="${2:?checkpoint path required}"
OUTPUT="${3:?output directory required}"
shift 3
PYTHON="${PYTHON:-/data/thinkwam/envs/thinkwam/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_SIZE="${WORLD_SIZE:-8}"
REPLAN_STEPS="${REPLAN_STEPS:-14}"
PARTS="$OUTPUT/parts"

case "$BENCHMARK" in
  libero) export THINKWAM_LIBERO_ROOT="${LIBERO_ROOT:-/data/thinkwam/src/LIBERO}" ;;
  libero-plus) export THINKWAM_LIBERO_ROOT="${LIBERO_PLUS_ROOT:?set LIBERO_PLUS_ROOT to the LIBERO-Plus checkout}" ;;
  *) echo "unknown benchmark: $BENCHMARK" >&2; exit 2 ;;
esac

mkdir -p "$PARTS"
pids=()
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  mkdir -p "$PARTS/rank${rank}"
  CUDA_VISIBLE_DEVICES="$rank" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa nohup "$PYTHON" "$REPO/scripts/evaluate_p1_libero.py" \
    --checkpoint "$CHECKPOINT" \
    --benchmark "$BENCHMARK" \
    --suite all \
    --episodes 50 \
    --replan-steps "$REPLAN_STEPS" \
    --no-videos \
    --rank "$rank" \
    --world-size "$WORLD_SIZE" \
    --manifest-root "$OUTPUT" \
    --output "$PARTS/rank${rank}" "$@" \
    >"$PARTS/rank${rank}.log" 2>&1 &
  pids+=("$!")
  echo "rank=$rank pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [ "$status" -ne 0 ]; then
  echo "one or more LIBERO evaluator ranks failed" >&2
  exit "$status"
fi

exec "$PYTHON" "$REPO/scripts/merge_libero_closed_loop.py" --parts "$PARTS" --output "$OUTPUT"
