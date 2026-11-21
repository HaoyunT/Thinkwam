#!/usr/bin/env bash
set -euo pipefail

# One persistent policy server and one simulator worker per GPU. Run
# make_plan.py first; aggregate.py reports Clean, Random, and their average.
ROOT="${1:?evaluation root with plan.json required}"
if [ "$#" -ne 1 ]; then echo 'usage: launch.sh EVALUATION_ROOT' >&2; exit 2; fi
PYTHON="${PYTHON:-/data/thinkwam/envs/thinkwam/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-$PYTHON}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=(${GPUS:-0 1 2 3})
PORT_BASE="${PORT_BASE:-19400}"
WORLD_SIZE="${#GPUS[@]}"

# A single content hash per launch is shared by all policy processes. Their
# loaded config, step and file-stat identity must still match this receipt.
ROOT="$(cd "$ROOT" && pwd)"
RECEIPT="$(mktemp "$ROOT/checkpoint_verification.XXXXXX.json")"
pids=()
sim_pids=()
cleanup() {
  if [ "${#sim_pids[@]}" -gt 0 ]; then kill "${sim_pids[@]}" 2>/dev/null || true; fi
  if [ "${#pids[@]}" -gt 0 ]; then kill "${pids[@]}" 2>/dev/null || true; fi
  rm -f -- "$RECEIPT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON" "$HERE/manifest.py" verify --root "$ROOT" --receipt "$RECEIPT" --world-size "$WORLD_SIZE"

mkdir -p "$ROOT/logs"
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  gpu="${GPUS[$rank]}"
  port=$((PORT_BASE + rank))
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" "$HERE/policy_server.py" \
    --root "$ROOT" --verification "$RECEIPT" --port "$port" \
    --output "$ROOT/policy/rank${rank}" >"$ROOT/logs/policy${rank}.log" 2>&1 &
  pids+=("$!")
  until "$PYTHON" "$HERE/manifest.py" ready --root "$ROOT" \
    --file "$ROOT/policy/rank${rank}/policy_ready.json" --pid "${pids[-1]}"; do
    kill -0 "${pids[-1]}" 2>/dev/null || { echo "policy server rank $rank exited" >&2; exit 1; }
    sleep 5
  done
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$SIM_PYTHON" "$HERE/sim_worker.py" \
    --root "$ROOT" --rank "$rank" --world-size "$WORLD_SIZE" --port "$port" \
    >"$ROOT/logs/sim${rank}.log" 2>&1 &
  sim_pids+=("$!")
done

status=0
for pid in "${sim_pids[@]}"; do
  wait "$pid" || status=$?
done
kill "${pids[@]}" 2>/dev/null || true
if [ "$status" -ne 0 ]; then
  echo "one or more RoboTwin simulator workers failed" >&2
  exit "$status"
fi
"$PYTHON" "$HERE/aggregate.py" --root "$ROOT"
