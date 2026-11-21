#!/usr/bin/env bash
set -euo pipefail

# Paper training recipes on four A100 GPUs (one torchrun rank per GPU):
#   libero-p0      40k steps, global batch 96          -> libero-p1
#   libero-p1      40k steps, global batch 96, warm start from libero-p0 step 40k
#   robotwin       one stage, 90k steps, global batch 80
#   robocasa24-p0  20k steps, global batch 96          -> robocasa24-p1
#   robocasa24-p1  40k steps, global batch 80, warm start from robocasa24-p0 step 20k
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${1:-}"
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
usage() {
  echo "usage: $0 libero-p0|libero-p1|robotwin|robocasa24-p0|robocasa24-p1 [--resume CHECKPOINT] [--output-dir DIRECTORY] [--save-every STEPS] [--p0-checkpoint CHECKPOINT]"
  echo "--resume continues the selected stage with its saved optimizer and LR schedule."
  echo "--p0-checkpoint is only for a fresh libero-p1 or robocasa24-p1 stage."
  echo "The appendix protocol requires four training processes; engine smoke tests are separate."
}
fail() { echo "$*" >&2; exit 2; }
if [[ "$RECIPE" == "--help" || "$RECIPE" == "-h" ]]; then usage; exit 0; fi
case "${RECIPE}" in
  libero-p0)    engine=train_p0.py; config=libero_p0.yaml; fixed=(--backend qwen) ;;
  libero-p1)    engine=train_p1.py; config=libero_p1.yaml; fixed=(--backend qwen) ;;
  robotwin)     engine=train_p1.py; config=robotwin2.yaml; fixed=(--backend qwen) ;;
  robocasa24-p0) engine=train_p0.py; config=robocasa24_p0.yaml; fixed=(--backend qwen) ;;
  robocasa24-p1) engine=train_p1.py; config=robocasa24_p1.yaml; fixed=(--backend qwen) ;;
  *) usage >&2; exit 2 ;;
esac
shift
[[ "$NPROC_PER_NODE" == 4 ]] || fail "The appendix recipes require NPROC_PER_NODE=4 to preserve the global batch size."

seen_options=""
options=()
while (($#)); do
  option="${1%%=*}"
  case "$option" in
    --resume|--output-dir|--save-every|--p0-checkpoint) ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unsupported appendix-training option: $1" ;;
  esac
  [[ " $seen_options " != *" $option "* ]] || fail "Duplicate option: $option"
  if [[ "$1" == *=* ]]; then
    value="${1#*=}"
    shift
  else
    (($# >= 2)) || fail "Missing value for $option"
    value="$2"
    shift 2
  fi
  [[ -n "$value" && "$value" != --* ]] || fail "Missing value for $option"
  if [[ "$option" == --save-every ]]; then
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "--save-every must be a positive integer"
  fi
  if [[ "$option" == --p0-checkpoint ]]; then
    [[ "$RECIPE" == libero-p1 || "$RECIPE" == robocasa24-p1 ]] || fail "--p0-checkpoint is not valid for $RECIPE"
  fi
  seen_options="$seen_options $option"
  options+=("$option" "$value")
done
if [[ " $seen_options " == *" --resume "* && " $seen_options " == *" --p0-checkpoint "* ]]; then
  fail "--resume and --p0-checkpoint cannot be combined"
fi
config="${REPO}/configs/paper/${config}"
if [[ "$RECIPE" == robocasa24-* ]]; then
  "${PYTHON}" "${REPO}/scripts/check_robocasa24_files.py" --config "$config"
fi
launch_command=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${REPO}/scripts/${engine}" --config "$config" "${fixed[@]}")
if [[ -n "$seen_options" ]]; then
  launch_command+=("${options[@]}")
fi
exec "${launch_command[@]}"
