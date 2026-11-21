# Released inference checkpoints

[Weights](https://huggingface.co/Adrain11/ThinkWAM) · [Project page](https://thinkwam-project.github.io/)

Use `--checkpoint-profile released` for the three published checkpoints.
The loader validates their configuration, SHA256, and model state against
model repository revision `5d6a0046c8498662a0d38ddf3c8fa67f7d281063`.

The checkpoints correspond to the paper's simulation results:

| Benchmark | Paper results | File |
|---|---|---|
| LIBERO / LIBERO-Plus | Tables 1 and 2 | `libero_thinkwam_checkpoint.pt` |
| RoboTwin 2.0 | Table 3 | `robotwin2.0_thinkwam_checkpoint.pt` |
| RoboCasa24 | Table 4 | `robocasa24_thinkwam_checkpoint.pt` |

See the [data preparation guide](TRAINING.md) for dataset requirements.

## Setup and verification

Use Python 3.10+ and install the model dependencies (tested CPU stack in
`requirements-release.txt`; install an appropriate CUDA PyTorch build for GPU
inference). Install simulator dependencies as described in the
[setup guide](TRAINING.md#environment-and-paths). Download the selected policy
and Qwen2.5-0.5B model/tokenizer:

```bash
python -m pip install -r requirements-release.txt
python -m pip install -e .
hf download Adrain11/ThinkWAM libero_thinkwam_checkpoint.pt \
  --revision 5d6a0046c8498662a0d38ddf3c8fa67f7d281063 --local-dir checkpoints
hf download Qwen/Qwen2.5-0.5B --local-dir backbones/Qwen2.5-0.5B
python scripts/check_released_checkpoint.py --benchmark libero \
  --checkpoint checkpoints/libero_thinkwam_checkpoint.pt
# Also construct the real model and strict-load every parameter:
python scripts/check_released_checkpoint.py --benchmark libero \
  --checkpoint checkpoints/libero_thinkwam_checkpoint.pt \
  --qwen-path backbones/Qwen2.5-0.5B --load-model
```

Change the benchmark and filename for RoboTwin/RoboCasa24. The standalone
V-JEPA checkpoint is not required for evaluation: its complete trained encoder
is embedded in each policy. Qwen's local model directory is still needed by the
constructor and tokenizer. `sdpa` is the portable evaluation attention default;
LIBERO optionally accepts `--attention flash_attention_2`.

## Evaluation commands

Prepare the original benchmark assets and normalization statistics. Paths below
are placeholders for your installation; do not substitute new normalization
statistics for those used by the checkpoint.

| Benchmark | Evaluation schedule |
|---|---|
| LIBERO | 40 tasks × first 50 initial states (2,000 episodes) |
| LIBERO-Plus | All 10,030 registry variants at initial state 0; unweighted mean across seven perturbation categories |
| RoboTwin 2.0 | Fixed 20-task subset × 100 episodes for each of Clean and Random (4,000 episodes) |
| RoboCasa24 | 24 tasks × 50 rollouts (1,200 episodes) |

```bash
# LIBERO
LIBERO_ROOT=/path/to/LIBERO PYTHON=python WORLD_SIZE=4 \
  bash scripts/run_p1_libero_closed_loop.sh libero \
  checkpoints/libero_thinkwam_checkpoint.pt results/libero-released \
  --checkpoint-profile released --qwen-path backbones/Qwen2.5-0.5B

# LIBERO-Plus: use the same LIBERO checkpoint.
LIBERO_PLUS_ROOT=/path/to/LIBERO-Plus PYTHON=python WORLD_SIZE=4 \
  bash scripts/run_p1_libero_closed_loop.sh libero-plus \
  checkpoints/libero_thinkwam_checkpoint.pt results/libero-plus-released \
  --checkpoint-profile released --qwen-path backbones/Qwen2.5-0.5B

# RoboCasa24
PYTHON=python RANKS=4 GPUS='0 1 2 3' \
  bash scripts/robocasa24_eval/launch.sh \
  checkpoints/robocasa24_thinkwam_checkpoint.pt results/robocasa-released \
  --checkpoint-profile released --qwen-path backbones/Qwen2.5-0.5B \
  --norm-stats /path/to/robocasa24/norm_stats.json

# RoboTwin
python scripts/robotwin_eval/make_plan.py --root results/robotwin-released \
  --checkpoint checkpoints/robotwin2.0_thinkwam_checkpoint.pt \
  --checkpoint-profile released --simulator /path/to/RoboTwin \
  --qwen-path backbones/Qwen2.5-0.5B \
  --stats /path/to/training/stats.json \
  --frame-manifest /path/to/training/manifest.json --world-size 4
PYTHON=python SIM_PYTHON=/path/to/robotwin-env/bin/python GPUS='0 1 2 3' \
  bash scripts/robotwin_eval/launch.sh results/robotwin-released
```

Output manifests bind the checkpoint hash and configuration. Use a new output
directory when changing weights, runtime paths, or protocol. Resuming preserves
completed episodes; aggregation requires the complete planned case set.

## Compatibility boundary and validation

Only the pinned released configurations are accepted. LIBERO's eight
`phase_head`/`phase_to_memory` tensors are excluded because
`phase_memory_max_scale` is zero.
All remaining tensor names and shapes must strict-load. RoboTwin and RoboCasa24
have no phase head. Unknown configs, active phase fusion, missing core tensors,
changed shapes and mismatched artifact hashes are rejected.

The regression fixtures are names/shapes/dtypes extracted from the published
checkpoint ZIP metadata; they contain no weight values. Tests instantiate the
full Qwen + V-JEPA + P1 architectures on PyTorch's meta device and check every
parameter. CPU tests cover the two-pass model and evaluation accounting. These
checks do not substitute for CUDA/simulator rollouts or reproduce paper scores.

```bash
python -m pytest -q tests/test_released_checkpoint.py \
  tests/test_evaluation_protocol.py tests/test_robotwin_eval_protocol.py
```
