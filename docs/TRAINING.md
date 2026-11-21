# Training and data preparation

The public training interface consists of `scripts/train_paper.sh` and the five configurations in `configs/paper/`. Run it from the repository root. `scripts/train_p0.py` and `scripts/train_p1.py` are shared implementation modules; they do not define additional experimental stages.

## Environment and paths

Use Linux with Python 3.10+ and a compatible PyTorch/CUDA installation. The launchers use Bash, and the cache and simulator utilities use POSIX file locks. Install the package and the additional dependencies used by the paper data readers and logging:

```bash
python -m pip install -e .
python -m pip install h5py pyarrow imageio-ffmpeg decord tensorboard
```

The paper YAML files select `qwen_attn_implementation: flash_attention_2`. Install a `flash-attn` build compatible with the environment's PyTorch, CUDA toolkit/runtime, compiler, and GPU before training; installing the repository alone does not install it. Check the [FlashAttention installation requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features) for the version being installed. Once those prerequisites are available, install it in the same Python environment:

```bash
python -m pip install packaging psutil ninja
python -m pip install flash-attn --no-build-isolation
```

Evaluation also requires the corresponding LIBERO/LIBERO-Plus, RoboTwin, or RoboCasa simulator checkout and assets. LIBERO evaluation defaults to OSMesa rendering, so its environment needs working OSMesa/OpenGL libraries. Optional LIBERO video export requires `imageio`.

Configure these paths before training:

- `dataset_root`: the source demonstrations for the chosen benchmark.
- `vjepa_checkpoint` and `qwen_path`: the pretrained V-JEPA 2.1 ViT-L and Qwen2.5-0.5B files, including the Qwen tokenizer. Add or update these keys in the selected paper YAML; omitted keys inherit the defaults in `thinkwam_p0/config.py`.
- `output_dir`: an output directory dedicated to the selected recipe.
- `p0_checkpoint` in the two P1 recipes: the final checkpoint from the corresponding P0 recipe.
- RoboTwin's `robotwin_frame_cache`, `robotwin_manifest`, and `robotwin_stats_path`: the prepared three-camera RGB manifest and action/state statistics.

The launcher requires `NPROC_PER_NODE=4` to preserve the appendix global batches and rejects a different process count. `CUDA_VISIBLE_DEVICES` should name the four GPUs assigned to the run.

## Dataset preparation

### LIBERO

Use only the standard Spatial, Object, Goal, and Long suites for training, with all 40 tasks. LIBERO-Plus assets are reserved for evaluation. Both P0 and P1 use the same seed-7 90/10 window split and apply image augmentation with probability 0.5 only to the training subset. Validation is unaugmented. A sample's current and future images share the geometric transform; in P1, its contact heatmaps receive the same spatial transform at their own resolution. Color and noise changes affect images only. Pixel translations are image augmentation, not simulated camera or object motion.

Both training stages check the exact official 40-task allowlist, 50 demonstrations per task, and matching task metadata. Symlinked or hard-linked task files, HDF5 external/soft links, virtual datasets, and external raw storage are rejected. These checks validate the task layout and metadata; they do not compare every payload byte with an official release.

P1 requires contact-heatmap labels associated with the standard demonstrations. The preparation tool checks the DeVA-to-HDF5 task/demo mapping and matching frame counts before writing; it does not compare every source action or image value:

```bash
python scripts/prepare_deva_libero_affordance.py \
  --deva-root /path/to/DeVA-LIBERO \
  --libero-root /path/to/standard-LIBERO
```

Add `--write` after reviewing the validation output to attach the labels. `scripts/validate_p1_data.py` checks label-key coverage; the training reader additionally validates label dimensions and values. `scripts/audit_libero_eval_inputs.py` is a separate diagnostic for an explicitly supplied feature cache with the expected first-shard layout. It does not run with the unchanged online paper configuration, whose `feature_cache` field is empty.

### RoboTwin 2.0

Use the Clean demonstrations for the 50 training tasks, with 50 demonstrations per task. Training samples each task with probability 1/50 and samples windows uniformly within that task after the 90/10 split; validation retains the held-out window distribution. Prepare the RGB views in the same order as the configured policy: high, left wrist, right wrist.

```bash
python scripts/prepare_robotwin_rgb.py \
  --root /path/to/robotwin-clean \
  --output /path/to/robotwin-clean-rgb384 \
  --camera-keys cam_high,cam_left_wrist,cam_right_wrist
```

The preparation output directory must be empty. It creates `manifest_3view.json` and per-episode `labels.npz`; RGB files are decoded or verified when first read. Point the paper configuration to this manifest and the source's `dataset_stats.json`. The source must be a byte-identical copy of the Clean export identified by `thinkwam_p1/robotwin_clean_source.py`; renaming another dataset to `clean` cannot satisfy the check. The manifest's source root and all video references must resolve to the configured source directory. The paper recipe uses online visual encoding of these RGB observations in its single training stage, with affordance guidance disabled.

Training verifies the Clean source against the committed source-file inventory, checks all 2,500 cached action/state sequences against the source Parquet data, and rejects Random, mixed sources, or paths escaping the validated roots. Before an RGB view is consumed, its cache must have a matching verification receipt. An existing view without valid evidence is decoded from its verified Clean source and compared pixel for pixel; a mismatch stops loading. Newly generated views record the same evidence. This check runs under the cache lock and may add decoding time on first use; metadata validation alone does not certify old RGB pixels.

To populate or verify all views in advance, run `scripts/verify_robotwin_rgb.py --root /path/to/robotwin-clean --frame-cache /path/to/robotwin-clean-rgb384`. This optional CPU job may decode and inspect roughly 730 GB of RGB pixels. Existing mismatching files are rejected rather than overwritten.

### RoboCasa24

The DeVA-compatible source contains 1,199 demonstrations, three tiled RGB views, instructions, 7D actions, normalization bounds, and contact heatmaps. Its root must contain `dataset_info.json`, `data_mapping.json`, `action.json`, `norm_stats.json`, `videos/demo_*.mp4`, and `metas/demo_*.txt`; P1 also requires `affordance/demo_*.npz`. Episode names must cover `demo_0` through `demo_1198`. The reader uses a deterministic, disjoint episode split of 1,080 training and 119 validation demonstrations in both stages.

```bash
python scripts/check_robocasa24_files.py --config configs/paper/robocasa24_p0.yaml
python scripts/check_robocasa24_files.py --config configs/paper/robocasa24_p1.yaml
```

The launcher runs these checks automatically. P0 does not require affordance labels; P1 does. Actions are normalized with the released low/high statistics, and the policy uses a zero proprioception placeholder.

Both stages require all 1,199 source episodes and their complete 24-task mapping, reject external file paths and invalid normalization statistics, and check the exact seed-7 episode split. A custom split manifest must reproduce that same 1,080/119 split. Missing or invalid supervision fails explicitly; an all-zero contact heatmap remains valid.

To check the configured sources without starting training or changing the datasets:

```bash
python scripts/validate_datasets.py
# A single stage can also be checked explicitly.
python scripts/validate_datasets.py --config configs/paper/libero_p0.yaml
```

This command verifies source metadata, file boundaries and the required benchmark contracts. RoboTwin RGB verification remains enforced before each view's first use; this command does not decode the complete RGB cache.

## Stage initialization and recovery

LIBERO P1 and RoboCasa24 P1 each start from the final checkpoint of their P0 stage. RoboTwin uses one joint recipe. Step budgets are defined in the recipe configurations.

Keep the P1 `p0_checkpoint` path consistent with any change to the P0 output directory. Initialization checks the benchmark, final P0 step, recipe configuration, and complete shared weights before starting P1. A P0 warm start initializes P1; it is different from resuming the optimizer and step counter of an interrupted run within the same recipe.

```bash
bash scripts/train_paper.sh libero-p1 \
  --resume /path/to/libero_p1/checkpoint.pt
```

The same `--resume` interface applies to all five selectors. Resume with the original recipe settings and target step budget. `--save-every` controls checkpoint frequency; `--output-dir` selects the run directory. These are execution options, not new stages or alternative training schedules. Existing checkpoints do not save the data-loader cursor or every RNG state; recovery preserves the optimizer and schedule progress but does not promise the identical future minibatch sequence.

LIBERO P1 progressively unfreezes the configured warm-started modules early in training. RoboTwin and RoboCasa24 train their configured trainable modules from the start. The visual backbone remains frozen.

## Derived caches

All five paper recipes use online observation encoding, enforced by their source guards. Shared latent feature-cache utilities remain available for separate diagnostics, but their output is not accepted as a substitute for the paper RGB inputs. The RoboCasa24 reader and model also implement a frozen future-target cache for separate checks, while keeping the current observation online. The fixed paper recipes and their checkpoint gates require `future_target_cache` to remain empty. See [Cache formats](P0_DATA_CACHE.md) for the formats and their provenance, horizon, and padding contracts.
