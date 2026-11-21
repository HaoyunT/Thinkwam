# Shared cache formats

Caches are derived data artifacts, not additional training recipes. The five appendix configurations in `configs/paper/` require online observation encoding and use uncached future targets by default. Their source demonstrations remain authoritative.

## Feature records

`thinkwam_p0.data.cache.FeatureCacheDataset` reads `manifest.json` and NPZ shards. Each shard entry specifies its file and record count:

```json
{"records": [{"file": "shard_00000.npz", "length": 512}]}
```

The loader accepts either shard entries or a list of individual NPZ filenames. In a sharded NPZ, each array has a leading record dimension. After selecting one record, the fields are:

| Field | Shape or meaning |
|---|---|
| `obs_features` | `[N_obs, D_v]` current visual features |
| `future_t<h>` | `[N_future, D_v]` frozen visual features for each configured horizon |
| `actions` | `[K, D_a]` action targets |
| `action_history` | `[L, D_a]` previously executed actions |
| `text_input_ids`, `attention_mask` | Tokenized instruction and validity mask |
| `text_features` | Optional precomputed instruction features |
| `proprio` | Optional benchmark-specific state |
| `affordance_targets` | Optional affordance supervision |

Dimensions are determined by the dataset and active configuration, not by fixed cache-wide action or token counts. In the paper recipes, action chunks are 16 x 7 for LIBERO/RoboCasa24 and 32 x 14 for RoboTwin, with future query budgets 288 / 144 / 72. Do not substitute an old cache's dimensions for these values.

The general feature-cache loader does not expose per-horizon or padded-action validity masks. The current paper training source guards reject this format as an alternative to online RGB. In particular, `build_libero_vjepa_cache.py` retains a separate 20-action, 64-observation-token format, and the RoboTwin feature builder has configurable dimensions and source variants; neither defines the active paper recipe. Both builders omit windows whose requested futures or action chunk would exceed the episode.

`scripts/merge_vjepa_cache_parts.py` moves per-rank shards into its output and derives the merged metadata from the first part. It does not verify equal metadata across ranks or certify source coverage. These feature-cache utilities are separate diagnostics; their output must not be treated as verified paper training data.

## RoboCasa24 frozen future targets

`thinkwam_p0.data.future_cache.FutureTargetCache` stores only the frozen future targets while the current observation remains an online RGB input. Its schema-2 manifest records:

- Source metadata and encoder hashes, preprocessing, view order, horizons, token counts, and action horizon.
- Complete episode coverage, including every real current frame.
- Per-horizon valid-window counts and source-video hashes.
- The sizes and shapes of per-episode `future_t<h>.npy` arrays.

Each array has shape `[episode_length, N_h, D_v]` and float16 storage. Targets are raw pooled features; the model applies the normalization required by each objective. Unavailable futures contain placeholders and are excluded by the dataset's horizon-validity masks. Padded action positions are independently masked.

`scripts/build_robocasa24_future_cache.py` constructs this format with an explicit output directory:

```bash
python scripts/build_robocasa24_future_cache.py \
  --config configs/paper/robocasa24_p0.yaml \
  --output /path/to/robocasa-future-targets
```

Use the direct builder command above with an explicit `--output` for a separate cache diagnostic. The reader's `future_target_cache` option identifies this directory and enables data/encoder/schema and coverage checks. Keep the option empty in the fixed paper workflow: final P0 warm-start validation and official P1 checkpoint validation compare it with the paper recipe's empty value. It also cannot change midway through a strict same-stage resume.

Neither cache format supplies task success labels at deployment. Evaluation uses the benchmark simulator's observations and native success predicate.
