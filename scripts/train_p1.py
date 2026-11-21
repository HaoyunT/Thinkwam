#!/usr/bin/env python3
"""Train LIBERO P1, RoboTwin, or RoboCasa24 P1 through the shared refinement model.

Use scripts/train_paper.sh for the five appendix recipes and same-stage resumes.
Engine-only CPU smoke tests use tests/fixtures/p1_smoke.yaml with --smoke.
Closed-loop evaluation is launched separately from training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset, random_split

from scripts.train_p0 import init_distributed, reduce_mean

from thinkwam_p0.data.cache import FeatureCacheDataset
from thinkwam_p0.training.warm_start import load_p0_warm_start
from thinkwam_p1.config import P1Config
from thinkwam_p1.data.collate import collate_p1
from thinkwam_p0.data.augmentation import make_libero_subsets
from thinkwam_p1.data.libero import P1LiberoDataset
from thinkwam_p1.libero_protocol import validate_standard_libero_source
from thinkwam_p1.data.synthetic import SyntheticP1Dataset
from thinkwam_p1.models.p1 import P1Model
from thinkwam_p1.models.observation_encoder import FrozenObservationEncoder
from thinkwam_p1.types import P1Batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_loader_worker(_worker_id: int) -> None:
    # Match P0's worker policy and prevent BLAS oversubscription per worker.
    worker_threads = max(1, int(os.environ.get("THINKWAM_DATALOADER_WORKER_THREADS", "1")))
    torch.set_num_threads(worker_threads)


def load_config(path: str) -> P1Config:
    values = yaml.safe_load(Path(path).read_text()) or {}
    if values.get("mixed_precision") is False:
        values["mixed_precision"] = "no"
    for key in ("jepa_horizons", "jepa_token_counts"):
        if key in values:
            values[key] = tuple(values[key])
    return P1Config(**values)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def cosine_warmup_lambda(step: int, config: P1Config) -> float:
    warmup = max(1, int(config.max_steps * config.warmup_ratio))
    if step < warmup:
        return float(step) / warmup
    progress = min(1.0, float(step - warmup) / max(1, config.max_steps - warmup))
    floor_ratio = config.min_learning_rate / config.learning_rate
    return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))



def validate_p1_resume(state: dict, config: P1Config) -> None:
    """Reject weight-only initialization or protocol changes disguised as resume."""
    required = {"model", "optimizer", "scheduler", "step", "config"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("P1 resume requires model, optimizer, scheduler, step, and config")
    if not isinstance(state["config"], dict):
        raise ValueError("P1 resume checkpoint config must be a mapping")
    # Operational settings can change without redefining the experiment.
    # p0_checkpoint is unused on a P1 resume: the complete P1 model is restored.
    runtime = {
        "output_dir", "p0_checkpoint", "num_workers", "checkpoint_save_limit",
        "tensorboard", "tensorboard_histogram_every", "dataloader_prefetch_factor",
        "dataloader_persistent_workers", "dataloader_pin_memory", "dataloader_worker_threads",
        "compile_vjepa", "vjepa_compile_mode", "compile_action_dit",
        "action_dit_compile_mode", "gradient_checkpointing",
    }
    current, saved = vars(config), state["config"]
    differences = [
        key for key in sorted((current.keys() | saved.keys()) - runtime)
        if key not in current or key not in saved
        or json.dumps(current[key], sort_keys=True, default=str)
        != json.dumps(saved[key], sort_keys=True, default=str)
    ]
    if differences:
        raise ValueError("P1 resume must use the same stage and training protocol; differing fields: "
                         + ", ".join(differences))
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= config.max_steps:
        raise ValueError("P1 resume step is outside the configured stage budget")


def restore_p1_training_state(optimizer, scheduler, state: dict) -> int:
    """Restore the original optimizer moments and schedule without an LR restart."""
    saved_optimizer = state["optimizer"]
    if not isinstance(saved_optimizer, dict) or not isinstance(state["scheduler"], dict):
        raise ValueError("P1 resume requires complete optimizer and scheduler states")
    current_groups = [(group.get("name"), len(group["params"])) for group in optimizer.param_groups]
    saved_groups = [(group.get("name"), len(group["params"]))
                    for group in saved_optimizer.get("param_groups", [])]
    if saved_groups != current_groups:
        raise ValueError(f"P1 resume optimizer groups differ: {saved_groups} != {current_groups}")
    step = state["step"]
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("P1 resume scheduler position does not match checkpoint step")
    saved_lrs = [group["lr"] for group in saved_optimizer["param_groups"]]
    if state["scheduler"].get("_last_lr") != saved_lrs:
        raise ValueError("P1 resume optimizer and scheduler learning rates disagree")
    # Construct the scheduler first; then restore its state and optimizer LRs.
    scheduler.load_state_dict(state["scheduler"])
    optimizer.load_state_dict(saved_optimizer)
    return step


def tensorboard_metric_enabled(key: str, value, config: P1Config) -> bool:
    """Return whether a scalar is meaningful for the current P1 contract.

    The model reports zero-valued metrics for disabled affordance guidance.
    Skip these placeholders and inactive loss terms rather than logging them.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            return False
        detached = value.detach()
    else:
        try:
            detached = torch.as_tensor(value)
        except (TypeError, ValueError):
            return False
        if detached.ndim != 0:
            return False
    if not torch.is_floating_point(detached) and not torch.is_complex(detached):
        detached = detached.float()
    if key.startswith("affordance/") or key in {"loss/affordance", "affordance_gate"}:
        if not config.affordance_enabled or config.affordance_loss_weight <= 0:
            return False
    if key.startswith("schedule/affordance") and (
        not config.affordance_enabled or config.affordance_loss_weight <= 0
    ):
        return False
    if (key == "loss/video_residual" or key.startswith("video/")) and config.video_residual_loss_weight <= 0:
        return False
    if key.startswith("loss/action_dit") and config.action_loss_weight <= 0:
        return False
    if key == "loss/action_dit_first" and config.first_action_loss_weight <= 0:
        return False
    if key == "loss/action_dit_refined" and config.action_refinement_loss_weight <= 0:
        return False
    if key == "loss/jepa" or key.startswith("jepa/"):
        if config.jepa_loss_weight <= 0:
            return False
    if key in {"loss/sigreg", "jepa/sigreg"} and config.sigreg_loss_weight <= 0:
        return False
    if not bool(torch.isfinite(detached).item()):
        return False
    # Skip zero-valued scalars; record a later non-zero value if the objective
    # is enabled. This does not test whether a non-zero series is constant.
    return float(detached) != 0.0


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def p1_optimizer_param_groups(model: torch.nn.Module, config: P1Config) -> list[dict]:
    """Group parameters by their configured learning rate in staged or joint training."""
    model = unwrap(model)
    buckets = {"new_modules": [], "memory_predictor": [], "qwen_lora": [], "vjepa_adapter": []}
    if config.action_dit_learning_rate > 0:
        buckets["action_dit"] = []
    for name, parameter in model.named_parameters():
        if config.action_dit_learning_rate > 0 and name.startswith("action_dit."):
            bucket = "action_dit"
        elif name.startswith("memory.") or name.startswith("predictor."):
            bucket = "memory_predictor"
        elif name.startswith("qwen."):
            bucket = "qwen_lora"
        elif name.startswith("vjepa_adapter."):
            bucket = "vjepa_adapter"
        else:
            bucket = "new_modules"
        buckets[bucket].append(parameter)
    rates = {
        "new_modules": config.learning_rate,
        "memory_predictor": config.memory_learning_rate,
        "qwen_lora": config.qwen_lora_learning_rate,
        "vjepa_adapter": config.vjepa_adapter_learning_rate,
    }
    if config.action_dit_learning_rate > 0:
        rates["action_dit"] = config.action_dit_learning_rate
    return [
        {"params": parameters, "lr": rates[name], "initial_lr": rates[name], "name": name}
        for name, parameters in buckets.items()
        if parameters
    ]


def save_checkpoint(model, optimizer, scheduler, step: int, output: Path, config: P1Config) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"checkpoint_step_{step:07d}.pt"
    temporary = output / f"checkpoint_step_{step:07d}.pt.tmp"
    torch.save(
        {
            "step": step,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": vars(config),
        },
        temporary,
    )
    temporary.replace(final)
    checkpoints = sorted(output.glob("checkpoint_step_*.pt"))
    for old in checkpoints[:-config.checkpoint_save_limit]:
        old.unlink(missing_ok=True)
    return final


def save_best_checkpoint(model, optimizer, scheduler, step: int, output: Path, config: P1Config, metric: float) -> Path:
    """Atomically publish the checkpoint selected by the configured metric."""
    output.mkdir(parents=True, exist_ok=True)
    final = output / "best.pt"
    temporary = output / "best.pt.tmp"
    torch.save(
        {
            "step": step,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": vars(config),
            "best_metric": float(metric),
            "best_metric_name": f"eval_open_loop/{config.p1_selection_metric.removeprefix('loss/')}",
        },
        temporary,
    )
    temporary.replace(final)
    (output / "best.json").write_text(
        json.dumps(
            {
                "step": step,
                "selection_metric": f"eval_open_loop/{config.p1_selection_metric.removeprefix('loss/')}",
                "selection_value": float(metric),
            },
            indent=2,
        )
    )
    return final


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches: int,
    distributed: bool = False,
    global_step: int = 0,
    eval_seed: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    cpu_rng = torch.random.get_rng_state() if eval_seed is not None else None
    cuda_rng = torch.cuda.get_rng_state(device) if eval_seed is not None and device.type == "cuda" else None
    if eval_seed is not None:
        torch.manual_seed(int(eval_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(eval_seed))
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0
    try:
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            with unwrap(model).autocast_context():
                output = model(
                    P1Batch(**move_to_device(raw, device)),
                    metrics_only=True,
                    global_step=global_step,
                )
            for key in (
                "loss",
                "loss/action_dit",
                "loss/action_dit_first",
                "loss/action_dit_refined",
                "loss/jepa",
                "loss/sigreg",
                "loss/video_residual",
                "loss/affordance",
                "affordance_gate",
            ):
                totals.setdefault(key, torch.zeros((), device=device))
                totals[key] += output[key].detach().float()
            count += 1
    finally:
        model.train(was_training)
        if cpu_rng is not None:
            torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    if count == 0:
        raise RuntimeError("P1 validation produced zero batches")
    return {key: float(reduce_mean(value / count, distributed)) for key, value in totals.items()}


def build_dataset(config: P1Config, smoke: bool):
    if smoke:
        return SyntheticP1Dataset(config, length=max(8, config.batch_size * 2), seed=config.seed)
    if config.dataset == "libero":
        validate_standard_libero_source(config)
    elif config.dataset == "robocasa24":
        from thinkwam_p1.robocasa_protocol import validate_robocasa_source

        validate_robocasa_source(config)
    elif config.dataset == "robotwin2.0":
        from thinkwam_p1.robotwin_protocol import validate_robotwin_training_config

        validate_robotwin_training_config(config)
    if config.online_vjepa:
        if config.dataset == "robotwin2.0":
            from thinkwam_p1.data.robotwin_online import OnlineRobotWinDataset
            if config.require_affordance_targets:
                raise ValueError("RoboTwin RGB adapter has no affordance labels")
            return OnlineRobotWinDataset(
                config.dataset_root, frame_cache=config.robotwin_frame_cache,
                action_horizon=config.action_horizon, action_history=config.memory_action_history,
                horizons=config.jepa_horizons, tokenizer_path=config.qwen_path,
                text_max_length=config.text_max_length, image_size=config.vjepa_image_size,
                expected_views=config.vjepa_views, manifest_name=config.robotwin_manifest,
                stats_path=config.robotwin_stats_path,
            )
        if config.dataset == "robocasa24":
            from thinkwam_p1.data.robocasa24 import RoboCasa24Dataset

            return RoboCasa24Dataset(
                config.dataset_root,
                action_horizon=config.action_horizon,
                action_history=config.memory_action_history,
                horizons=config.jepa_horizons,
                tokenizer_path=config.qwen_path,
                text_max_length=config.text_max_length,
                require_affordance=config.robocasa24_require_affordance,
                future_target_cache=config.future_target_cache,
                future_token_counts=config.jepa_token_counts,
                vjepa_checkpoint=config.vjepa_checkpoint,
            )
        if config.dataset != "libero":
            raise ValueError("unsupported raw-image dataset: expected LIBERO, RoboTwin2.0, or RoboCasa24")
        standard = P1LiberoDataset(
            config.dataset_root,
            action_horizon=config.action_horizon,
            action_history=config.memory_action_history,
            horizons=config.jepa_horizons,
            tokenizer_path=config.qwen_path,
            text_max_length=config.text_max_length,
            affordance_key=config.affordance_target_key,
        )
        return standard
    return FeatureCacheDataset(config.feature_cache, config.jepa_horizons)


def robocasa_split_indices(dataset, config: P1Config) -> tuple[list[int], list[int], dict]:
    """Resolve a protected episode split into current full-tail window indices."""
    if config.robocasa_split_manifest:
        manifest_path = Path(config.robocasa_split_manifest).expanduser()
        payload = json.loads(manifest_path.read_text())
        split = payload.get("split_episodes")
        if not isinstance(split, dict) or set(split) != {"train", "val"}:
            raise ValueError("robocasa_split_manifest must contain split_episodes.train and .val")
        train_episodes, val_episodes = (set(split["train"]), set(split["val"]))
        if len(train_episodes) != len(split["train"]) or len(val_episodes) != len(split["val"]):
            raise ValueError("robocasa_split_manifest contains duplicate episode IDs")
        if train_episodes & val_episodes:
            raise ValueError("robocasa_split_manifest leaks episodes between train and val")
        eligible = set(dataset.episode_lengths)
        if train_episodes | val_episodes != eligible:
            raise ValueError("robocasa_split_manifest does not cover exactly the current dataset episodes")
        membership = {episode: "train" for episode in train_episodes}
        membership.update({episode: "val" for episode in val_episodes})
        train_indices = [i for i, (episode, _frame) in enumerate(dataset.index) if membership[episode] == "train"]
        val_indices = [i for i, (episode, _frame) in enumerate(dataset.index) if membership[episode] == "val"]
        return train_indices, val_indices, {
            "path": str(manifest_path),
            "train_episodes": sorted(train_episodes),
            "val_episodes": sorted(val_episodes),
        }
    from thinkwam_p0.data.splits import episode_holdout_indices

    train_indices, val_indices = episode_holdout_indices(
        dataset.index, eval_fraction=config.eval_fraction, seed=config.seed
    )
    return train_indices, val_indices, {
        "path": "",
        "train_episodes": sorted({dataset.index[i][0] for i in train_indices}),
        "val_episodes": sorted({dataset.index[i][0] for i in val_indices}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=("qwen", "tiny"), default="qwen")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        default="",
        help="Resume the same P1 stage without changing its training protocol or LR schedule.",
    )
    parser.add_argument(
        "--p0-checkpoint",
        default=None,
        help="Override config.p0_checkpoint for the P0 warm start.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-missing-affordance", action="store_true")
    args = parser.parse_args()
    if args.resume and args.p0_checkpoint is not None:
        parser.error("--resume and --p0-checkpoint cannot be combined")

    config = load_config(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.p0_checkpoint is not None:
        config.p0_checkpoint = args.p0_checkpoint
    if args.allow_missing_affordance:
        config.require_affordance_targets = False
    if not args.smoke:
        if config.dataset in {"libero", "robocasa24"} and not args.resume and not config.p0_checkpoint:
            raise ValueError("LIBERO/RoboCasa24 P1 requires its final paper P0 checkpoint")
        if config.dataset == "robotwin2.0" and config.p0_checkpoint:
            raise ValueError("RoboTwin trains in one stage without a P0 checkpoint")
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"P1 resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_p1_resume(resume_state, config)
    distributed, rank, local_rank, world_size = init_distributed()
    is_main = rank == 0
    seed_everything(config.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, args.smoke)
    val_size = max(1, int(len(dataset) * config.eval_fraction))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("P1 dataset must contain at least two records")
    split_generator = torch.Generator().manual_seed(config.seed)
    split_metadata = None
    if config.dataset == "robocasa24" and hasattr(dataset, "index"):
        train_indices, val_indices, split_metadata = robocasa_split_indices(dataset, config)
        from thinkwam_p1.robocasa_protocol import validate_robocasa_split

        validate_robocasa_split(dataset, train_indices, val_indices, config)
        train_size, val_size = len(train_indices), len(val_indices)
    else:
        permutation = torch.randperm(len(dataset), generator=split_generator).tolist()
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]
    if isinstance(dataset, P1LiberoDataset):
        train_dataset, val_dataset = make_libero_subsets(dataset, train_indices, val_indices, config)
    elif config.dataset == "robocasa24" and hasattr(dataset, "index"):
        train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    else:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=split_generator
        )
    if config.dataset == "robotwin2.0" and not args.smoke:
        from thinkwam_p1.data.robotwin_task_sampling import UniformRobotWinTaskSampler

        train_sampler = UniformRobotWinTaskSampler(
            train_dataset, num_replicas=world_size, rank=rank, seed=config.seed,
        )
    else:
        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True, seed=config.seed) if distributed else RandomSampler(train_dataset)
        )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    loader_kwargs = {
        "batch_size": config.batch_size,
        "collate_fn": collate_p1,
        "drop_last": True,
        "num_workers": 0 if args.smoke else config.num_workers,
        "pin_memory": config.dataloader_pin_memory and torch.cuda.is_available(),
        "worker_init_fn": init_loader_worker,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor
        loader_kwargs["persistent_workers"] = config.dataloader_persistent_workers
    loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    if len(loader) == 0:
        raise ValueError("P1 training split is smaller than batch_size per rank")
    val_kwargs = dict(loader_kwargs)
    val_kwargs["drop_last"] = False
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **val_kwargs)

    output_dir = config.resolved_output_dir
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if split_metadata is not None:
            (output_dir / "split_manifest.json").write_text(json.dumps({
                "schema": 1,
                "split_unit": "complete_source_episode",
                "split_episodes": {
                    "train": split_metadata["train_episodes"],
                    "val": split_metadata["val_episodes"],
                },
                "source": split_metadata["path"] or "derived_from_seed",
                "window_rule": "all current-observation starts with independent action/future masks",
            }, indent=2))

    vjepa = None
    if config.online_vjepa:
        checkpoint = config.vjepa_checkpoint if Path(config.vjepa_checkpoint).is_file() else None
        if checkpoint is None:
            raise FileNotFoundError(f"P1 online V-JEPA checkpoint not found: {config.vjepa_checkpoint}")
        vjepa = FrozenObservationEncoder(checkpoint, config.vjepa_image_size, config.vjepa_patch_size)
    model = P1Model(config, qwen_backend=args.backend, vjepa_encoder=vjepa).to(device)
    if resume_state is not None:
        missing, unexpected = model.load_state_dict(resume_state["model"], strict=True)
        if is_main:
            print(
                f"loaded_p1_resume checkpoint={args.resume} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    elif config.p0_checkpoint:
        if not Path(config.p0_checkpoint).is_file():
            raise FileNotFoundError(f"P0 warm-start checkpoint not found: {config.p0_checkpoint}")
        summary = load_p0_warm_start(model, config.p0_checkpoint, config)
        if is_main:
            print(f"loaded_p0 checkpoint={config.p0_checkpoint} {json.dumps(summary, sort_keys=True)}", flush=True)
    if config.compile_vjepa:
        model.compile_vjepa(config.vjepa_compile_mode)
    if config.compile_action_dit:
        if is_main:
            print(f"compiling_action_dit mode={config.action_dit_compile_mode}", flush=True)
        model.compile_action_dit(config.action_dit_compile_mode)
    if is_main:
        print("p1_parameter_summary", json.dumps(model.trainable_parameter_summary(), sort_keys=True), flush=True)
        print(f"world_size={world_size} per_rank_batch={config.batch_size} global_batch={config.batch_size * world_size}", flush=True)
    if distributed:
        # LIBERO P1 progressively unfreezes modules within the same training
        # stage, so DDP must handle parameters that are initially unused.
        # The bucket options below are optional DDP memory settings.
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": True,
        }
        if os.environ.get("THINKWAM_DDP_GRADIENT_AS_BUCKET_VIEW", "0") == "1":
            ddp_kwargs["gradient_as_bucket_view"] = True
        bucket_cap_mb = os.environ.get("THINKWAM_DDP_BUCKET_CAP_MB")
        if bucket_cap_mb:
            ddp_kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
        if is_main:
            print(
                "ddp_options "
                + json.dumps(
                    {
                        "gradient_as_bucket_view": ddp_kwargs.get("gradient_as_bucket_view", False),
                        "bucket_cap_mb": ddp_kwargs.get("bucket_cap_mb", "default"),
                        "worker_threads": os.environ.get("THINKWAM_DATALOADER_WORKER_THREADS", "1"),
                        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                        "mkl_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        # Register all parameters that may unfreeze during LIBERO P1 before
        # DDP wraps the model, then apply the within-stage schedule in the loop.
        model.set_stage(max(config.freeze_qwen_steps, config.freeze_vjepa_steps, config.freeze_memory_steps))
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(
        # Include initially frozen parameters so the staged schedule can open
        # them later without rebuilding AdamW's state or DDP buckets.
        p1_optimizer_param_groups(model, config),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        **({"fused": True} if torch.cuda.is_available() else {}),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_warmup_lambda(step, config))
    resume_step = 0
    if resume_state is not None:
        resume_step = restore_p1_training_state(optimizer, scheduler, resume_state)
        if is_main:
            print(f"resume_optimizer_state=loaded resume_scheduler_state=loaded step={resume_step}", flush=True)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(config), indent=2, default=list))
    writer = None
    if config.tensorboard and is_main:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output_dir / "tensorboard"))

    step = resume_step
    epoch = step // max(1, len(loader))
    best_metric = float(resume_state.get("best_metric", float("inf"))) if resume_state is not None else float("inf")
    if is_main and resume_state is not None:
        print(f"resumed_step={step} resumed_epoch={epoch}", flush=True)
    while step < config.max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        for raw in loader:
            model.train()
            unwrap(model).set_stage(step)
            batch = P1Batch(**move_to_device(raw, device))
            optimizer.zero_grad(set_to_none=True)
            with unwrap(model).autocast_context():
                output = model(batch, metrics_only=True, global_step=step)
            if not torch.isfinite(output["loss"]).all():
                raise FloatingPointError(f"nonfinite P1 loss at step {step}: {float(output['loss'])}")
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if writer is not None:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and (key.startswith("loss") or key.startswith("jepa/") or key.startswith("video/") or key.startswith("affordance/") or key.startswith("schedule/") or key.endswith("_gate")):
                        if tensorboard_metric_enabled(key, value, config):
                            writer.add_scalar(key, float(value.detach()), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                if batch.action_mask is not None:
                    writer.add_scalar("data/action_valid_ratio", float(batch.action_mask.float().mean()), step)
                if batch.future_masks is not None:
                    for horizon, mask in batch.future_masks.items():
                        writer.add_scalar(f"data/future_t{horizon}_valid_ratio", float(mask.float().mean()), step)
                for group in optimizer.param_groups:
                    writer.add_scalar(f"train/lr/{group['name']}", group["lr"], step)
            if is_main and (step == 1 or step % 100 == 0):
                print(
                    f"step={step} loss={float(output['loss']):.6f} "
                    f"first={float(output['loss/action_dit_first']):.6f} "
                    f"refined={float(output['loss/action_dit_refined']):.6f} "
                    f"jepa={float(output['loss/jepa']):.6f} "
                    f"sigreg={float(output['loss/sigreg']):.6f} "
                    f"video={float(output['loss/video_residual']):.6f} "
                    f"affordance={float(output['loss/affordance']):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
            if is_main and (step % args.save_every == 0 or step == config.max_steps):
                save_checkpoint(model, optimizer, scheduler, step, output_dir / "checkpoints", config)
            if config.open_loop_eval and config.eval_every > 0 and (step % config.eval_every == 0 or step == config.max_steps):
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    config.eval_batches,
                    distributed,
                    global_step=step,
                    eval_seed=config.seed + 1000003,
                )
                if is_main:
                    print(f"open_loop step={step} " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()), flush=True)
                if writer is not None:
                    for key, value in metrics.items():
                        if tensorboard_metric_enabled(key, value, config):
                            writer.add_scalar("eval_open_loop/" + key.removeprefix("loss/"), value, step)
                selection_key = config.p1_selection_metric
                if selection_key not in metrics:
                    raise KeyError(f"P1 selection metric {selection_key!r} missing from validation output")
                selection_value = metrics[selection_key]
                if is_main and config.save_best_checkpoint and selection_value < best_metric:
                    best_metric = selection_value
                    save_best_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step,
                        output_dir / "checkpoints",
                        config,
                        best_metric,
                    )
            if step >= config.max_steps:
                break
        epoch += 1
    if writer is not None:
        writer.close()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if is_main:
        print("p1_training_complete", output_dir, flush=True)


if __name__ == "__main__":
    main()
