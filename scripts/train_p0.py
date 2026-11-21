#!/usr/bin/env python3
"""Train the LIBERO or RoboCasa24 P0 stage with its configured input adapter.

Supports one process and ``torchrun`` DDP. ``--resume`` restores the original
stage's model, optimizer, scheduler and global step without restarting warmup.
Validation is open-loop on the configured held-out split.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset, random_split

from thinkwam_p0.config import P0Config
from thinkwam_p0.data.adapters import make_dataset_source
from thinkwam_p0.data.cache import FeatureCacheDataset
from thinkwam_p0.data.collate import collate_p0
from thinkwam_p0.data.libero_online import OnlineLiberoDataset
from thinkwam_p0.data.augmentation import make_libero_subsets
from thinkwam_p1.libero_protocol import validate_standard_libero_source
from thinkwam_p0.data.splits import episode_holdout_indices
from thinkwam_p0.models.p0 import P0Model
from thinkwam_p0.models.vjepa_encoder import FrozenVJEPA21
from thinkwam_p0.types import P0Batch
from thinkwam_p0.training.checkpoint import (
    restore_p0_training_state,
    validate_p0_resume_checkpoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_loader_worker(_worker_id: int) -> None:
    # Prevent eight workers per rank from each spawning a full BLAS thread pool.
    torch.set_num_threads(1)


def load_config(path: str) -> P0Config:
    values = yaml.safe_load(Path(path).read_text()) or {}
    if values.get("mixed_precision") is False:
        values["mixed_precision"] = "no"
    for key in ("jepa_horizons", "jepa_token_counts"):
        if key in values:
            values[key] = tuple(values[key])
    return P0Config(**values)


def move_to_device(value, device):
    """Recursively move nested training batches, including future features/images."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-process training requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


def configure_attention_backends() -> None:
    """Enable CUDA SDPA kernels used by fused QKV ActionDiT attention."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value /= torch.distributed.get_world_size()
    return value


def prune_checkpoints(output: Path, limit: int) -> None:
    checkpoints = sorted(output.glob("checkpoint_step_*.pt"))
    for path in checkpoints[:-limit]:
        path.unlink(missing_ok=True)


def save_checkpoint(model, optimizer, scheduler, step: int, output: Path, config: P0Config, *, best_metric=None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"checkpoint_step_{step:07d}.pt"
    tmp = output / f"checkpoint_step_{step:07d}.pt.tmp"
    state = {key: value for key, value in unwrap(model).state_dict().items() if not key.startswith("vjepa.")}
    torch.save({
        "step": step,
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(config),
        "best_metric": best_metric,
    }, tmp)
    tmp.replace(final)
    prune_checkpoints(output, config.checkpoint_save_limit)
    return final


def save_best(model, optimizer, scheduler, step: int, output: Path, config: P0Config, metric: float) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    final = output / "best.pt"
    tmp = output / "best.pt.tmp"
    state = {key: value for key, value in unwrap(model).state_dict().items() if not key.startswith("vjepa.")}
    torch.save({
        "step": step,
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(config),
        "best_metric": metric,
        "best_metric_name": "eval_open_loop/loss",
    }, tmp)
    tmp.replace(final)
    (output / "best.json").write_text(json.dumps({
        "step": step,
        "selection_metric": "eval_open_loop/loss",
        "open_loop_loss": metric,
    }, indent=2))
    return final


@torch.no_grad()
def evaluate_open_loop(model, loader, device, distributed: bool, max_batches: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0
    try:
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            batch = P0Batch(**move_to_device(raw, device))
            with unwrap(model).autocast_context():
                output = model(batch)
            for key in ("loss", "loss/action_dit", "loss/jepa", "loss/sigreg"):
                totals.setdefault(key, torch.zeros((), device=device))
                totals[key] += output[key].detach().float()
            count += 1
        if count == 0:
            raise RuntimeError("open-loop validation produced zero batches")
        return {key: float(reduce_mean(value / count, distributed)) for key, value in totals.items()}
    finally:
        model.train(was_training)


def cosine_warmup_lambda(step: int, config: P0Config) -> float:
    warmup = max(1, int(config.max_steps * config.warmup_ratio))
    if step < warmup:
        return float(step) / warmup
    progress = min(1.0, float(step - warmup) / max(1, config.max_steps - warmup))
    floor_ratio = config.min_learning_rate / config.learning_rate
    return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def tensorboard_metric_enabled(key: str, value, config: P0Config) -> bool:
    """Skip inactive objectives and zero-valued scalar metrics in TensorBoard."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            return False
        scalar = value.detach()
    else:
        try:
            scalar = torch.as_tensor(value)
        except (TypeError, ValueError):
            return False
        if scalar.ndim != 0:
            return False
    if key == "loss/action_dit" and config.action_loss_weight <= 0:
        return False
    if key in {"loss/jepa", "jepa/total"} and config.jepa_loss_weight <= 0:
        return False
    if key in {"loss/sigreg", "jepa/sigreg"} and config.sigreg_loss_weight <= 0:
        return False
    if not torch.isfinite(scalar).item():
        return False
    return float(scalar) != 0.0


def log_latent_distributions(writer, output: dict, step: int) -> None:
    """Write z_t histograms and collapse indicators for each JEPA horizon."""
    predictions = output.get("latent_predictions", {})
    targets = output.get("latent_targets", {})
    for horizon, prediction in predictions.items():
        pred = prediction.detach().float()
        pred_flat = pred.reshape(-1)
        pred_norm = F.normalize(pred, dim=-1)
        pred_examples = pred.reshape(-1, pred.shape[-1])
        pred_feature_std = pred_examples.std(dim=0, unbiased=False).mean()
        pred_norm_feature_std = pred_norm.reshape(-1, pred_norm.shape[-1]).std(dim=0, unbiased=False).mean()
        pred_token_norm = pred.norm(dim=-1)
        tag = f"latent/z_t{horizon}"
        writer.add_histogram(f"{tag}/pred", pred_flat, step)
        writer.add_histogram(f"{tag}/pred_normalized", pred_norm.detach().reshape(-1), step)
        writer.add_scalar(f"{tag}/pred_mean", pred_flat.mean(), step)
        writer.add_scalar(f"{tag}/pred_std", pred_flat.std(unbiased=False), step)
        writer.add_scalar(f"{tag}/pred_feature_std", pred_feature_std, step)
        writer.add_scalar(f"{tag}/pred_normalized_feature_std", pred_norm_feature_std, step)
        writer.add_scalar(f"{tag}/pred_token_norm_mean", pred_token_norm.mean(), step)
        writer.add_scalar(f"{tag}/pred_token_norm_std", pred_token_norm.std(unbiased=False), step)
        if horizon in targets:
            target = targets[horizon].detach().float()
            writer.add_histogram(f"{tag}/target_normalized", target.reshape(-1), step)
            writer.add_scalar(f"{tag}/target_feature_std", target.reshape(-1, target.shape[-1]).std(dim=0, unbiased=False).mean(), step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=("qwen", "tiny"), default="qwen")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume", default="",
        help="Resume the same P0 stage, including optimizer, scheduler and global step.",
    )
    args = parser.parse_args()
    distributed, rank, local_rank, _ = init_distributed()
    configure_attention_backends()
    config = load_config(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"P0 resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_p0_resume_checkpoint(resume_state, vars(config))
    seed_everything(config.seed + rank)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    source = make_dataset_source(config.dataset, config.dataset_root)
    if is_main:
        print("dataset_source", json.dumps(source.validate(), sort_keys=True), flush=True)
    if config.dataset == "libero":
        validate_standard_libero_source(config)
    elif config.dataset == "robocasa24":
        from thinkwam_p1.robocasa_protocol import validate_robocasa_source

        report = validate_robocasa_source(config)
        if is_main:
            print("training_source", json.dumps(report, sort_keys=True), flush=True)
    if config.online_vjepa:
        if config.dataset == "libero":
            dataset = OnlineLiberoDataset(
                config.dataset_root,
                action_horizon=config.action_horizon,
                action_history=config.memory_action_history,
                horizons=config.jepa_horizons,
                tokenizer_path=config.qwen_path,
                text_max_length=config.text_max_length,
            )
        elif config.dataset == "robocasa24":
            # Reuse the DeVA-compatible RGB/action adapter. P0 ignores the
            # optional affordance target, while P1 uses the same samples.
            from thinkwam_p1.data.robocasa24 import RoboCasa24Dataset

            dataset = RoboCasa24Dataset(
                config.dataset_root,
                action_horizon=config.action_horizon,
                action_history=config.memory_action_history,
                horizons=config.jepa_horizons,
                tokenizer_path=config.qwen_path,
                text_max_length=config.text_max_length,
                require_affordance=False,
                future_target_cache=config.future_target_cache,
                future_token_counts=config.jepa_token_counts,
                vjepa_checkpoint=config.vjepa_checkpoint,
            )
        else:
            raise ValueError(
                "online_vjepa supports the LIBERO and RoboCasa24 adapters"
            )
    else:
        dataset = FeatureCacheDataset(config.feature_cache, config.jepa_horizons)
    val_size = max(1, int(len(dataset) * config.eval_fraction))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("feature cache must contain at least two records for train/validation")
    split_generator = torch.Generator().manual_seed(config.seed)
    if config.dataset == "robocasa24" and hasattr(dataset, "index"):
        train_indices, val_indices = episode_holdout_indices(
            dataset.index, eval_fraction=config.eval_fraction, seed=config.seed
        )
        from thinkwam_p1.robocasa_protocol import validate_robocasa_split

        validate_robocasa_split(dataset, train_indices, val_indices, config)
        train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
        if config.dataset == "libero":
            train_dataset, val_dataset = make_libero_subsets(
                dataset, train_dataset.indices, val_dataset.indices, config
            )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed) if distributed else RandomSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    loader_kwargs = {
        "batch_size": config.batch_size,
        "collate_fn": collate_p0,
        "drop_last": True,
        "num_workers": config.num_workers,
        "pin_memory": config.dataloader_pin_memory and torch.cuda.is_available(),
        "worker_init_fn": init_loader_worker,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor
        loader_kwargs["persistent_workers"] = config.dataloader_persistent_workers
    loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_kwargs = dict(loader_kwargs)
    val_kwargs["drop_last"] = False
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **val_kwargs)

    vjepa = FrozenVJEPA21(config.vjepa_checkpoint) if Path(config.vjepa_checkpoint).is_file() else None
    if config.online_vjepa and vjepa is None:
        raise FileNotFoundError(f"online_vjepa requires a V-JEPA checkpoint: {config.vjepa_checkpoint}")
    model = P0Model(config, qwen_backend=args.backend, vjepa_encoder=vjepa).to(device)
    if config.compile_vjepa:
        if not config.online_vjepa or vjepa is None:
            raise ValueError("compile_vjepa requires online_vjepa with a V-JEPA checkpoint")
        if is_main:
            print(f"compiling_vjepa mode={config.vjepa_compile_mode}", flush=True)
        model.compile_vjepa(config.vjepa_compile_mode)
    if config.compile_action_dit:
        if is_main:
            print(f"compiling_action_dit mode={config.action_dit_compile_mode}", flush=True)
        model.compile_action_dit(config.action_dit_compile_mode)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        **({"fused": True} if torch.cuda.is_available() else {}),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_warmup_lambda(step, config))
    step = 0
    best_open_loop_loss = float("inf")
    if resume_state is not None:
        step, best_open_loop_loss = restore_p0_training_state(
            unwrap(model), optimizer, scheduler, resume_state, vars(config)
        )
        del resume_state
        if is_main:
            print(
                f"resumed_p0 checkpoint={args.resume} step={step} "
                f"max_steps={config.max_steps} best_open_loop_loss={best_open_loop_loss} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}",
                flush=True,
            )
    output_dir = config.resolved_output_dir
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(config), indent=2, default=list))
    if config.tensorboard and is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but torch.utils.tensorboard is unavailable"
            ) from exc
        writer = SummaryWriter(str(output_dir / "tensorboard"))
    else:
        writer = None

    # Existing P0 checkpoints do not contain a sampler cursor or RNG state.
    # Resume preserves the optimizer clock, but begins a fresh loader epoch.
    epoch = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < config.max_steps:
        if distributed:
            train_sampler.set_epoch(epoch)
        for raw in loader:
            model.train()
            batch = P0Batch(**move_to_device(raw, device))
            sync_gradients = (micro_step + 1) % config.gradient_accumulation_steps == 0
            ddp_context = nullcontext() if sync_gradients or not distributed else model.no_sync()
            with ddp_context:
                with unwrap(model).autocast_context():
                    output = model(batch)
                (output["loss"] / config.gradient_accumulation_steps).backward()
            micro_step += 1
            if not sync_gradients:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if writer is not None:
                for key, value in output.items():
                    if (key.startswith("loss") or key.startswith("jepa/")) and tensorboard_metric_enabled(key, value, config):
                        writer.add_scalar(key, float(value.detach()), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                if step == 1 or step % config.tensorboard_histogram_every == 0:
                    log_latent_distributions(writer, output, step)
            if is_main and (step == 1 or step % 100 == 0):
                print(f"step={step} loss={float(output['loss']):.6f} action={float(output['loss/action_dit']):.6f} jepa={float(output['loss/jepa']):.6f} sigreg={float(output['loss/sigreg']):.6f} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
            # Persist the trained weights before validation: an evaluator error
            # must not discard a completed save interval. best.pt is updated
            # separately, only after validation succeeds and improves.
            if is_main and (step % args.save_every == 0 or step == config.max_steps):
                save_checkpoint(model, optimizer, scheduler, step, output_dir / "checkpoints", config, best_metric=best_open_loop_loss)
            should_eval = config.open_loop_eval and config.eval_every > 0 and (step % config.eval_every == 0 or step == config.max_steps)
            if should_eval:
                metrics = evaluate_open_loop(model, val_loader, device, distributed, config.eval_batches)
                if is_main:
                    print(f"open_loop step={step} " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()), flush=True)
                    if writer is not None:
                        for key, value in metrics.items():
                            if tensorboard_metric_enabled(key, value, config):
                                writer.add_scalar("eval_open_loop/" + key.removeprefix("loss/"), value, step)
                    if config.save_best_checkpoint and metrics["loss"] < best_open_loop_loss:
                        best_open_loop_loss = metrics["loss"]
                        save_best(model, optimizer, scheduler, step, output_dir / "checkpoints", config, best_open_loop_loss)
            if step >= config.max_steps:
                break
        epoch += 1
    if writer is not None:
        writer.close()
    if distributed:
        torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
