"""Restore a P0 checkpoint within the same configured training stage.

The P0 checkpoint contract saves model, optimizer, scheduler, global step and
best validation loss. It omits the frozen ``vjepa.*`` state, RNG state and the
data-loader cursor; restoring an old checkpoint does not promise identical
future minibatches. The frozen encoder must be loaded from its configured
pretrained checkpoint before this helper is called.
"""

from collections.abc import Mapping
import math


# These affect output, loading throughput or evaluation, not the training
# recipe. In particular, max_steps and all learning-rate settings must match.
_RUNTIME_FIELDS = frozenset({
    "output_dir", "num_workers", "dataloader_prefetch_factor",
    "dataloader_persistent_workers", "dataloader_pin_memory",
    "dataloader_worker_threads", "tensorboard", "tensorboard_histogram_every",
    "checkpoint_save_limit", "eval_every", "eval_batches", "open_loop_eval",
    "save_best_checkpoint",
})


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def validate_p0_resume_checkpoint(state: Mapping, config: Mapping) -> None:
    """Reject weights-only loads, cross-recipe resumes and a reset LR clock."""
    if not isinstance(state, Mapping):
        raise ValueError("P0 resume checkpoint must be a mapping")
    for name in ("model", "optimizer", "scheduler", "config"):
        if not isinstance(state.get(name), Mapping):
            raise ValueError(f"P0 resume requires the saved {name} state")
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("P0 resume requires a non-negative integer global step")
    if step > int(config["max_steps"]):
        raise ValueError("P0 resume step exceeds this stage's max_steps")
    saved_config = state["config"]
    changed = sorted(
        key for key in (set(config) | set(saved_config)) - _RUNTIME_FIELDS
        if key not in config or key not in saved_config
        or _normalize(config[key]) != _normalize(saved_config[key])
    )
    if changed:
        raise ValueError(
            "P0 resume must retain the original stage configuration; changed fields: "
            + ", ".join(changed)
        )
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("P0 scheduler position does not match the saved global step")
    if not state["optimizer"].get("param_groups"):
        raise ValueError("P0 resume checkpoint has no optimizer parameter groups")
    metric = state.get("best_metric")
    if metric is not None and math.isnan(float(metric)):
        raise ValueError("P0 resume best_metric must not be NaN")


def restore_p0_training_state(model, optimizer, scheduler, state, config) -> tuple[int, float]:
    """Restore the saved optimizer/LR clock without restarting warmup.

    ``model`` must be the unwrapped P0 model. Only frozen encoder keys may be
    missing, matching ``train_p0.py``'s save contract. The optimizer and
    scheduler must already be constructed before their states are loaded.
    """
    validate_p0_resume_checkpoint(state, config)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    missing = [key for key in missing if not key.startswith("vjepa.")]
    if missing or unexpected:
        raise RuntimeError(
            f"P0 resume model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # LambdaLR.load_state_dict does not update the optimizer's LR. The saved
    # optimizer supplies the first resumed update's LR, and the saved scheduler
    # supplies the following step; neither is recomputed at step zero.
    restored_lrs = [group["lr"] for group in optimizer.param_groups]
    if restored_lrs != list(scheduler.get_last_lr()):
        raise ValueError("P0 optimizer and scheduler contain inconsistent learning rates")
    metric = state.get("best_metric")
    return int(state["step"]), float("inf") if metric is None else float(metric)
