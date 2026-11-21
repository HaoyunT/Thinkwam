"""Small helpers for sample-weighted losses under data parallel training."""

from __future__ import annotations

import torch
import torch.distributed as dist


def ddp_mean_from_sum(local_sum: torch.Tensor, local_count: torch.Tensor | int | float):
    """Return a global-sample mean while preserving DDP's gradient averaging.

    DDP averages parameter gradients across ranks. Scaling each rank's local
    sum by ``world_size / global_count`` therefore gives the gradient of the
    concatenated-batch mean after the reducer averages it. The count is kept
    outside autograd; it is only a normalization statistic.
    """
    if not isinstance(local_count, torch.Tensor):
        local_count = local_sum.new_tensor(float(local_count))
    else:
        local_count = local_count.detach().to(device=local_sum.device, dtype=local_sum.dtype)
    distributed = dist.is_available() and dist.is_initialized()
    global_count = local_count.clone()
    world_size = 1
    if distributed:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    value = local_sum * float(world_size) / global_count.clamp_min(1.0)
    return value, global_count
