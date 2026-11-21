# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from LeJEPA; see THIRD_PARTY_NOTICES.md and licenses/LeJEPA-CC-BY-NC-4.0.txt.
"""SIGReg's sliced Epps--Pulley statistic, with differentiable DDP statistics.

Method: LeJEPA, https://arxiv.org/abs/2511.08544
Reference quadrature: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
The token pooling used by TemporalJEPA is a ThinkWAM adaptation.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce


class SIGReg(nn.Module):
    """Regularize [N, D] or [groups, N, D], independently across groups.

    N is the batch axis, never the spatial-token axis. With DDP, ranks share
    random projection directions and aggregate characteristic-function sums
    with autograd, then multiply the statistic by the global sample count.
    Evaluation uses fixed directions for repeatable validation.
    """

    def __init__(self, *, knots: int = 17, num_projections: int = 128, t_max: float = 3.0) -> None:
        super().__init__()
        if knots < 2 or num_projections <= 0 or not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("SIGReg requires knots >= 2, positive projections and finite positive t_max")
        frequencies = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        normal_cf = torch.exp(-0.5 * frequencies.square())
        quadrature = torch.ones_like(frequencies) * (2.0 * t_max / (knots - 1))
        quadrature[0] *= 0.5
        quadrature[-1] *= 0.5
        self.num_projections = num_projections
        # Deterministic derived constants must not break old checkpoint loads.
        self.register_buffer("t", frequencies, persistent=False)
        self.register_buffer("phi", normal_cf, persistent=False)
        self.register_buffer("weights", quadrature * normal_cf, persistent=False)

    def forward(self, embeddings: torch.Tensor, sample_mask: torch.Tensor | None = None) -> torch.Tensor:
        if embeddings.ndim not in (2, 3) or embeddings.shape[-2] < 1 or embeddings.shape[-1] < 1:
            raise ValueError("SIGReg expects nonempty [N, D] or [groups, N, D] embeddings")
        distributed = dist.is_available() and dist.is_initialized()
        if sample_mask is not None and (
            sample_mask.dtype != torch.bool or sample_mask.shape != embeddings.shape[-2:-1]
        ):
            raise ValueError("SIGReg sample_mask must be bool [N]")
        # .float() alone does not protect matrix multiplication from autocast.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            samples = embeddings.float()
            if sample_mask is not None:
                samples = torch.where(sample_mask[..., None], samples, torch.zeros_like(samples))
            generator = None
            if not self.training:
                generator = torch.Generator(device=samples.device).manual_seed(0)
            directions = torch.randn(
                samples.shape[-1], self.num_projections,
                device=samples.device, dtype=torch.float32, generator=generator,
            )
            if distributed:
                dist.broadcast(directions, src=0)
            directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
            phase = (samples @ directions).unsqueeze(-1) * self.t.float()
            # Preserve groups/horizons; sum only over distinct batch samples.
            if sample_mask is None:
                moments = torch.stack((phase.cos().sum(dim=-3), phase.sin().sum(dim=-3)))
                count = samples.new_tensor(float(samples.shape[-2]))
            else:
                weights = sample_mask[..., None, None].to(phase.dtype)
                moments = torch.stack(((phase.cos() * weights).sum(dim=-3), (phase.sin() * weights).sum(dim=-3)))
                count = sample_mask.sum().to(device=samples.device, dtype=samples.dtype)
            if distributed:
                moments = all_reduce(moments, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            moments = moments / count.clamp_min(1)
            discrepancy = (moments[0] - self.phi.float()).square() + moments[1].square()
            value = count * (discrepancy * self.weights.float()).sum(dim=-1).mean()
            # Keep every DDP rank in the collectives even when its valid subset is empty.
            return torch.where(count >= 2, value, value * 0.0) if sample_mask is not None else value
