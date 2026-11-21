"""Multi-horizon V-JEPA token predictor with temporal token decay."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .memory import _uniform_pool
from .sigreg import SIGReg
from .loss_utils import ddp_mean_from_sum


class TemporalJEPA(nn.Module):
    def __init__(
        self,
        vjepa_dim: int,
        qwen_dim: int,
        memory_dim: int,
        horizons: Iterable[int] = (7, 15, 31),
        token_counts: Iterable[int] = (128, 64, 32),
        context_dim: int = 1024,
        layers: int = 2,
        heads: int = 8,
        observation_tokens: int = 64,
        projection_hidden: int = 2048,
        sigreg_num_projections: int = 128,
        sigreg_num_knots: int = 17,
        sigreg_t_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.token_counts = tuple(token_counts)
        self.observation_tokens = int(observation_tokens)
        self.sigreg = SIGReg(
            knots=sigreg_num_knots,
            num_projections=sigreg_num_projections,
            t_max=sigreg_t_max,
        )
        if len(self.horizons) != len(self.token_counts):
            raise ValueError("horizons and token_counts must have equal length")
        # Qwen's native 896-d visual hidden is expanded through the same
        # JEPA-WAM-style 896->2048->1024 MLP used by the temporal context.
        self.obs_proj = nn.Sequential(
            nn.Linear(qwen_dim, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, context_dim),
        )
        self.memory_proj = nn.Linear(memory_dim, context_dim)
        self.queries = nn.ParameterDict()
        self.attn = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        self.head = nn.ModuleDict()
        for horizon, count in zip(self.horizons, self.token_counts):
            key = f"t{horizon}"
            self.queries[key] = nn.Parameter(torch.randn(1, count, context_dim) * 0.02)
            self.attn[key] = nn.MultiheadAttention(context_dim, heads, batch_first=True)
            self.norm[key] = nn.LayerNorm(context_dim)
            self.head[key] = nn.Sequential(
                nn.Linear(context_dim, 2 * vjepa_dim),
                nn.GELU(),
                nn.Linear(2 * vjepa_dim, vjepa_dim),
            )
        self.context_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(context_dim, heads, 4 * context_dim, batch_first=True, norm_first=True) for _ in range(layers)]
        )

    def encode_context(self, visual_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Encode the current visual/memory context before horizon queries.

        P1 uses this representation as the shared latent space for its
        affordance decoder.  The returned sequence contains the projected
        visual tokens first and the memory slots second.
        """
        context = torch.cat(
            [self.obs_proj(_uniform_pool(visual_hidden, self.observation_tokens)), self.memory_proj(memory)],
            dim=1,
        )
        for layer in self.context_layers:
            context = layer(context)
        return context

    def forward(
        self,
        visual_hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> Dict[int, torch.Tensor] | tuple[Dict[int, torch.Tensor], torch.Tensor]:
        context = self.encode_context(visual_hidden, memory)
        predictions = {}
        for horizon, count in zip(self.horizons, self.token_counts):
            key = f"t{horizon}"
            query = self.queries[key].expand(context.shape[0], -1, -1)
            update, _ = self.attn[key](query, context, context, need_weights=False)
            predictions[horizon] = self.head[key](self.norm[key](query + update))
        if return_context:
            return predictions, context
        return predictions

    @staticmethod
    def target_tokens(target: torch.Tensor, count: int) -> torch.Tensor:
        # Adaptive average pooling gives distant targets a deliberately coarse
        # spatial summary without throwing away the target embedding dimension.
        pooled = F.adaptive_avg_pool1d(target.transpose(1, 2), count).transpose(1, 2)
        return F.normalize(pooled, dim=-1)

    def loss(
        self, predictions: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor],
        *, compute_sigreg: bool = False, future_masks: Dict[int, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        active = []
        for horizon, count in zip(self.horizons, self.token_counts):
            prediction = predictions[horizon]
            target = targets[horizon]
            mask = None if future_masks is None else future_masks[horizon]
            if mask is not None:
                if mask.dtype != torch.bool or mask.shape != prediction.shape[:1]:
                    raise ValueError(f"future_masks[{horizon}] must be bool [B]")
                prediction, target = prediction[mask], target[mask]
            has_target = prediction.shape[0] > 0
            if has_target:
                pred = F.normalize(prediction, dim=-1)
                target = self.target_tokens(target, count).detach()
                local_sum = (1.0 - (pred * target).sum(dim=-1)).sum()
            else:
                local_sum = predictions[horizon].sum() * 0.0
            # Average over B * N_h tokens so every horizon is weighted equally
            # despite its token budget.
            value, global_count = ddp_mean_from_sum(local_sum, prediction.shape[0] * count)
            losses[f"jepa/t{horizon}"] = value
            active.append(bool(global_count.item() > 0))
        if compute_sigreg:
            # Test each query position across the batch. This prevents token
            # collapse without treating the different spatial/query positions
            # as interchangeable samples; horizons receive equal weight.
            sigreg_terms = [
                self.sigreg(
                    predictions[h].float().transpose(0, 1),
                    sample_mask=None if future_masks is None else future_masks[h],
                )
                for h in self.horizons
            ]
            losses["jepa/sigreg"] = torch.stack(sigreg_terms).mean()
        else:
            losses["jepa/sigreg"] = predictions[self.horizons[0]].new_zeros((), dtype=torch.float32)
        prediction_terms = [losses[f"jepa/t{horizon}"] for horizon in self.horizons]
        stacked = torch.stack(prediction_terms)
        losses["jepa/total"] = stacked.mean() if all(active) else stacked.sum() / max(sum(active), 1)
        return losses
