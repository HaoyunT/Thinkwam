"""Action-conditioned latent video residual generator.

The context Transformer reads current visual tokens and the first ActionDiT
chunk. Horizon-specific queries predict residual V-JEPA tokens at the three
configured horizons; the module does not generate RGB video.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from thinkwam_p0.models.memory import _uniform_pool
from thinkwam_p0.models.loss_utils import ddp_mean_from_sum


class ActionConditionedVideoResidualModel(nn.Module):
    def __init__(
        self,
        vjepa_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        layers: int = 2,
        heads: int = 8,
        horizons: Iterable[int] = (7, 15, 31),
        token_counts: Iterable[int] = (288, 144, 72),
        context_tokens: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("video hidden_dim must be divisible by heads")
        self.horizons = tuple(horizons)
        self.token_counts = tuple(token_counts)
        if len(self.horizons) != len(self.token_counts):
            raise ValueError("video horizons and token_counts must have equal length")
        self.context_tokens = context_tokens
        self.visual_proj = nn.Linear(vjepa_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.queries = nn.ParameterDict()
        self.cross = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        self.head = nn.ModuleDict()
        for horizon, count in zip(self.horizons, self.token_counts):
            key = f"t{horizon}"
            self.queries[key] = nn.Parameter(torch.randn(1, count, hidden_dim) * 0.02)
            self.cross[key] = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
            self.norm[key] = nn.LayerNorm(hidden_dim)
            self.head[key] = nn.Sequential(
                nn.Linear(hidden_dim, 2 * vjepa_dim),
                nn.GELU(),
                nn.Linear(2 * vjepa_dim, vjepa_dim),
            )
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, heads, 4 * hidden_dim, batch_first=True, norm_first=True
            ),
            num_layers=layers,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        action_condition: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        visual = self.visual_proj(_uniform_pool(visual_tokens, self.context_tokens))
        action = self.action_proj(action_condition)
        context = self.context_encoder(torch.cat([visual, action], dim=1))
        outputs: Dict[int, torch.Tensor] = {}
        for horizon, count in zip(self.horizons, self.token_counts):
            key = f"t{horizon}"
            query = self.queries[key].expand(context.shape[0], -1, -1)
            update, _ = self.cross[key](query, context, context, need_weights=False)
            outputs[horizon] = self.head[key](self.norm[key](query + update))
        return outputs

    @staticmethod
    def target_residual(target: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Create a stop-gradient residual target in the raw latent space."""
        pooled = F.adaptive_avg_pool1d(target.transpose(1, 2), base.shape[1]).transpose(1, 2)
        # ``corrected_visual = sg(base) + residual`` is performed in the raw
        # latent space.  The target must therefore be a raw-space residual;
        # subtracting normalized vectors would add a normalized delta to an
        # unnormalized prediction and produce a scale mismatch.
        return (pooled - base).detach()

    def loss(
        self,
        residuals: Dict[int, torch.Tensor],
        base_predictions: Dict[int, torch.Tensor],
        future_targets: Dict[int, torch.Tensor],
        future_masks: Dict[int, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        values = []
        active = []
        for horizon in self.horizons:
            prediction = residuals[horizon]
            future, base = future_targets[horizon], base_predictions[horizon]
            mask = None if future_masks is None else future_masks[horizon]
            if mask is not None:
                if mask.dtype != torch.bool or mask.shape != prediction.shape[:1]:
                    raise ValueError(f"future_masks[{horizon}] must be bool [B]")
                prediction, future, base = prediction[mask], future[mask], base[mask]
            has_target = prediction.shape[0] > 0
            if has_target:
                target = self.target_residual(future, base)
                local_sum = F.smooth_l1_loss(prediction, target, reduction="sum")
                local_count = prediction.numel()
            else:
                local_sum = residuals[horizon].sum() * 0.0
                local_count = 0
            value, global_count = ddp_mean_from_sum(local_sum, local_count)
            losses[f"video/residual_t{horizon}"] = value
            values.append(value)
            active.append(bool(global_count.item() > 0))
        stacked = torch.stack(values)
        losses["video/residual"] = stacked.mean() if all(active) else stacked.sum() / max(sum(active), 1)
        return losses
