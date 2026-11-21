"""Contact-heatmap decoder on the temporal predictor's visual context.

The caller applies the configured upstream-gradient and memory-fusion
schedules. The decoder is evaluated before action generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thinkwam_p0.models.memory import _uniform_pool


class LatentAffordanceDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        channels: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("affordance hidden_dim must be divisible by heads")
        self.channels = channels
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.latent_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, heads, 4 * hidden_dim, batch_first=True, norm_first=True
            ),
            num_layers=layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Decode one affordance logit per latent visual token."""
        latent = self.latent_decoder(self.latent_proj(latent_tokens))
        return self.head(self.norm(latent))

    @staticmethod
    def canonicalize_target(
        target: torch.Tensor,
        *,
        batch_size: int,
        token_count: int,
        channels: int,
    ) -> torch.Tensor:
        """Convert token or camera-map labels to ``[B,N,C]`` in [0,1]."""
        if target.shape[0] != batch_size:
            raise ValueError(f"affordance batch mismatch: expected {batch_size}, got {target.shape[0]}")
        if target.ndim == 2:
            target = target[..., None]
        elif target.ndim == 3 and target.shape[-1] != channels:
            # [B,V,H] is a single-channel map with no explicit channel axis.
            target = target.reshape(batch_size, -1, 1)
        elif target.ndim == 4:
            # [B,V,H,W] camera maps.
            target = target.reshape(batch_size, -1, 1)
        elif target.ndim == 5:
            # [B,V,H,W,C] camera maps.
            target = target.reshape(batch_size, -1, target.shape[-1])
        elif target.ndim != 3:
            raise ValueError("affordance target must be [B,N], [B,N,C], [B,V,H,W], or [B,V,H,W,C]")
        if target.shape[-1] == 1 and channels > 1:
            target = target.expand(-1, -1, channels)
        if target.shape[-1] != channels:
            raise ValueError(f"affordance target channels {target.shape[-1]} != decoder channels {channels}")
        if target.shape[1] != token_count:
            target = F.adaptive_avg_pool1d(target.transpose(1, 2), token_count).transpose(1, 2)
        return target.float().clamp(0.0, 1.0)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target = self.canonicalize_target(
            target,
            batch_size=logits.shape[0],
            token_count=logits.shape[1],
            channels=logits.shape[2],
        ).to(device=logits.device, dtype=logits.dtype)
        if sample_mask is None:
            sample_mask = torch.ones(
                logits.shape[0], device=logits.device, dtype=logits.dtype
            )
        else:
            sample_mask = sample_mask.to(device=logits.device, dtype=logits.dtype)
        if sample_mask.sum() <= 0:
            zero = logits.sum() * 0.0
            return {"affordance/bce": zero, "affordance/dice": zero, "affordance/total": zero}
        weights = sample_mask[:, None, None]
        bce_values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce_values * weights).sum() / (
            weights.sum().clamp_min(1.0) * logits.shape[1] * logits.shape[2]
        )
        probability = logits.sigmoid()
        reduce_dims = (1, 2)
        intersection = (probability * target).sum(dim=reduce_dims)
        denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice_values = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
        dice = (dice_values * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)
        return {"affordance/bce": bce, "affordance/dice": dice, "affordance/total": bce + dice}
