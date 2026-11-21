"""Typed batch contract shared by all P0 dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class P0Batch:
    """A P0 batch supporting online images and feature-level inputs.

    Paper recipes encode current RGB observations and future targets online.
    The separate feature-level form supports synthetic tests and cache readers,
    including RoboCasa24 frozen future targets. Source and checkpoint validators
    determine which inputs a training workflow permits.
    """

    actions: torch.Tensor  # [B, action_horizon, action_dim]
    action_history: torch.Tensor  # [B, memory_action_history, action_dim]
    obs_features: Optional[torch.Tensor] = None  # [B, N_obs, D_vjepa] when using a feature cache
    future_features: Optional[Dict[int, torch.Tensor]] = None  # horizon -> [B, N, D_vjepa]
    obs_images: Optional[torch.Tensor] = None  # [B, V, 3, H, W] for online V-JEPA
    future_images: Optional[Dict[int, torch.Tensor]] = None  # horizon -> [B, V, 3, H, W]
    text_features: Optional[torch.Tensor] = None  # optional precomputed instruction features
    proprio: Optional[torch.Tensor] = None  # [B, proprio_dim]
    text_input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None
    action_mask: Optional[torch.Tensor] = None  # [B, action_horizon], real action labels only
    future_masks: Optional[Dict[int, torch.Tensor]] = None  # horizon -> [B], real future targets only

    def validate(
        self,
        horizons: tuple[int, ...],
        vjepa_dim: int,
        action_horizon: int,
        *,
        visual_token_count: int = 0,
        online_vjepa: bool = False,
        vjepa_views: int = 2,
        future_target_cache: bool = False,
    ) -> None:
        if online_vjepa:
            if self.obs_features is not None:
                raise ValueError("online_vjepa observations must not contain precomputed features")
            if self.obs_images is None:
                raise ValueError("online_vjepa batches require obs_images")
            if future_target_cache:
                if self.future_features is None or self.future_images is not None:
                    raise ValueError("future-target cache requires features instead of future images")
            elif self.future_features is not None or self.future_images is None:
                raise ValueError("online_vjepa batches require future_images, not cached features")
            if not future_target_cache and set(self.future_images) != set(horizons):
                raise ValueError(f"future_images keys must be {horizons}, got {tuple(self.future_images)}")
            if self.obs_images.ndim != 5 or self.obs_images.shape[1] != vjepa_views:
                raise ValueError(
                    f"obs_images must be [B,{vjepa_views},3,H,W], got {tuple(self.obs_images.shape)}"
                )
            for horizon in (() if future_target_cache else horizons):
                target = self.future_images[horizon]
                if target.ndim != 5 or target.shape[0] != self.obs_images.shape[0] or target.shape[1] != vjepa_views:
                    raise ValueError(
                        f"future_images[{horizon}] must be [B,{vjepa_views},3,H,W], got {tuple(target.shape)}"
                    )
        else:
            if self.obs_features is None or self.future_features is None:
                raise ValueError("feature-cache batches require obs_features and future_features")
            if self.obs_features.ndim != 3 or self.obs_features.shape[-1] != vjepa_dim:
                raise ValueError(f"obs_features must be [B,N,{vjepa_dim}], got {tuple(self.obs_features.shape)}")
            if visual_token_count > 0 and self.obs_features.shape[1] != visual_token_count:
                raise ValueError(
                    f"obs_features must contain {visual_token_count} tokens, got {self.obs_features.shape[1]}"
                )
        if self.actions.ndim != 3 or self.actions.shape[1] != action_horizon:
            raise ValueError(f"actions must be [B,{action_horizon},A], got {tuple(self.actions.shape)}")
        if self.action_mask is not None:
            if self.action_mask.dtype != torch.bool or self.action_mask.shape != self.actions.shape[:2]:
                raise ValueError("action_mask must be bool [B,action_horizon]")
        if self.future_masks is not None:
            if set(self.future_masks) != set(horizons):
                raise ValueError(f"future_masks keys must be {horizons}")
            for horizon, mask in self.future_masks.items():
                if mask.dtype != torch.bool or mask.shape != self.actions.shape[:1]:
                    raise ValueError(f"future_masks[{horizon}] must be bool [B]")
        if self.action_history.ndim != 3 or self.action_history.shape[0] != self.actions.shape[0]:
            raise ValueError("action_history must be [B,H,A] with the same batch size as actions")
        if not online_vjepa or future_target_cache:
            if set(self.future_features) != set(horizons):
                raise ValueError(f"future_features keys must be {horizons}, got {tuple(self.future_features)}")
            for horizon in horizons:
                target = self.future_features[horizon]
                if target.ndim != 3 or target.shape[0] != self.actions.shape[0] or target.shape[-1] != vjepa_dim:
                    raise ValueError(f"future_features[{horizon}] has invalid shape {tuple(target.shape)}")
