"""P1 batch contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from thinkwam_p0.types import P0Batch


@dataclass
class P1Batch(P0Batch):
    """P0 inputs plus affordance supervision.

    ``affordance_targets`` can be token labels ``[B,N]``/``[B,N,1]`` or a
    camera map ``[B,V,H,W]``/``[B,V,H,W,1]``.  The decoder canonicalizes maps
    to the visual-token grid before computing its loss.
    """

    affordance_targets: Optional[torch.Tensor] = None
    # Samples without an available contact label are excluded from the
    # affordance loss, as specified by the supervision contract.
    affordance_mask: Optional[torch.Tensor] = None

    def validate_p1(
        self,
        *,
        horizons: tuple[int, ...],
        vjepa_dim: int,
        action_horizon: int,
        visual_token_count: int,
        online_vjepa: bool,
        vjepa_views: int = 2,
        require_affordance_targets: bool,
        future_target_cache: bool = False,
    ) -> None:
        self.validate(
            horizons,
            vjepa_dim,
            action_horizon,
            visual_token_count=visual_token_count if not online_vjepa else 0,
            online_vjepa=online_vjepa,
            vjepa_views=vjepa_views,
            future_target_cache=future_target_cache,
        )
        if require_affordance_targets and self.affordance_targets is None:
            raise ValueError(
                "This recipe requires affordance_targets; prepare the contact heatmaps before training"
            )

    @classmethod
    def from_p0(cls, batch: P0Batch, affordance_targets: Optional[torch.Tensor]) -> "P1Batch":
        values = dict(batch.__dict__)
        values["affordance_targets"] = affordance_targets
        return cls(**values)

