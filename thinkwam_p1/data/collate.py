"""P1 collator, compatible with P0 feature/image samples."""

from __future__ import annotations

import torch

from thinkwam_p0.data.collate import collate_p0


def collate_p1(samples: list[dict]) -> dict:
    batch = collate_p0(samples)
    values = [sample.get("affordance_targets", sample.get("affordance")) for sample in samples]
    valid = torch.tensor([value is not None for value in values], dtype=torch.bool)
    batch["affordance_mask"] = valid
    if not valid.any():
        batch["affordance_targets"] = None
    else:
        template = next(value for value in values if value is not None)
        template = template if isinstance(template, torch.Tensor) else torch.as_tensor(template)
        batch["affordance_targets"] = torch.stack(
            [
                (value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
                if value is not None else torch.zeros_like(template)
                for value in values
            ]
        ).float()
    return batch
