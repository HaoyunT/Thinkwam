from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _preprocess_images(values: list[np.ndarray], image_size: int = 384) -> torch.Tensor:
    """Normalize mixed-resolution HWC uint8 views to ``[B,V,3,S,S]``."""
    tensors = []
    for value in values:
        image = torch.from_numpy(np.asarray(value)).float()
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Expected image arrays [V,H,W,3], got {tuple(image.shape)}")
        image = image.permute(0, 3, 1, 2).div_(255.0)
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
        tensors.append(image)
    images = torch.stack(tensors, dim=0)
    batch, views = images.shape[:2]
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return ((images - mean) / std).reshape(batch, views, 3, image_size, image_size)


def collate_p0(samples: list[dict]) -> dict:
    batch = {
        "actions": torch.stack([sample["actions"] for sample in samples]),
        "action_history": torch.stack([sample["action_history"] for sample in samples]),
    }
    action_masks = [sample.get("action_mask") for sample in samples]
    if any(mask is not None for mask in action_masks):
        batch["action_mask"] = torch.stack([
            torch.as_tensor(mask, dtype=torch.bool) if mask is not None
            else torch.ones(sample["actions"].shape[0], dtype=torch.bool)
            for sample, mask in zip(samples, action_masks)
        ])
    future_masks = [sample.get("future_masks") for sample in samples]
    if any(masks is not None for masks in future_masks):
        horizons = set(next(masks for masks in future_masks if masks is not None))
        if any(masks is not None and set(masks) != horizons for masks in future_masks):
            raise ValueError("future_masks must contain the same horizons in every sample")
        batch["future_masks"] = {
            h: torch.stack([
                torch.as_tensor(masks[h], dtype=torch.bool) if masks is not None
                else torch.tensor(True)
                for masks in future_masks
            ])
            for h in sorted(horizons)
        }
    if samples[0].get("obs_features") is not None:
        batch["obs_features"] = torch.stack([sample["obs_features"] for sample in samples])
    else:
        batch["obs_features"] = None
    text_features = [sample.get("text_features") for sample in samples]
    batch["text_features"] = None if text_features[0] is None else torch.stack(text_features)
    text_ids = [sample.get("text_input_ids") for sample in samples]
    masks = [sample.get("attention_mask") for sample in samples]
    if text_ids[0] is None:
        batch["text_input_ids"] = None
        batch["attention_mask"] = None
    else:
        batch["text_input_ids"] = pad_sequence(text_ids, batch_first=True, padding_value=0)
        if masks[0] is None:
            batch["attention_mask"] = batch["text_input_ids"].ne(0)
        else:
            batch["attention_mask"] = pad_sequence(masks, batch_first=True, padding_value=0).bool()
    if samples[0].get("future_features") is not None:
        horizons = tuple(samples[0]["future_features"])
        batch["future_features"] = {
            h: torch.stack([sample["future_features"][h] for sample in samples]) for h in horizons
        }
    else:
        batch["future_features"] = None
    proprio = [sample.get("proprio") for sample in samples]
    batch["proprio"] = None if proprio[0] is None else torch.stack(proprio)
    if samples[0].get("obs_images") is not None:
        batch["obs_images"] = _preprocess_images([sample["obs_images"] for sample in samples])
        future_images = {}
        for horizon in tuple(samples[0].get("future_images") or {}):
            future_images[horizon] = _preprocess_images(
                [sample["future_images"][horizon] for sample in samples]
            )
        batch["future_images"] = future_images or None
    return batch
