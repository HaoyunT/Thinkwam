"""Shared training-image augmentation with spatially aligned soft labels.

Camera and object-position options are pixel-translation proxies on recorded
images, not new camera poses or object placements. The same translations are
applied to every observation, future target and camera-space affordance map.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isqrt

import numpy as np
import torch
from torch.utils.data import Subset


@dataclass
class ImageAugmentationConfig:
    enabled: bool = True
    probability: float = 0.50
    illumination_probability: float = 0.50
    texture_probability: float = 0.35
    camera_probability: float = 0.25
    noise_probability: float = 0.25
    object_position_probability: float = 0.25
    max_translation: int = 12
    noise_std: float = 0.025

    @classmethod
    def from_training_config(cls, config):
        names = {"enabled": "image_augmentations", "probability": "image_augmentation_probability"}
        return cls(**{field.name: getattr(config, names.get(field.name, "image_" + field.name))
                      for field in fields(cls)})


def _translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an HWC image with zero fill, without wrapping its borders."""
    height, width = image.shape[:2]
    out = np.zeros_like(image)
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    if x1 > x0 and y1 > y0:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[y0:y1, x0:x1]
    return out


def _translate_soft(maps: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Bilinear translation in heatmap coordinates, retaining soft values."""
    height, width = maps.shape[-2:]
    x = np.arange(width, dtype=np.float64) - dx
    y = np.arange(height, dtype=np.float64) - dy
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx, wy = x - x0, y - y0
    out = np.zeros_like(maps, dtype=np.float32)
    for ix, xweight in ((x0, 1 - wx), (x0 + 1, wx)):
        for iy, yweight in ((y0, 1 - wy), (y0 + 1, wy)):
            valid = ((iy[:, None] >= 0) & (iy[:, None] < height)
                     & (ix[None, :] >= 0) & (ix[None, :] < width))
            values = maps[..., np.clip(iy, 0, height - 1)[:, None],
                          np.clip(ix, 0, width - 1)[None, :]]
            out += values * (yweight[:, None] * xweight[None, :] * valid)
    return out


def _translate_affordance(target, *, views: int, image_hw: tuple[int, int], shifts):
    """Support [V,H,W], singleton-channel maps and flattened square grids.

    Flattened grids may be [V,H*W] or [V*H*W,1]. Unsupported layouts fail
    explicitly rather than silently leaving spatial supervision unaligned.
    """
    tensor = isinstance(target, torch.Tensor)
    values = target.detach().cpu().numpy() if tensor else np.asarray(target)
    shape = values.shape
    if values.ndim == 3 and shape[0] == views:
        maps = values
    elif values.ndim == 4 and shape[0] == views and shape[1] == 1:
        maps = values[:, 0]
    elif values.ndim == 4 and shape[0] == views and shape[-1] == 1:
        maps = values[..., 0]
    elif values.ndim == 2 and (shape[0] == views or shape[1] == 1):
        count = values.size // views
        side = isqrt(count)
        if values.size % views or side * side != count:
            raise ValueError(f"Affordance target is not a square grid per view: {shape}")
        maps = values.reshape(views, side, side)
    else:
        raise ValueError(f"Unsupported camera-space affordance shape {shape} for {views} views")
    maps = maps.astype(np.float32, copy=True)
    height, width = image_hw
    for dx, dy in shifts:
        maps = _translate_soft(maps, dx * maps.shape[-1] / width,
                               dy * maps.shape[-2] / height)
    result = maps.reshape(shape)
    if tensor:
        return torch.as_tensor(result, dtype=target.dtype, device=target.device)
    return result.astype(values.dtype, copy=False)


class ImageAugmenter:
    """Share geometric transforms across views, time targets and labels.

    The default RNG follows numpy's process/worker seed. DataLoader seeds each
    worker independently; explicit seeds are useful for isolated CPU tests.
    """

    def __init__(self, config: ImageAugmentationConfig, seed: int | None = None):
        self.config = config
        self._rng = None if seed is None else np.random.RandomState(seed)

    @property
    def rng(self):
        # Do not pickle a module into spawn-based DataLoader workers.
        return np.random if self._rng is None else self._rng

    @rng.setter
    def rng(self, value):
        self._rng = value

    def _transform(self, image: np.ndarray, recipe: dict) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        if recipe["illumination"]:
            values = values * recipe["brightness"] + recipe["color_shift"]
        if recipe["texture"]:
            values = np.clip(values / 255.0, 0.0, 1.0) ** recipe["gamma"]
            values = values * recipe["tint"] * 255.0
        for dx, dy in recipe["shifts"]:
            values = _translate(values, dx, dy)
        if recipe["noise"]:
            values = values + self.rng.normal(0.0, self.config.noise_std * 255.0, values.shape)
        return np.clip(values, 0.0, 255.0).astype(np.uint8)

    def __call__(self, sample: dict) -> dict:
        cfg = self.config
        if not cfg.enabled or self.rng.random() >= cfg.probability:
            return sample
        recipe = {
            "illumination": self.rng.random() < cfg.illumination_probability,
            "texture": self.rng.random() < cfg.texture_probability,
            "camera": self.rng.random() < cfg.camera_probability,
            "noise": self.rng.random() < cfg.noise_probability,
            "object_position": self.rng.random() < cfg.object_position_probability,
            "brightness": float(self.rng.uniform(0.75, 1.25)),
            "color_shift": self.rng.uniform(-10.0, 10.0, size=(1, 1, 3)),
            "gamma": float(self.rng.uniform(0.85, 1.15)),
            "tint": self.rng.uniform(0.9, 1.1, size=(1, 1, 3)),
        }
        shifts = [tuple(int(x) for x in self.rng.randint(-cfg.max_translation,
                   cfg.max_translation + 1, size=2)) for _ in range(2)]
        recipe["shifts"] = [shift for name, shift in zip(("camera", "object_position"), shifts)
                            if recipe[name]]
        augmented = dict(sample)
        augmented["obs_images"] = np.stack([self._transform(image, recipe)
                                            for image in sample["obs_images"]])
        augmented["future_images"] = {
            horizon: np.stack([self._transform(image, recipe) for image in images])
            for horizon, images in sample["future_images"].items()
        }
        if recipe["shifts"] and sample.get("affordance_targets") is not None:
            images = np.asarray(sample["obs_images"])
            augmented["affordance_targets"] = _translate_affordance(
                sample["affordance_targets"], views=len(images), image_hw=images.shape[1:3],
                shifts=recipe["shifts"],
            )
        return augmented


class AugmentedSubset(Subset):
    """A training subset whose base dataset remains unmodified for validation."""

    def __init__(self, dataset, indices, augmentation: ImageAugmentationConfig):
        super().__init__(dataset, indices)
        self.augmenter = ImageAugmenter(augmentation)

    def __getitem__(self, index):
        return self.augmenter(self.dataset[self.indices[index]])

    def __getitems__(self, indices):
        # Subset's batched fetch bypasses __getitem__; route it through here.
        return [self[index] for index in indices]


def make_libero_subsets(dataset, train_indices, val_indices, config):
    return (AugmentedSubset(dataset, train_indices, ImageAugmentationConfig.from_training_config(config)),
            Subset(dataset, val_indices))
