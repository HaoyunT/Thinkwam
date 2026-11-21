"""RoboTwin contact-point to camera heatmap conversion.

This module deliberately depends only on NumPy.  The simulator-side hook can
call :func:`make_frame_record` immediately after ``task.get_obs()`` and before
the frame is persisted.  The resulting JSONL records can be converted to
per-camera ``[T,H,W]`` arrays with ``scripts/export_robotwin_affordance.py``.

The intended contact source is RoboTwin's
``get_gripper_actor_contact_position(actor_name)``.  Asset contact points and
functional points have different semantics and should not be passed here as
runtime contacts without an explicit policy choice.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def _area_weights(source_size: int, target_size: int) -> np.ndarray:
    """Return exact pixel-area weights for one spatial dimension."""
    if source_size <= 0 or target_size <= 0:
        raise ValueError("source_size and target_size must be positive")
    source_edges = np.arange(source_size + 1, dtype=np.float64)
    target_edges = np.linspace(0.0, float(source_size), target_size + 1)
    weights = np.zeros((target_size, source_size), dtype=np.float32)
    for index in range(target_size):
        left, right = target_edges[index], target_edges[index + 1]
        overlap = np.maximum(
            0.0,
            np.minimum(source_edges[1:], right) - np.maximum(source_edges[:-1], left),
        )
        weights[index] = overlap / max(right - left, 1e-12)
    return weights


def _area_resize_batch(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Area-pool ``[T,H,W]`` maps without an image-library dependency."""
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_hw must contain positive dimensions")
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [T,H,W] affordance maps, got {values.shape}")
    wy = _area_weights(values.shape[1], target_h)
    wx = _area_weights(values.shape[2], target_w)
    intermediate = np.einsum("yh,thw->tyw", wy, values, optimize=True)
    return np.einsum("xw,tyw->tyx", wx, intermediate, optimize=True)


def deva_normalize_and_pool(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Normalize each frame like DeVA and area-pool to the target grid."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [T,H,W] affordance maps, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    maximum = values.reshape(values.shape[0], -1).max(axis=1)[:, None, None]
    normalized = np.divide(
        values,
        np.maximum(maximum, 1e-12),
        where=maximum > 0,
        out=np.zeros_like(values),
    )
    pooled = _area_resize_batch(normalized, target_hw)
    return np.clip(pooled, 0.0, 1.0).astype(np.float16, copy=False)


def _as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = matrix
        return result
    if matrix.shape != (4, 4):
        raise ValueError(f"camera extrinsic must be [3,4] or [4,4], got {matrix.shape}")
    return matrix


def project_world_points(
    points_world: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with RoboTwin's OpenCV camera matrices.

    ``Camera.get_extrinsic_matrix()`` is treated as a world-to-camera matrix.
    The returned tuple contains pixel coordinates ``[N,2]`` and a validity mask
    requiring finite values and positive camera depth.  Keeping the mask
    explicit prevents points behind the camera from creating false labels.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera intrinsic must be [3,3], got {intrinsic.shape}")
    extrinsic = _as_homogeneous(np.asarray(extrinsic_cv, dtype=np.float64))
    camera = (extrinsic @ np.concatenate([points, np.ones((len(points), 1))], axis=1).T).T[:, :3]
    depth = camera[:, 2]
    projected = (intrinsic @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[:, :2] / projected[:, 2:3]
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    return pixels, valid


def gaussian_contact_heatmap(
    pixels: np.ndarray,
    valid: np.ndarray,
    height: int,
    width: int,
    sigma: float = 6.0,
) -> np.ndarray:
    """Rasterize projected contacts as a max-composed Gaussian heatmap."""
    if height <= 0 or width <= 0 or sigma <= 0:
        raise ValueError("height, width and sigma must be positive")
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if len(pixels) != len(valid):
        raise ValueError("pixels and valid must have the same length")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = 4.0 * float(sigma)
    for (x, y), keep in zip(pixels, valid):
        if not keep or x < -radius or x >= width + radius or y < -radius or y >= height + radius:
            continue
        x0, x1 = max(0, int(np.floor(x - radius))), min(width, int(np.ceil(x + radius + 1)))
        y0, y1 = max(0, int(np.floor(y - radius))), min(height, int(np.ceil(y + radius + 1)))
        local = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2.0 * sigma * sigma))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], local)
    return heatmap


def make_frame_record(
    frame_index: int,
    camera_configs: Mapping[str, Mapping[str, object]],
    contacts_world: Sequence[Sequence[float]],
) -> dict:
    """Create a serializable per-frame record from a RoboTwin task.

    ``camera_configs`` is the corresponding subset of
    ``task.cameras.get_config()``.  RGB persistence remains the responsibility
    of RoboTwin's existing collector; this record only stores calibration and
    contacts needed for deterministic label export.
    """
    configs = {}
    for name, config in camera_configs.items():
        configs[name] = {
            "height": int(config.get("height", config.get("h", 0))),
            "width": int(config.get("width", config.get("w", 0))),
            "intrinsic_cv": np.asarray(config["intrinsic_cv"], dtype=float).tolist(),
            "extrinsic_cv": np.asarray(config["extrinsic_cv"], dtype=float).tolist(),
        }
    return {
        "frame_index": int(frame_index),
        "contacts_world": np.asarray(list(contacts_world), dtype=float).reshape(-1, 3).tolist(),
        "cameras": configs,
    }


def frame_to_heatmaps(record: Mapping[str, object], sigma: float = 6.0) -> dict[str, np.ndarray]:
    """Convert one JSON-compatible record to per-camera heatmaps."""
    contacts = np.asarray(record.get("contacts_world", []), dtype=np.float64).reshape(-1, 3)
    result = {}
    for camera_name, config in dict(record["cameras"]).items():
        height = int(config["height"])
        width = int(config["width"])
        pixels, valid = project_world_points(contacts, config["intrinsic_cv"], config["extrinsic_cv"])
        in_frame = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        result[camera_name] = gaussian_contact_heatmap(pixels, in_frame, height, width, sigma)
    return result
