"""Numerical input contract shared by the LIBERO cache and online evaluator."""

import numpy as np
import torch
import torch.nn.functional as F


def make_image_batch(encoder, arrays, device: torch.device) -> torch.Tensor:
    """Keep the original cache builder's tensor resize, without PIL conversion."""
    images = torch.from_numpy(np.stack(arrays, axis=0)).to(device=device, dtype=torch.float32)
    images = images.permute(0, 3, 1, 2).div_(255.0)
    images = F.interpolate(images, size=(encoder.image_size, encoder.image_size), mode="bicubic", align_corners=False)
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (images - mean) / std


def uniform_pool(tokens: np.ndarray, count: int) -> np.ndarray:
    """Match the cache's rounded linspace indices (not torch.long truncation)."""
    if tokens.shape[1] == count:
        return tokens
    indices = np.linspace(0, tokens.shape[1] - 1, count).round().astype(np.int64)
    return tokens[:, indices]


@torch.no_grad()
def encode_pairs(encoder, pairs, device):
    images = torch.stack(
        [make_image_batch(encoder, [a for a, _ in pairs], device), make_image_batch(encoder, [b for _, b in pairs], device)],
        dim=1,
    )
    return encoder(images).float().cpu().numpy()


def online_cache_features(encoder, obs, device, token_count: int) -> torch.Tensor:
    tokens = encode_pairs(encoder, [(obs["agentview_image"], obs["robot0_eye_in_hand_image"])], device)
    # The training loader reads float16 cache entries back into float32.
    tokens = uniform_pool(tokens, token_count).astype(np.float16).astype(np.float32)
    return torch.from_numpy(tokens).to(device)


def prepare_raw_action(action):
    """LIBERO HDF5 actions are raw controller commands with a +/-1 gripper."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid LIBERO action: {action}")
    action = np.clip(action, -1.0, 1.0)
    action[-1] = 1.0 if action[-1] >= 0 else -1.0
    return action
