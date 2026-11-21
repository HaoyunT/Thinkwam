"""Frozen V-JEPA 2.1 ViT-L/16 adapter copied from the public JEPA-WAM contract."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from .vjepa import vision_transformer as vit_module


class FrozenVJEPA21(nn.Module):
    """Frozen encoder returning spatial tokens; no V-JEPA gradients are allowed."""

    def __init__(self, checkpoint: Optional[str] = None, image_size: int = 384, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = 1024
        self.tubelet_size = 2
        self.featurizer = vit_module.vit_large(
            patch_size=patch_size,
            img_size=(image_size, image_size),
            num_frames=16,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        if checkpoint:
            self._load_checkpoint(Path(checkpoint))
        self.featurizer = self.featurizer.to(dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.eval()
        self.image_transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = None
        for key in ("target_encoder", "ema_encoder", "encoder"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        if state is None:
            state = checkpoint
        cleaned = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
        missing, unexpected = self.featurizer.load_state_dict(cleaned, strict=False)
        missing = [key for key in missing if "pos_embed" not in key]
        if missing or unexpected:
            raise RuntimeError(f"V-JEPA checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError("images must have shape [B,V,3,H,W]")
        batch, views = images.shape[:2]
        x = images.reshape(batch * views, 3, images.shape[-2], images.shape[-1]).unsqueeze(2)
        x = x.to(device=next(self.featurizer.parameters()).device, dtype=torch.bfloat16)
        output = self.featurizer(x)
        if isinstance(output, list):
            output = output[-1]
        return output.reshape(batch, views * output.shape[1], output.shape[2])

    @torch.inference_mode()
    def encode_future(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 6:
            raise ValueError("clips must have shape [B,V,T,3,H,W]")
        batch, views, frames = clips.shape[:3]
        x = clips.reshape(batch * views, frames, 3, clips.shape[-2], clips.shape[-1]).permute(0, 2, 1, 3, 4)
        x = x.to(device=next(self.featurizer.parameters()).device, dtype=torch.bfloat16)
        output = self.featurizer(x)
        if isinstance(output, list):
            output = output[-1]
        temporal = max(1, frames // self.tubelet_size)
        side = self.image_size // self.patch_size
        return output.reshape(batch, views * temporal * side * side, output.shape[-1])
