"""Configuration for the P0 JEPA-WAM style model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Tuple


@dataclass
class P0Config:
    # Frozen V-JEPA 2.1 ViT-L/16 output contract.
    vjepa_checkpoint: str = "/data/thinkwam/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    vjepa_dim: int = 1024
    vjepa_image_size: int = 384
    vjepa_patch_size: int = 16
    vjepa_views: int = 2

    # Qwen2.5 conditioning with the LIBERO paper defaults.
    qwen_path: str = "/data/thinkwam/models/Qwen2.5-0.5B"
    qwen_dim: int = 896
    qwen_max_length: int = 512
    qwen_action_tokens: int = 64
    qwen_lora: bool = True
    qwen_lora_rank: int = 32
    qwen_lora_alpha: int = 64
    qwen_lora_dropout: float = 0.1
    qwen_lora_target_modules: str = "all-linear"
    qwen_attn_implementation: str = "sdpa"
    # Online V-JEPA emits the full per-view grid, but Qwen can use a
    # uniformly sampled/pooled subset to reduce quadratic attention cost.
    # Zero uses visual_token_count without further reduction.
    qwen_visual_tokens: int = 0
    # P0 consumes hidden states only; optionally skip the unused vocabulary head.
    qwen_hidden_only: bool = True
    # P0 conditioning width. Qwen stays at its native 896 dimensions;
    # visual/action context is projected through 896->2048->1024.
    action_context_dim: int = 1024
    conditioning_mlp_hidden: int = 2048

    # ``online_vjepa`` consumes raw images.  The raw token grid is
    # ``vjepa_views * (image_size / patch_size)^2`` and may be pooled to the
    # configured visual_token_count before entering the language model.
    online_vjepa: bool = True
    compile_vjepa: bool = False
    vjepa_compile_mode: str = "reduce-overhead"
    compile_action_dit: bool = False
    action_dit_compile_mode: str = "reduce-overhead"
    text_max_length: int = 128
    visual_token_count: int = 1152

    # JEPA temporal prediction: details decay with temporal distance.
    jepa_horizons: Tuple[int, ...] = (7, 15, 31)
    # Progressively coarser latent alignment for horizons 7, 15, and 31.
    jepa_token_counts: Tuple[int, ...] = (288, 144, 72)
    jepa_context_dim: int = 1024
    # Number of current visual tokens retained by the JEPA context transformer.
    # Zero uses visual_token_count.
    jepa_observation_tokens: int = 0
    jepa_layers: int = 2
    jepa_heads: int = 8
    jepa_loss_weight: float = 0.5
    # The paper objective disables the optional SIGReg regularizer.
    sigreg_loss_weight: float = 0.0
    sigreg_num_projections: int = 128
    sigreg_num_knots: int = 17
    sigreg_t_max: float = 3.0

    # Memory m_t reads current observation tokens and recent executed actions.
    # Instructions enter through Qwen, not through the memory module.
    memory_dim: int = 1024
    memory_slots: int = 16
    memory_observation_tokens: int = 256
    memory_action_history: int = 8
    memory_layers: int = 2
    memory_heads: int = 8

    # Action DiT follows the GR00T/JEPA-WAM expert: 1536 internal width
    # (32 heads x 48 head dimension), 16 layers, and 32 future tokens. The
    # output projection remains 1024 wide, matching the reference head.
    action_dim: int = 7
    proprio_dim: int = 8
    action_horizon: int = 16
    action_dit_hidden: int = 1024
    action_dit_layers: int = 16
    action_dit_heads: int = 32
    # Zero retains the small-width behavior used by unit-test configurations;
    # the default follows the paper ActionDiT width.
    action_dit_inner_dim: int = 1536
    action_dit_future_tokens: int = 32
    flow_steps: int = 4
    action_loss_weight: float = 1.0

    # Shared LIBERO P0/P1 training-image augmentation; validation is unchanged.
    # Camera/object-position perturbations are image translations only.
    image_augmentations: bool = True
    image_augmentation_probability: float = 0.50
    image_illumination_probability: float = 0.50
    image_texture_probability: float = 0.35
    image_camera_probability: float = 0.25
    image_noise_probability: float = 0.25
    image_object_position_probability: float = 0.25
    image_max_translation: int = 12
    image_noise_std: float = 0.025

    # LIBERO paper defaults; each benchmark recipe specifies its own budget.
    batch_size: int = 24
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_steps: int = 40_000
    num_workers: int = 8
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True
    dataloader_worker_threads: int = 1
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    checkpoint_save_limit: int = 3
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    eval_every: int = 5_000
    eval_batches: int = 32
    eval_fraction: float = 0.1
    tensorboard: bool = True
    # Log latent distributions periodically; every step is unnecessarily
    # expensive for TensorBoard histograms.
    tensorboard_histogram_every: int = 100
    open_loop_eval: bool = True
    save_best_checkpoint: bool = True
    seed: int = 7
    dataset: str = "libero"
    dataset_root: str = "/data/thinkwam/datasets/LIBERO"
    feature_cache: str = ""
    # Cache only frozen future targets; observations still use the online encoder.
    future_target_cache: str = ""
    output_dir: str = "./runs/p0_jepawam"

    def __post_init__(self) -> None:
        for name in (
            "image_augmentation_probability", "image_illumination_probability",
            "image_texture_probability", "image_camera_probability",
            "image_noise_probability", "image_object_position_probability",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.image_max_translation < 0 or not math.isfinite(self.image_noise_std) or self.image_noise_std < 0:
            raise ValueError("Image augmentation magnitudes must be non-negative")
        if self.future_target_cache and (not self.online_vjepa or self.dataset != "robocasa24"):
            raise ValueError("future_target_cache requires online RoboCasa24 observations")
        for name in ("action_context_dim", "conditioning_mlp_hidden", "jepa_context_dim", "memory_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.jepa_horizons) != len(self.jepa_token_counts):
            raise ValueError("jepa_horizons and jepa_token_counts must have the same length")
        if tuple(sorted(self.jepa_horizons)) != self.jepa_horizons:
            raise ValueError("jepa_horizons must be increasing")
        if any(k <= 0 for k in self.jepa_token_counts):
            raise ValueError("jepa_token_counts must be positive")
        if any(a < b for a, b in zip(self.jepa_token_counts, self.jepa_token_counts[1:])):
            raise ValueError("JEPA token counts must decay with horizon")
        if not math.isfinite(self.sigreg_loss_weight) or self.sigreg_loss_weight < 0:
            raise ValueError("sigreg_loss_weight must be non-negative")
        if (self.sigreg_num_projections <= 0 or self.sigreg_num_knots < 2
                or not math.isfinite(self.sigreg_t_max) or self.sigreg_t_max <= 0):
            raise ValueError("SIGReg projection, knot, and t_max settings must be positive")
        if self.action_horizon <= 0 or self.memory_action_history <= 0:
            raise ValueError("action_horizon and memory_action_history must be positive")
        if self.action_dit_inner_dim < 0 or self.action_dit_future_tokens < 0:
            raise ValueError("action_dit_inner_dim and action_dit_future_tokens must be non-negative")
        resolved_inner = self.action_dit_inner_dim or self.action_dit_hidden
        if resolved_inner <= 0 or self.action_dit_heads <= 0 or resolved_inner % self.action_dit_heads:
            raise ValueError("ActionDiT inner width must be positive and divisible by action_dit_heads")
        if self.visual_token_count <= 0:
            raise ValueError("visual_token_count must be positive")
        if self.qwen_visual_tokens < 0 or self.jepa_observation_tokens < 0:
            raise ValueError("qwen_visual_tokens and jepa_observation_tokens must be non-negative")
        if self.qwen_visual_tokens > self.visual_token_count:
            raise ValueError("qwen_visual_tokens cannot exceed visual_token_count")
        if self.jepa_observation_tokens > self.visual_token_count:
            raise ValueError("jepa_observation_tokens cannot exceed visual_token_count")
        full_vjepa_tokens = self.vjepa_views * (self.vjepa_image_size // self.vjepa_patch_size) ** 2
        if self.online_vjepa and self.visual_token_count > full_vjepa_tokens:
            raise ValueError(
                "online_vjepa visual_token_count cannot exceed the raw V-JEPA token count "
                f"({full_vjepa_tokens})"
            )
        if self.qwen_lora and self.qwen_lora_rank <= 0:
            raise ValueError("qwen_lora_rank must be positive")
        if self.qwen_lora_dropout < 0 or self.qwen_lora_dropout >= 1:
            raise ValueError("qwen_lora_dropout must be in [0, 1)")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be between 0 and learning_rate")
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in (0, 1)")
        if self.num_workers < 0 or self.dataloader_prefetch_factor <= 0:
            raise ValueError("num_workers must be non-negative and prefetch_factor positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.checkpoint_save_limit <= 0:
            raise ValueError("checkpoint_save_limit must be positive")
        if self.tensorboard_histogram_every <= 0:
            raise ValueError("tensorboard_histogram_every must be positive")
        if self.mixed_precision not in {"no", "bf16", "fp16"}:
            raise ValueError("mixed_precision must be one of no, bf16, fp16")

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def resolved_qwen_visual_tokens(self) -> int:
        return self.qwen_visual_tokens or self.visual_token_count

    @property
    def resolved_jepa_observation_tokens(self) -> int:
        return self.jepa_observation_tokens or self.visual_token_count
