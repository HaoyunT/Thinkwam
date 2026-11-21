"""Configuration for the shared two-pass action-refinement model.

LIBERO and RoboCasa24 use this model in P1 after a matching P0 warm start.
RoboTwin uses it in one training stage without a P0 checkpoint. The benchmark
recipes specify architecture, loss weights and within-stage unfreezing.
"""

from __future__ import annotations

from dataclasses import dataclass

from thinkwam_p0.config import P0Config


@dataclass
class P1Config(P0Config):
    # RoboTwin uses the shared action-refinement architecture with guidance
    # heads disabled. Disabled heads are not constructed and require no labels.
    # RGB pixel cache for RoboTwin online V-JEPA (never latent features).
    robotwin_frame_cache: str = ""
    robotwin_manifest: str = "manifest.json"
    robotwin_stats_path: str = ""

    # Released DeVA RoboCasa24 export: three tiled RGB views, 7D absolute
    # actions normalized by the adapter, and per-view simulator affordance maps.
    robocasa24_require_affordance: bool = True
    robocasa_split_manifest: str = ""
    p1_selection_metric: str = "loss"
    # Zero keeps ActionDiT in the shared new_modules optimizer group,
    # preserving the existing checkpoint group layout.
    action_dit_learning_rate: float = 0.0

    affordance_enabled: bool = True

    # Adapt current-observation tokens with a residual adapter while the
    # released V-JEPA encoder remains frozen. The adapter's unfreezing time is
    # configured within the selected benchmark training stage.
    vjepa_adapter_enabled: bool = True
    vjepa_adapter_dim: int = 256
    vjepa_adapter_learning_rate: float = 1.0e-4

    # Latent-space physical guidance head. P1 predicts one affordance logit
    # per visual latent token after the TemporalJEPA context Transformer; a
    # spatial affordance map is accepted by the data adapter and pooled to
    # this token contract.
    affordance_decoder_dim: int = 256
    affordance_decoder_layers: int = 2
    affordance_decoder_heads: int = 8
    affordance_channels: int = 1
    affordance_loss_weight: float = 0.05
    affordance_dice_weight: float = 0.5
    affordance_loss_ramp_steps: int = 3_000
    affordance_target_key: str = "affordance"
    require_affordance_targets: bool = True

    # The action-conditioned predictor produces latent residuals in the
    # 1024-dimensional V-JEPA feature space.
    video_model_dim: int = 1024
    video_model_layers: int = 2
    video_model_heads: int = 8
    # Number of pooled current-observation tokens used by the latent-residual
    # predictor; the feature width is set by video_model_dim.
    video_context_tokens: int = 256
    video_residual_loss_weight: float = 0.2
    action_refinement_loss_weight: float = 1.4
    first_action_loss_weight: float = 1.2
    # Euler integration steps for inference and for the differentiable
    # training proposal that conditions latent residual prediction.
    video_condition_steps: int = 4
    video_condition_train_steps: int = 1
    action_condition_scale: float = 1.0

    # ``staged`` progressively unfreezes modules within LIBERO P1. ``joint``
    # enables configured trainable modules from the first update, as used by
    # RoboTwin and RoboCasa24.
    p1_stage: str = "staged"
    p0_checkpoint: str = ""

    # Optimizer-step thresholds within one training stage. The V-JEPA
    # threshold controls its observation adapter, not the frozen encoder.
    # The optimizer includes parameters that will unfreeze later, so their
    # activation does not require a new optimizer or a new training stage.
    freeze_qwen_steps: int = 10_000
    freeze_vjepa_steps: int = 10_000
    freeze_memory_steps: int = 5_000

    # Zero-initialized affordance-to-memory residual with a confidence gate.
    # Each recipe sets its fusion ramp and the separate gradient scale into
    # shared features; these need not match the affordance-loss ramp.
    affordance_memory_start_step: int = 0
    affordance_memory_ramp_steps: int = 5_000
    affordance_memory_max_scale: float = 0.05
    affordance_confidence_threshold: float = 0.70
    affordance_confidence_temperature: float = 0.10
    affordance_upstream_start_step: int = 0
    affordance_upstream_ramp_steps: int = 5_000
    affordance_upstream_max_scale: float = 0.05

    # Parameter-group learning rates. The paper recipes use a common rate
    # across trainable groups within each benchmark training stage.
    memory_learning_rate: float = 1.0e-4
    qwen_lora_learning_rate: float = 1.0e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.p1_stage not in {"joint", "staged"}:
            raise ValueError("p1_stage must be 'joint' or 'staged'")
        if self.p1_selection_metric not in {
            "loss", "loss/action_dit", "loss/action_dit_first", "loss/action_dit_refined"
        }:
            raise ValueError("unsupported P1 selection metric")
        if self.action_dit_learning_rate < 0:
            raise ValueError("action_dit_learning_rate must be non-negative")
        for name in (
            "affordance_decoder_dim",
            "affordance_decoder_layers",
            "affordance_decoder_heads",
            "affordance_channels",
            "vjepa_adapter_dim",
            "video_model_dim",
            "video_model_layers",
            "video_model_heads",
            "video_context_tokens",
            "video_condition_steps",
            "video_condition_train_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.affordance_loss_weight < 0 or self.affordance_dice_weight < 0:
            raise ValueError("affordance loss weights must be non-negative")
        if self.video_residual_loss_weight < 0 or self.action_refinement_loss_weight < 0:
            raise ValueError("P1 loss weights must be non-negative")
        if self.action_condition_scale < 0:
            raise ValueError("action_condition_scale must be non-negative")
        for name in (
            "memory_learning_rate", "qwen_lora_learning_rate",
            "vjepa_adapter_learning_rate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "freeze_qwen_steps", "freeze_vjepa_steps", "freeze_memory_steps",
            "affordance_memory_start_step", "affordance_memory_ramp_steps",
            "affordance_upstream_start_step", "affordance_upstream_ramp_steps",
            "affordance_loss_ramp_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "affordance_memory_max_scale", "affordance_upstream_max_scale",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.affordance_confidence_threshold <= 1.0:
            raise ValueError("affordance_confidence_threshold must be in [0, 1]")
        if self.affordance_confidence_temperature <= 0:
            raise ValueError("affordance_confidence_temperature must be positive")
