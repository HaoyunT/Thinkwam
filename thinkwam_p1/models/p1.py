"""Shared proposal-conditioned refinement model for the appendix recipes.

LIBERO and RoboCasa24 use affordance guidance; RoboTwin disables that head.
The visual backbone remains frozen, with a trainable observation adapter.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from thinkwam_p0.models.action_dit import ActionDiT
from thinkwam_p0.models.memory import ObservationActionMemory, _uniform_pool
from thinkwam_p0.models.predictor import TemporalJEPA
from thinkwam_p0.models.qwen import QwenConditionEncoder

from ..config import P1Config
from ..types import P1Batch
from .affordance import LatentAffordanceDecoder
from .vjepa_adapter import VJEPAResidualAdapter
from .video_residual import ActionConditionedVideoResidualModel


class P1Model(nn.Module):
    """Frozen visual features, scheduled affordance guidance, and two action passes.

    The affordance decoder reads the temporal predictor's visual context.
    A confidence-gated residual updates memory using the selected recipe's
    fusion and upstream-gradient schedules within the same training stage.
    """

    def __init__(
        self,
        config: P1Config,
        *,
        qwen_backend: str = "qwen",
        vjepa_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vjepa = vjepa_encoder
        if self.vjepa is not None:
            # Only the current-observation adapter is optimized. The same
            # frozen encoder supplies current features and future targets.
            self.vjepa.requires_grad_(False)
            self.vjepa.eval()
            self.vjepa_adapter = (
                VJEPAResidualAdapter(config.vjepa_dim, config.vjepa_adapter_dim)
                if config.vjepa_adapter_enabled else nn.Identity()
            )
        else:
            self.vjepa_adapter = nn.Identity()
        self.qwen = QwenConditionEncoder(
            vjepa_dim=config.vjepa_dim,
            qwen_dim=config.qwen_dim,
            qwen_path=config.qwen_path,
            action_tokens=config.qwen_action_tokens,
            max_length=config.qwen_max_length,
            backend=qwen_backend,
            gradient_checkpointing=config.gradient_checkpointing,
            lora=config.qwen_lora,
            lora_rank=config.qwen_lora_rank,
            lora_alpha=config.qwen_lora_alpha,
            lora_dropout=config.qwen_lora_dropout,
            lora_target_modules=config.qwen_lora_target_modules,
            attn_implementation=config.qwen_attn_implementation,
            hidden_only=config.qwen_hidden_only,
        )
        self.memory = ObservationActionMemory(
            config.vjepa_dim,
            config.action_dim,
            config.memory_dim,
            config.memory_slots,
            config.memory_observation_tokens,
            config.memory_action_history,
            config.memory_layers,
            config.memory_heads,
        )
        self.predictor = TemporalJEPA(
            config.vjepa_dim,
            config.qwen_dim,
            config.memory_dim,
            config.jepa_horizons,
            config.jepa_token_counts,
            config.jepa_context_dim,
            config.jepa_layers,
            config.jepa_heads,
            config.resolved_jepa_observation_tokens,
            config.conditioning_mlp_hidden,
            config.sigreg_num_projections,
            config.sigreg_num_knots,
            config.sigreg_t_max,
        )
        self.qwen_action_to_context = nn.Sequential(
            nn.Linear(config.qwen_dim, config.conditioning_mlp_hidden),
            nn.GELU(),
            nn.Linear(config.conditioning_mlp_hidden, config.action_context_dim),
        )
        self.memory_to_action = nn.Linear(config.memory_dim, config.action_context_dim)
        self.action_dit = ActionDiT(
            action_dim=config.action_dim,
            proprio_dim=config.proprio_dim,
            context_dim=config.action_context_dim,
            horizon=config.action_horizon,
            hidden=config.action_dit_hidden,
            layers=config.action_dit_layers,
            heads=config.action_dit_heads,
            inner_dim=config.action_dit_inner_dim or config.action_dit_hidden,
            future_tokens=config.action_dit_future_tokens,
            inference_steps=config.flow_steps,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        self.affordance = (
            LatentAffordanceDecoder(
                config.jepa_context_dim,
                config.affordance_decoder_dim,
                config.affordance_decoder_layers,
                config.affordance_decoder_heads,
                config.affordance_channels,
            )
            if config.affordance_enabled else None
        )
        self.video_residual = ActionConditionedVideoResidualModel(
            config.vjepa_dim,
            config.action_dim,
            config.video_model_dim,
            config.video_model_layers,
            config.video_model_heads,
            config.jepa_horizons,
            config.jepa_token_counts,
            context_tokens=config.video_context_tokens,
        )
        self.visual_to_action = nn.ModuleDict(
            {
                f"t{horizon}": nn.Sequential(
                    nn.Linear(config.vjepa_dim, config.action_context_dim),
                    nn.GELU(),
                    nn.Linear(config.action_context_dim, config.action_context_dim),
                )
                for horizon in config.jepa_horizons
            }
        )
        # Zero initialization makes the affordance memory contribution zero
        # before this adapter learns, as specified in the appendix.
        if config.affordance_enabled:
            self.affordance_to_memory = nn.Sequential(
                nn.Linear(config.jepa_context_dim, config.memory_dim),
                nn.GELU(),
                nn.Linear(config.memory_dim, config.memory_dim),
            )
            nn.init.zeros_(self.affordance_to_memory[-1].weight)
            nn.init.zeros_(self.affordance_to_memory[-1].bias)
        else:
            self.affordance_to_memory = None
        # Keep the original trainability contract (for example Qwen LoRA
        # leaves the base model frozen) and restore only those parameters when
        # a staged freeze expires.
        self._baseline_trainable = {
            "qwen": {id(parameter): parameter.requires_grad for parameter in self.qwen.parameters()},
            "vjepa": ({id(parameter): parameter.requires_grad for parameter in self.vjepa.parameters()}
                       if self.vjepa is not None else {}),
            "memory": {id(parameter): parameter.requires_grad for parameter in self.memory.parameters()},
            "predictor": {id(parameter): parameter.requires_grad for parameter in self.predictor.parameters()},
            "vjepa_adapter": {
                id(parameter): parameter.requires_grad
                for parameter in self.vjepa_adapter.parameters()
            },
        }
        self._stage_step = -1
        self.set_stage(0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vjepa is not None:
            self.vjepa.eval()
        if mode:
            # Re-entering training after validation must preserve the modes
            # of frozen modules, even when set_stage sees the same step again.
            step = max(0, self._stage_step)
            staged = self.config.p1_stage == "staged"
            self.qwen.train(not staged or step >= self.config.freeze_qwen_steps)
            memory_open = not staged or step >= self.config.freeze_memory_steps
            self.memory.train(memory_open)
            self.predictor.train(memory_open)
            self.vjepa_adapter.train(self.config.vjepa_adapter_enabled and
                                     (not staged or step >= self.config.freeze_vjepa_steps))
        return self

    @staticmethod
    def _ramp(step: int, start: int, duration: int, maximum: float) -> float:
        if maximum <= 0 or step < start:
            return 0.0
        if duration <= 0:
            return float(maximum)
        return float(maximum) * min(1.0, float(step - start) / float(duration))

    def _set_module_trainable(self, module: nn.Module, enabled: bool, baseline: dict[int, bool]) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(bool(enabled and baseline.get(id(parameter), False)))

    def set_stage(self, global_step: int) -> None:
        """Apply the staged freeze schedule before a forward pass."""
        step = int(global_step)
        if step == self._stage_step:
            return
        self._stage_step = step
        staged = self.config.p1_stage == "staged"
        qwen_open = not staged or step >= self.config.freeze_qwen_steps
        for name, parameter in self.qwen.named_parameters():
            # Staged P1 keeps the P0 projector/placeholders frozen. Joint
            # training also learns these heads, with or without a P0 warm start.
            is_lora = "lora_" in name.lower()
            baseline = self._baseline_trainable["qwen"].get(id(parameter), False)
            trainable_head = self.config.p1_stage == "joint" or is_lora
            parameter.requires_grad_(bool(qwen_open and trainable_head and baseline))
        if self.training:
            self.qwen.train(qwen_open)
        if self.vjepa is not None:
            # The encoder remains frozen throughout; the observation adapter
            # follows the within-stage unfreezing schedule.
            self._set_module_trainable(self.vjepa, False, self._baseline_trainable["vjepa"])
            adapter_open = not staged or step >= self.config.freeze_vjepa_steps
            self._set_module_trainable(
                self.vjepa_adapter,
                bool(self.config.vjepa_adapter_enabled and adapter_open),
                self._baseline_trainable["vjepa_adapter"],
            )
            if self.training:
                self.vjepa.eval()
                self.vjepa_adapter.train(bool(self.config.vjepa_adapter_enabled and adapter_open))
        memory_open = not staged or step >= self.config.freeze_memory_steps
        self._set_module_trainable(self.memory, memory_open, self._baseline_trainable["memory"])
        self._set_module_trainable(self.predictor, memory_open, self._baseline_trainable["predictor"])
        if self.training:
            self.memory.train(memory_open)
            self.predictor.train(memory_open)

    def _affordance_gradient_scale(self, step: int) -> float:
        return self._ramp(
            step,
            self.config.affordance_upstream_start_step,
            self.config.affordance_upstream_ramp_steps,
            self.config.affordance_upstream_max_scale,
        )

    @staticmethod
    def _gradient_interpolation(value: torch.Tensor, scale: float) -> torch.Tensor:
        if scale <= 0:
            return value.detach()
        detached = value.detach()
        return detached + float(scale) * (value - detached)

    def autocast_context(self):
        device = self.memory_to_action.weight.device
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and self.config.mixed_precision == "bf16",
        )

    def _qwen_visual_input(self, obs_features: torch.Tensor) -> torch.Tensor:
        """Reduce the full V-JEPA grid before the Qwen sequence."""
        return _uniform_pool(obs_features, self.config.resolved_qwen_visual_tokens)

    def compile_vjepa(self, mode: str = "reduce-overhead") -> None:
        """Compile in place so saved V-JEPA parameter names remain stable."""
        if self.vjepa is None:
            raise ValueError("compile_vjepa requires online_vjepa with an encoder")
        self.vjepa.featurizer.compile(mode=mode, fullgraph=False)

    def compile_action_dit(self, mode: str = "reduce-overhead") -> None:
        """Match P0's compiled velocity core and stable checkpoint keys."""
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        # P1 reuses the velocity graph across differentiable rollouts and
        # flow losses. As in P0, disable CUDA Graph capture for these calls.
        self.action_dit._compiled_velocity = torch.compile(
            self.action_dit.velocity,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    def _latent_affordance_target(self, target, visual_token_count: int):
        if target is None or self.affordance is None:
            return None
        # Match the two uniform token selections on the visual feature path.
        # Direct average pooling to the context length would label different locations.
        target = self.affordance.canonicalize_target(
            target, batch_size=target.shape[0], token_count=visual_token_count,
            channels=self.config.affordance_channels,
        )
        return _uniform_pool(
            _uniform_pool(target, self.config.resolved_qwen_visual_tokens),
            self.config.resolved_jepa_observation_tokens,
        )

    def _encode_images(self, batch: P1Batch):
        if self.vjepa is None or batch.obs_images is None:
            raise ValueError("image-mode P1 requires a frozen observation encoder and image tensors")
        # Frozen BF16 encoder outputs cross a float32 boundary into the
        # trainable adapter. CUDA autocast handles subsequent projections.
        observation = self.vjepa_adapter(self.vjepa(batch.obs_images).float())
        if batch.future_features is not None:
            return observation, {h: value.detach() for h, value in batch.future_features.items()}
        # Future targets use the frozen encoder without the observation
        # adapter and never receive gradients.
        future_images = torch.cat(
            [batch.future_images[h] for h in self.config.jepa_horizons], dim=0
        )
        with torch.no_grad():
            future_encoded = self.vjepa(future_images).float()
        batch_size = batch.obs_images.shape[0]
        future_chunks = future_encoded.split(batch_size, dim=0)
        return observation, {
            h: target.detach() for h, target in zip(self.config.jepa_horizons, future_chunks)
        }

    @staticmethod
    def _zeros_like_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def _correct_futures(
        self,
        predictions: Dict[int, torch.Tensor],
        residuals: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Return ``sg(z_hat) + delta_z`` for every horizon.

        The base prediction supplies the forward value of the visual reference,
        but the stop-gradient keeps the refinement path from back-propagating
        into ``z_hat``. The second-pass action loss therefore trains the
        proposal-conditioned residual through this addition, while the base
        predictor is still trained by its own JEPA loss on the undetached
        ``predictions``. The detach only cuts this additive path; shared
        memory/language modules can still receive second-pass gradients
        through the first-pass action context and the differentiable proposal.
        """
        return {
            horizon: predictions[horizon].detach() + residuals[horizon]
            for horizon in self.config.jepa_horizons
        }

    @staticmethod
    def _normalize_zt_for_action(zt: torch.Tensor) -> torch.Tensor:
        """Normalize each corrected future token before ActionDiT injection.

        Residual regression compares the raw delta with a stop-gradient
        ``future - base`` target. The raw ``sg(base) + residual`` sum is then
        L2-normalized along the feature dimension before projection into the
        refined ActionDiT context.
        This is parameter-free and deliberately does not add a residual gate.
        """
        # Accumulate the norm in FP32 under BF16/FP16 training, then restore
        # the feature dtype at the projection boundary.
        values = zt.float() if zt.dtype in (torch.bfloat16, torch.float16) else zt
        return F.normalize(values, p=2.0, dim=-1, eps=1e-6).to(dtype=zt.dtype)

    def _affordance_losses(
        self,
        logits: torch.Tensor,
        target: Optional[torch.Tensor],
        prefix: str,
        sample_mask: Optional[torch.Tensor] = None,
    ):
        if target is None:
            zero = self._zeros_like_loss(logits)
            values = {"bce": zero, "dice": zero, "total": zero}
        else:
            raw = self.affordance.loss(logits, target, sample_mask=sample_mask)
            values = {
                "bce": raw["affordance/bce"],
                "dice": raw["affordance/dice"],
                "total": raw["affordance/bce"] + self.config.affordance_dice_weight * raw["affordance/dice"],
            }
        return {f"affordance/{prefix}_{key}": value for key, value in values.items()}

    def _empty_affordance_losses(self, reference: torch.Tensor, prefix: str):
        zero = self._zeros_like_loss(reference)
        return {
            f"affordance/{prefix}_bce": zero,
            f"affordance/{prefix}_dice": zero,
            f"affordance/{prefix}_total": zero,
        }

    def _fuse_guidance(
        self,
        memory: torch.Tensor,
        latent_visual: torch.Tensor,
        affordance_logits: Optional[torch.Tensor],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory plus its confidence-gated affordance residual."""
        fused = memory
        zero_gate = memory.new_zeros((memory.shape[0],))
        affordance_gate = zero_gate
        if self.config.affordance_enabled:
            assert affordance_logits is not None
            assert self.affordance_to_memory is not None
            probability = affordance_logits.sigmoid()
            confidence_per_token = (probability - 0.5).abs().mul(2.0)
            confidence = confidence_per_token.mean(dim=(1, 2)).detach()
            confidence_gate = torch.sigmoid(
                (confidence - self.config.affordance_confidence_threshold)
                / self.config.affordance_confidence_temperature
            )
            affordance_summary = (latent_visual * confidence_per_token.detach()).sum(dim=1)
            # s_aff = sum(rho * U) / (sum(rho) + delta), with stop-gradient rho.
            normalizer = confidence_per_token.detach().sum(dim=1) + 1e-5
            affordance_summary = affordance_summary / normalizer
            affordance_scale = self._ramp(
                step,
                self.config.affordance_memory_start_step,
                self.config.affordance_memory_ramp_steps,
                self.config.affordance_memory_max_scale,
            )
            affordance_gate = affordance_scale * confidence_gate
            affordance_delta = self.affordance_to_memory(affordance_summary).unsqueeze(1)
            fused = fused + affordance_gate[:, None, None] * affordance_delta

        return fused, affordance_gate

    def _differentiable_action_condition(
        self,
        context: torch.Tensor,
        proprio: torch.Tensor,
        *,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Make the first ActionDiT rollout differentiable for P1 conditioning."""
        if steps is None:
            steps = (
                self.config.video_condition_train_steps
                if self.training else self.config.video_condition_steps
            )
        steps = max(1, int(steps))
        # Match ActionDiT.sample's flow-matching initial distribution. Using
        # zeros here while inference starts from Gaussian noise makes the
        # residual branch see a different action-condition distribution.
        actions = torch.randn(
            context.shape[0], self.config.action_horizon, self.config.action_dim,
            device=context.device, dtype=context.dtype,
        )
        dt = 1.0 / float(steps)
        velocity_fn = self.action_dit._compiled_velocity or self.action_dit.velocity
        for index in range(steps):
            t = torch.full((context.shape[0],), index / float(steps), device=context.device, dtype=context.dtype)
            actions = actions + dt * velocity_fn(actions, t, proprio, context)
        return actions * self.config.action_condition_scale

    def forward(
        self,
        batch: P1Batch,
        *,
        metrics_only: bool = False,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor | Dict[int, torch.Tensor]]:
        self.set_stage(global_step)
        batch.validate_p1(
            horizons=self.config.jepa_horizons,
            vjepa_dim=self.config.vjepa_dim,
            action_horizon=self.config.action_horizon,
            visual_token_count=self.config.visual_token_count,
            online_vjepa=self.config.online_vjepa,
            vjepa_views=self.config.vjepa_views,
            require_affordance_targets=self.config.require_affordance_targets,
            future_target_cache=bool(self.config.future_target_cache),
        )
        obs_features = batch.obs_features
        future_features = batch.future_features
        if obs_features is None:
            obs_features, future_features = self._encode_images(batch)
        else:
            future_features = {h: value.detach() for h, value in future_features.items()}

        qwen = self.qwen(
            self._qwen_visual_input(obs_features),
            text_features=batch.text_features,
            text_input_ids=batch.text_input_ids,
            attention_mask=batch.attention_mask,
        )
        memory = self.memory(obs_features, batch.action_history)
        proprio = batch.proprio
        if proprio is None:
            proprio = batch.actions.new_zeros(batch.actions.shape[0], self.config.proprio_dim)

        # The trailing memory slots have no spatial affordance labels, so only
        # the visual prefix is sent to the optional guidance heads.
        predictions, latent_context = self.predictor(
            qwen["visual_hidden"], memory, return_context=True
        )
        visual_token_count = self.predictor.observation_tokens
        latent_visual = latent_context[:, :visual_token_count]
        if self.config.affordance_enabled:
            assert self.affordance is not None
            # The recipe independently schedules the supervised loss weight
            # and the fraction of its gradients entering shared features.
            affordance_input = self._gradient_interpolation(
                latent_visual, self._affordance_gradient_scale(global_step)
            )
            affordance_logits = self.affordance(affordance_input)
            affordance_losses = self._affordance_losses(
                affordance_logits,
                self._latent_affordance_target(batch.affordance_targets, obs_features.shape[1]),
                "latent",
                sample_mask=batch.affordance_mask,
            )
            affordance_losses["affordance/total"] = affordance_losses["affordance/latent_total"]
        else:
            affordance_logits = None
            affordance_losses = self._empty_affordance_losses(latent_visual, "latent")
            affordance_losses["affordance/total"] = affordance_losses["affordance/latent_total"]

        fused_memory, affordance_gate = self._fuse_guidance(
            memory, latent_visual, affordance_logits, global_step
        )

        base_context = torch.cat(
            [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(fused_memory)], dim=1
        )
        first_actions = self._differentiable_action_condition(base_context, proprio)

        jepa_losses = self.predictor.loss(
            predictions,
            future_features,
            compute_sigreg=self.config.sigreg_loss_weight > 0,
            future_masks=batch.future_masks,
        )
        residuals = self.video_residual(obs_features, first_actions)
        video_losses = self.video_residual.loss(residuals, predictions, future_features, future_masks=batch.future_masks)
        # z_tilde = sg(z_hat) + delta_z: the second-pass action loss trains the
        # residual branch through this addition without updating z_hat directly.
        corrected_visual = self._correct_futures(predictions, residuals)
        normalized_zt = {
            horizon: self._normalize_zt_for_action(corrected_visual[horizon])
            for horizon in self.config.jepa_horizons
        }
        correction_context = torch.cat(
            [self.visual_to_action[f"t{horizon}"](normalized_zt[horizon]) for horizon in self.config.jepa_horizons],
            dim=1,
        )
        # Corrected JEPA tokens are appended to the latent-guided action
        # context. No affordance decoder is placed after this ActionDiT pass.
        refined_context = torch.cat(
            [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(fused_memory), correction_context], dim=1
        )
        # This rollout is only a diagnostic tensor. It is not consumed by any
        # loss; skip it in the metrics-only training path while keeping it for
        # full forward inspection and inference-oriented callers.
        refined_actions = (
            None
            if metrics_only else self._differentiable_action_condition(refined_context, proprio)
        )

        loss_first = self.action_dit.loss(base_context, batch.actions, proprio, action_mask=batch.action_mask)
        loss_refined = self.action_dit.loss(refined_context, batch.actions, proprio, action_mask=batch.action_mask)
        loss_jepa = jepa_losses["jepa/total"]
        loss_sigreg = jepa_losses["jepa/sigreg"]
        loss_video = video_losses["video/residual"]
        loss_affordance = affordance_losses["affordance/total"]
        affordance_loss_scale = loss_affordance.new_tensor(self._ramp(
            global_step, 0, self.config.affordance_loss_ramp_steps,
            self.config.affordance_loss_weight,
        ))
        loss_action = (
            self.config.first_action_loss_weight * loss_first
            + self.config.action_refinement_loss_weight * loss_refined
        )
        total = (
            self.config.action_loss_weight * loss_action
            + self.config.jepa_loss_weight * loss_jepa
            + self.config.sigreg_loss_weight * loss_sigreg
            + self.config.video_residual_loss_weight * loss_video
            + affordance_loss_scale * loss_affordance
        )
        output: dict[str, torch.Tensor | Dict[int, torch.Tensor]] = {
            "loss": total,
            "loss/action_dit": loss_refined,
            "loss/action_dit_first": loss_first,
            "loss/action_dit_refined": loss_refined,
            "loss/jepa": loss_jepa,
            "loss/sigreg": loss_sigreg,
            "loss/video_residual": loss_video,
            "loss/affordance": loss_affordance,
            "schedule/affordance_loss_weight": affordance_loss_scale,
            "first_actions": first_actions,
            "refined_actions": refined_actions,
            "affordance_logits": affordance_logits,
            "affordance_logits_latent": affordance_logits,
            "memory_tokens": memory,
            "memory_tokens_refined": fused_memory,
            "affordance_gate": affordance_gate.mean(),
            "latent_context": latent_context,
            "corrected_visual_tokens": corrected_visual,
            "normalized_zt_tokens": normalized_zt,
        }
        output.update(jepa_losses)
        output.update(video_losses)
        output.update(affordance_losses)
        if metrics_only:
            # DDP traces only the optimized loss; diagnostic tensors are
            # detached so they do not retain unrelated graph roots.
            return {
                key: value if key == "loss" else value.detach()
                for key, value in output.items()
                if isinstance(value, torch.Tensor) and value.ndim == 0
            }
        return output

    @torch.no_grad()
    def predict(
        self,
        obs_features: torch.Tensor,
        action_history: torch.Tensor,
        *,
        proprio: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Run optional guidance -> visual correction -> refined action."""
        with self.autocast_context():
            qwen = self.qwen(
                self._qwen_visual_input(obs_features),
                text_features=text_features,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
            )
            memory = self.memory(obs_features, action_history)
            predictions, latent_context = self.predictor(
                qwen["visual_hidden"], memory, return_context=True
            )
            visual_token_count = self.predictor.observation_tokens
            latent_visual = latent_context[:, :visual_token_count]
            affordance_logits = self.affordance(latent_visual) if self.config.affordance_enabled else None
            inference_step = (
                self.config.affordance_memory_start_step + self.config.affordance_memory_ramp_steps
                if global_step is None else int(global_step)
            )
            memory, _affordance_gate = self._fuse_guidance(
                memory, latent_visual, affordance_logits, inference_step
            )
            base_context = torch.cat(
                [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(memory)], dim=1
            )
            if proprio is None:
                proprio = obs_features.new_zeros(obs_features.shape[0], self.config.proprio_dim)
            first_actions = self.action_dit.sample(
                base_context,
                proprio,
                steps=steps or self.config.video_condition_steps,
            )
            action_condition = first_actions * self.config.action_condition_scale
            residuals = self.video_residual(obs_features, action_condition)
            corrected = self._correct_futures(predictions, residuals)
            normalized_zt = {
                h: self._normalize_zt_for_action(corrected[h])
                for h in self.config.jepa_horizons
            }
            correction_context = torch.cat(
                [self.visual_to_action[f"t{h}"](normalized_zt[h]) for h in self.config.jepa_horizons], dim=1
            )
            refined_context = torch.cat(
                [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(memory), correction_context], dim=1
            )
            return self.action_dit.sample(refined_context, proprio, steps=steps)

    def trainable_parameter_summary(self) -> dict[str, int]:
        vjepa = sum(p.numel() for p in self.vjepa.parameters()) if self.vjepa is not None else 0
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "vjepa": vjepa,
            "vjepa_adapter": sum(p.numel() for p in self.vjepa_adapter.parameters()),
            "qwen": sum(p.numel() for p in self.qwen.parameters()),
            "affordance": sum(p.numel() for p in self.affordance.parameters()) if self.affordance is not None else 0,
            "affordance_to_memory": sum(p.numel() for p in self.affordance_to_memory.parameters()) if self.affordance_to_memory is not None else 0,
            "video_residual": sum(p.numel() for p in self.video_residual.parameters()),
        }
