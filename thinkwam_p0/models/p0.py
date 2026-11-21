"""P0 model: frozen V-JEPA, Qwen/LoRA, Memory, JEPA predictor and ActionDiT."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..config import P0Config
from ..types import P0Batch
from .action_dit import ActionDiT
from .memory import ObservationActionMemory, _uniform_pool
from .predictor import TemporalJEPA
from .qwen import QwenConditionEncoder


class P0Model(nn.Module):
    def __init__(self, config: P0Config, *, qwen_backend: str = "qwen", vjepa_encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        # The encoder is optional in feature-cache training, but when attached it
        # is always frozen and remains part of the explicit P0 graph contract.
        self.vjepa = vjepa_encoder
        if self.vjepa is not None:
            self.vjepa.requires_grad_(False)
            self.vjepa.eval()
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
        # Keep Qwen's native 896-d hidden state at the Qwen boundary, then
        # project into the configured P0 action-conditioning width. Memory
        # uses a separate projection into the same space.
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

    def train(self, mode: bool = True):
        """Keep the frozen V-JEPA in eval mode when the trainable graph toggles."""
        super().train(mode)
        if self.vjepa is not None:
            self.vjepa.eval()
        return self

    def autocast_context(self):
        """Use the training BF16 policy for validation and action inference too.

        V-JEPA/Qwen use BF16 while the trainable heads keep FP32 parameters.
        eval()/no_grad() alone do not reconcile these mixed dtypes. Preserve
        the existing training policy: CUDA BF16, disabled otherwise.
        """
        device = self.memory_to_action.weight.device
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and self.config.mixed_precision == "bf16",
        )

    def _qwen_visual_input(self, obs_features: torch.Tensor) -> torch.Tensor:
        """Reduce the full V-JEPA grid before the expensive Qwen sequence."""
        return _uniform_pool(obs_features, self.config.resolved_qwen_visual_tokens)

    def compile_vjepa(self, mode: str = "reduce-overhead") -> None:
        """Compile the frozen V-JEPA forward graph after it is placed on CUDA."""
        if self.vjepa is None:
            raise RuntimeError("compile_vjepa requires an attached V-JEPA encoder")
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        # Compile only the tensor-heavy ViT, keeping the small adapter's shape
        # checks and view concatenation in eager Python. This avoids tracing
        # device discovery and dynamic batch/view bookkeeping.
        target = getattr(self.vjepa, "featurizer", self.vjepa)
        compiled = torch.compile(target, mode=mode, fullgraph=False)
        if hasattr(self.vjepa, "featurizer"):
            self.vjepa.featurizer = compiled
        else:
            self.vjepa = compiled

    def compile_action_dit(self, mode: str = "reduce-overhead") -> None:
        """Compile ActionDiT entry points without changing checkpoint keys."""
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        # Compile bound methods instead of replacing the module. This keeps the
        # original state_dict names stable for future resume/evaluation loads.
        # Compile the deterministic tensor core only. Keeping Beta time
        # sampling and the outer flow loop eager avoids CUDA Graph capture of
        # random/inference tensors while still compiling every DiT block.
        self.action_dit._compiled_velocity = torch.compile(
            self.action_dit.velocity,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    def forward(self, batch: P0Batch) -> dict[str, torch.Tensor]:
        batch.validate(
            self.config.jepa_horizons,
            self.config.vjepa_dim,
            self.config.action_horizon,
            visual_token_count=self.config.visual_token_count if self.config.online_vjepa else 0,
            online_vjepa=self.config.online_vjepa,
            vjepa_views=self.config.vjepa_views,
            future_target_cache=bool(self.config.future_target_cache),
        )
        obs_features = batch.obs_features
        future_features = batch.future_features
        if obs_features is None:
            if self.vjepa is None or batch.obs_images is None:
                raise ValueError("online_vjepa requires a frozen V-JEPA and image tensors in P0Batch")
            if future_features is not None:
                obs_features = self.vjepa(batch.obs_images).clone()
                future_features = {h: value.detach() for h, value in future_features.items()}
            else:
                image_groups = [batch.obs_images] + [batch.future_images[h] for h in self.config.jepa_horizons]
                # Clone outside the frozen encoder's inference_mode so trainable
                # heads can save the visual inputs for their backward pass.
                encoded = self.vjepa(torch.cat(image_groups, dim=0)).clone()
                chunks = encoded.split(batch.obs_images.shape[0], dim=0)
                obs_features = chunks[0]
                future_features = {h: chunk for h, chunk in zip(self.config.jepa_horizons, chunks[1:])}
        qwen = self.qwen(
            self._qwen_visual_input(obs_features),
            text_features=batch.text_features,
            text_input_ids=batch.text_input_ids,
            attention_mask=batch.attention_mask,
        )
        memory = self.memory(obs_features, batch.action_history)
        action_context = torch.cat(
            [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(memory)], dim=1
        )
        proprio = batch.proprio
        if proprio is None:
            proprio = batch.actions.new_zeros(batch.actions.shape[0], self.config.proprio_dim)
        loss_action = self.action_dit.loss(action_context, batch.actions, proprio, action_mask=batch.action_mask)
        predictions = self.predictor(qwen["visual_hidden"], memory)
        jepa_losses = self.predictor.loss(
            predictions, future_features, compute_sigreg=self.config.sigreg_loss_weight > 0,
            future_masks=batch.future_masks,
        )
        loss_jepa = jepa_losses["jepa/total"]
        loss_sigreg = jepa_losses["jepa/sigreg"]
        total = (
            self.config.action_loss_weight * loss_action
            + self.config.jepa_loss_weight * loss_jepa
            + self.config.sigreg_loss_weight * loss_sigreg
        )
        output = {
            "loss": total,
            "loss/action_dit": loss_action,
            "loss/jepa": loss_jepa,
            "loss/sigreg": loss_sigreg,
        }
        output.update(jepa_losses)
        output["memory_tokens"] = memory
        # Keep the three temporal latent streams available to the trainer for
        # TensorBoard distribution/collapse diagnostics. Targets are already
        # detached by the JEPA loss path; predictions retain gradients until
        # the caller finishes the optimizer step.
        output["latent_predictions"] = predictions
        output["latent_targets"] = {
            horizon: self.predictor.target_tokens(targets, count).detach()
            for (horizon, count), targets in zip(
                zip(self.config.jepa_horizons, self.config.jepa_token_counts),
                (future_features[horizon] for horizon in self.config.jepa_horizons),
            )
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
    ) -> torch.Tensor:
        """Generate one action chunk for closed-loop evaluators."""
        with self.autocast_context():
            qwen = self.qwen(
                self._qwen_visual_input(obs_features),
                text_features=text_features,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
            )
            memory = self.memory(obs_features, action_history)
            context = torch.cat(
                [self.qwen_action_to_context(qwen["action_hidden"]), self.memory_to_action(memory)], dim=1
            )
            if proprio is None:
                proprio = obs_features.new_zeros(obs_features.shape[0], self.config.proprio_dim)
            return self.action_dit.sample(context, proprio, steps=steps)

    def trainable_parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "qwen": sum(p.numel() for p in self.qwen.parameters()),
        }
