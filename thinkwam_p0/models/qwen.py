"""Qwen condition encoder with optional JEPA-WAM-style LoRA tuning."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class QwenConditionEncoder(nn.Module):
    """Encode visual tokens, text, and learned action placeholders with Qwen.

    ``backend='qwen'`` loads the local Qwen2.5-0.5B checkpoint.  With ``lora``
    enabled, the Qwen backbone is wrapped with JEPA-WAM-compatible LoRA and
    only its adapter weights are trainable; the visual projection and action
    placeholders are separate trainable heads before the enclosing model's
    freeze schedule is applied. ``backend='tiny'`` is only for fast CPU tests.
    """

    def __init__(
        self,
        vjepa_dim: int,
        qwen_dim: int = 896,
        qwen_path: Optional[str] = None,
        action_tokens: int = 64,
        max_length: int = 512,
        backend: str = "qwen",
        gradient_checkpointing: bool = False,
        lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_target_modules: str = "all-linear",
        attn_implementation: str = "sdpa",
        hidden_only: bool = False,
        tiny_layers: int = 2,
        tiny_heads: int = 4,
    ) -> None:
        super().__init__()
        self.qwen_dim = qwen_dim
        self.action_tokens = action_tokens
        self.backend = backend
        self.lora = bool(lora)
        self.hidden_only = bool(hidden_only)
        # Match JEPA-WAM's released GELU projector and its deterministic
        # projector seed (the reference seeds from the vision embed dim).
        # fork_rng avoids changing the caller's global training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(vjepa_dim)
            self.visual_proj = nn.Sequential(
                nn.Linear(vjepa_dim, qwen_dim),
                nn.GELU(),
                nn.Linear(qwen_dim, qwen_dim),
            )
        self.action_placeholders = nn.Parameter(torch.randn(1, action_tokens, qwen_dim) * 0.02)

        if backend == "qwen":
            if not qwen_path:
                raise ValueError("qwen_path is required when backend='qwen'")
            from transformers import AutoModelForCausalLM

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            qwen_model = AutoModelForCausalLM.from_pretrained(
                qwen_path,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            if self.lora:
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:  # pragma: no cover - environment-specific
                    raise RuntimeError("qwen_lora=True requires the `peft` package") from exc

                peft_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target_modules,
                )
                self.qwen = get_peft_model(qwen_model, peft_config)
            else:
                self.qwen = qwen_model
            if gradient_checkpointing and hasattr(self.qwen, "gradient_checkpointing_enable"):
                # Non-reentrant checkpointing supports DDP unused-parameter detection.
                self.qwen.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                # Qwen receives inputs_embeds rather than input_ids.  PEFT's
                # checkpointed path needs input gradients to reach LoRA layers.
                if hasattr(self.qwen, "enable_input_require_grads"):
                    self.qwen.enable_input_require_grads()
            actual_dim = int(self.qwen.config.hidden_size)
            if actual_dim != qwen_dim:
                raise ValueError(f"qwen_dim={qwen_dim} disagrees with checkpoint hidden_size={actual_dim}")
            self.text_proj = nn.Identity()
        elif backend == "tiny":
            layer = nn.TransformerEncoderLayer(qwen_dim, tiny_heads, 4 * qwen_dim, batch_first=True, norm_first=True)
            self.qwen = nn.TransformerEncoder(layer, num_layers=tiny_layers)
            self.text_proj = nn.Linear(qwen_dim, qwen_dim)
        else:
            raise ValueError(f"Unknown Qwen backend: {backend}")

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # Project in the projector's parameter dtype, then cast the visual
        # prefix to Qwen's embedding dtype below.
        projector_dtype = next(self.visual_proj.parameters()).dtype
        visual = self.visual_proj(visual_tokens.to(dtype=projector_dtype))
        if self.backend == "qwen":
            qwen_dtype = self.qwen.get_input_embeddings().weight.dtype
            visual = visual.to(dtype=qwen_dtype)
            if text_input_ids is not None:
                text = self.qwen.get_input_embeddings()(text_input_ids)
                text_mask = attention_mask if attention_mask is not None else text_input_ids.ne(0)
            elif text_features is not None:
                text = text_features
                text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            else:
                text = visual.new_zeros(visual.shape[0], 1, self.qwen_dim)
                text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            text = text.to(dtype=visual.dtype)
        else:
            if text_features is None:
                text = visual.new_zeros(visual.shape[0], 1, self.qwen_dim)
            elif text_features.ndim == 2:
                text = text_features[:, None, :]
            else:
                text = self.text_proj(text_features)
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)

        action = self.action_placeholders.expand(visual.shape[0], -1, -1).to(dtype=visual.dtype)
        embeddings = torch.cat([visual, text, action], dim=1)
        mask = torch.cat(
            [
                torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device),
                text_mask.to(dtype=torch.bool),
                torch.ones(action.shape[:2], dtype=torch.bool, device=visual.device),
            ],
            dim=1,
        )
        if self.backend == "qwen":
            if self.hidden_only:
                # PEFT has already injected LoRA into the decoder layers. Keep
                # the same registered model/state_dict and skip only lm_head.
                causal_lm = self.qwen.get_base_model() if self.lora else self.qwen
                output = causal_lm.model(
                    inputs_embeds=embeddings,
                    attention_mask=mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state
            else:
                output = self.qwen(
                    inputs_embeds=embeddings,
                    attention_mask=mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = output.hidden_states[-1]
        else:
            hidden = self.qwen(embeddings, src_key_padding_mask=~mask)
        visual_count = visual.shape[1]
        text_count = text.shape[1]
        return {
            "hidden": hidden,
            "visual_hidden": hidden[:, :visual_count],
            "text_hidden": hidden[:, visual_count : visual_count + text_count],
            "action_hidden": hidden[:, -self.action_tokens :],
            "attention_mask": mask,
        }
