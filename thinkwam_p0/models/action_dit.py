"""GR00T-style flow-matching ActionDiT shared by the P0 and P1 policies.

The reference JEPA-WAM action expert uses a 16-layer DiT with an internal
1536-dimensional attention stream (32 heads x 48 dimensions), AdaLayerNorm
time conditioning, alternating cross/self-attention blocks, and learnable
future tokens. P0 uses one action context; P1 reuses the same head for the
proposal and refinement contexts through its ``velocity``/``loss``/``sample``
interfaces.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .loss_utils import ddp_mean_from_sum


class _TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by the DiT MLP."""

    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.proj = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.frequency_dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps[:, None] * freq[None, :]
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if embedding.shape[-1] < self.frequency_dim:
            embedding = F.pad(embedding, (0, self.frequency_dim - embedding.shape[-1]))
        return self.proj(embedding.to(dtype=next(self.parameters()).dtype))


class _AdaLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are generated from the flow time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _FusedAttention(nn.Module):
    """Multi-head attention backed by SDPA and a fused self-attention QKV.

    Self-attention uses one ``Linear(dim, 3 * dim)`` projection so Q, K, and
    V are produced by one GEMM.  Cross-attention cannot use one QKV matrix
    because the query stream and context stream have different widths; it uses
    separate Q and K/V projections while still dispatching the attention
    product through ``scaled_dot_product_attention`` (FlashAttention when the
    CUDA kernel supports the input).
    """

    def __init__(self, dim: int, heads: int, dropout: float, kv_dim: int, *, fused_qkv: bool) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention dim={dim} must be divisible by heads={heads}")
        self.embed_dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.kdim = kv_dim
        self.vdim = kv_dim
        self.dropout = float(dropout)
        # The query and context widths can happen to match while still being
        # different streams. The caller therefore specifies whether this is
        # self-attention instead of inferring it from the dimensions.
        self.fused_qkv = bool(fused_qkv)
        if self.fused_qkv and kv_dim != dim:
            raise ValueError("fused self-attention requires matching query/key-value widths")
        if self.fused_qkv:
            self.qkv = nn.Linear(dim, 3 * dim)
        else:
            # Separate projections support different query/context widths and
            # expose each key/value parameter's gradient for inspection.
            self.q_proj_weight = nn.Parameter(torch.empty(dim, dim))
            self.k_proj_weight = nn.Parameter(torch.empty(dim, kv_dim))
            self.v_proj_weight = nn.Parameter(torch.empty(dim, kv_dim))
            self.q_proj_bias = nn.Parameter(torch.empty(dim))
            self.k_proj_bias = nn.Parameter(torch.empty(dim))
            self.v_proj_bias = nn.Parameter(torch.empty(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.fused_qkv:
            nn.init.xavier_uniform_(self.qkv.weight)
            nn.init.zeros_(self.qkv.bias)
        else:
            for weight in (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight):
                nn.init.xavier_uniform_(weight)
            for bias in (self.q_proj_bias, self.k_proj_bias, self.v_proj_bias):
                nn.init.zeros_(bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if self.fused_qkv:
            q, k, v = self.qkv(query).chunk(3, dim=-1)
        else:
            q = F.linear(query, self.q_proj_weight, self.q_proj_bias)
            k = F.linear(key, self.k_proj_weight, self.k_proj_bias)
            v = F.linear(value, self.v_proj_weight, self.v_proj_bias)
        batch, query_length, _ = q.shape
        q = q.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.embed_dim)
        return self.out_proj(attended), None


class _GR00TBlock(nn.Module):
    """One reference-style DiT block.

    Even-indexed blocks cross-attend to Qwen/Memory context and odd-indexed
    blocks perform self-attention, matching ``interleave_self_attention`` in
    the released GR00T action head.
    """

    def __init__(self, dim: int, heads: int, dropout: float, use_cross_attention: bool, context_dim: int) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.norm1 = _AdaLayerNorm(dim)
        # Reference diffusers Attention applies dropout to its output, not to
        # softmax attention weights. Self-attention uses a fused QKV projection;
        # cross K/V project directly from context.
        kv_dim = context_dim if use_cross_attention else dim
        self.attn = _FusedAttention(
            dim,
            heads,
            dropout=0.0,
            kv_dim=kv_dim,
            fused_qkv=not use_cross_attention,
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.attention_output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x, time_embedding)
        if self.use_cross_attention:
            update, _ = self.attn(normalized, context, context)
        else:
            update, _ = self.attn(normalized, normalized, normalized)
        x = x + self.attention_output_dropout(update)
        return x + self.ff(self.norm3(x))


class _ActionEncoder(nn.Module):
    """Embed noisy actions together with a per-example flow timestep."""

    def __init__(self, action_dim: int, dim: int) -> None:
        super().__init__()
        self.action = nn.Linear(action_dim, dim)
        self.frequency_dim = dim
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timesteps.float()[:, None, None] * freq[None, None, :]
        time = torch.cat([angles.sin(), angles.cos()], dim=-1)
        time = time.expand(-1, actions.shape[1], -1)
        if time.shape[-1] < self.frequency_dim:
            time = F.pad(time, (0, self.frequency_dim - time.shape[-1]))
        return self.proj(torch.cat([self.action(actions), time.to(dtype=actions.dtype)], dim=-1))


class ActionDiT(nn.Module):
    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        context_dim: int,
        horizon: int = 16,
        hidden: int = 1024,
        layers: int = 16,
        heads: int = 32,
        inference_steps: int = 4,
        use_gradient_checkpointing: bool = False,
        *,
        inner_dim: int | None = None,
        future_tokens: int = 32,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        dropout: float = 0.2,
        state_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.inference_steps = inference_steps
        self.inner_dim = int(inner_dim or hidden)
        self.hidden = hidden
        self.future_token_count = future_tokens
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._compiled_velocity = None
        if self.inner_dim % heads != 0:
            raise ValueError(f"ActionDiT inner_dim={self.inner_dim} must be divisible by heads={heads}")

        self.time = _TimestepEmbedding(self.inner_dim)
        self.action_in = _ActionEncoder(action_dim, self.inner_dim)
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.inner_dim))
        self.state_dropout = nn.Dropout(state_dropout)
        self.future = nn.Embedding(future_tokens, self.inner_dim)
        nn.init.normal_(self.future.weight, mean=0.0, std=0.02)
        self.pos = nn.Embedding(max(1024, horizon), self.inner_dim)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _GR00TBlock(self.inner_dim, heads, dropout, use_cross_attention=(idx % 2 == 0), context_dim=context_dim)
                for idx in range(layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 2 * self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, hidden)
        self.action_decoder_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, action_dim)

    def _prepare_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        return proprio

    def velocity(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        proprio: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        proprio = self._prepare_proprio(proprio)
        # Use the reference 0..1000 timestep scale for sinusoidal/AdaLN
        # conditioning while retaining the public normalized t interface.
        timestep = (t * 1000.0).long()
        time_embedding = self.time(timestep)
        action = self.action_in(noisy_actions, timestep)
        positions = torch.arange(noisy_actions.shape[1], device=noisy_actions.device)
        action = action + self.pos(positions)[None, :, :]
        state = self.state_dropout(self.proprio(proprio)).unsqueeze(1)
        future = self.future.weight[None, :, :].expand(noisy_actions.shape[0], -1, -1)
        x = torch.cat([state, future, action], dim=1)
        ctx = context
        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(
                    lambda hidden, conditioning, temb, current_block=block: current_block(
                        hidden, conditioning, temb
                    ),
                    x,
                    ctx,
                    time_embedding,
                    use_reentrant=False,
                )
            else:
                x = block(x, ctx, time_embedding)
        shift, scale = self.out_modulation(time_embedding).chunk(2, dim=-1)
        x = self.out_norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = self.out_proj(x)
        return self.out(self.action_decoder_hidden(x[:, -noisy_actions.shape[1] :]))

    def loss(
        self, context: torch.Tensor, actions: torch.Tensor,
        proprio: torch.Tensor | None = None, action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if proprio is None:
            proprio = actions.new_zeros(actions.shape[0], self.proprio[0].in_features)
        if action_mask is not None:
            if action_mask.dtype != torch.bool or action_mask.shape != actions.shape[:2]:
                raise ValueError("action_mask must be bool [B,action_horizon]")
            # Padded labels must not influence valid tokens through self-attention.
            actions = torch.where(action_mask[..., None], actions, torch.zeros_like(actions))
        noise = torch.randn_like(actions)
        if action_mask is not None:
            # Keep padded action tokens inert in the DiT self-attention stream;
            # masking their loss alone would still let random padding affect
            # predictions at valid timesteps.
            noise = torch.where(action_mask[..., None], noise, torch.zeros_like(noise))
        beta = torch.distributions.Beta(self.noise_beta_alpha, self.noise_beta_beta)
        sample = beta.sample((actions.shape[0],)).to(device=actions.device, dtype=actions.dtype)
        # Match the released head's reversed Beta schedule, including its
        # noise_s scaling (not direct Beta sampling of the interpolation time).
        t = (0.999 - sample) / 0.999
        noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
        target = actions - noise
        velocity_fn = self._compiled_velocity or self.velocity
        error = (velocity_fn(noisy, t, proprio, context) - target).square()
        if action_mask is None:
            if dist.is_available() and dist.is_initialized():
                return ddp_mean_from_sum(error.sum(), error.numel())[0]
            return error.mean()
        valid = action_mask[..., None].expand_as(error)
        local_sum = torch.where(valid, error, torch.zeros_like(error)).sum()
        return ddp_mean_from_sum(local_sum, valid.sum())[0]

    @torch.no_grad()
    def sample(self, context: torch.Tensor, proprio: torch.Tensor | None = None, steps: int | None = None) -> torch.Tensor:
        if proprio is None:
            proprio = context.new_zeros(context.shape[0], self.proprio[0].in_features)
        steps = steps or self.inference_steps
        actions = torch.randn(
            context.shape[0], self.horizon, self.action_dim, device=context.device, dtype=context.dtype
        )
        dt = 1.0 / steps
        velocity_fn = self._compiled_velocity or self.velocity
        for i in range(steps):
            t = torch.full((context.shape[0],), i / steps, device=context.device, dtype=context.dtype)
            actions = actions + dt * velocity_fn(actions, t, proprio, context)
        return actions
