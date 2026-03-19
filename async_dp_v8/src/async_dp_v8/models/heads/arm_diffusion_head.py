"""Arm diffusion head: predicts noise for 6-axis arm actions only."""
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1d(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.act = nn.GELU()

    def forward(self, x, cond):
        # FiLM conditioning
        scale_shift = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return x + h


class ArmDiffusionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        pred_horizon: int = 12,
        cond_dim: int = 512,
        hidden_dim: int = 256,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition projection (concat cond + time)
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Input projection
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)

        # Residual blocks with FiLM conditioning
        self.blocks = nn.ModuleList([
            ConditionalResBlock1d(hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        cond: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy_actions: [B, H, A] noisy action trajectory
        cond: [B, D] conditioning from backbone
        timestep: [B] diffusion timestep
        Returns: [B, H, A] predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(timestep)

        # Fuse condition with time
        cond_fused = self.cond_fuse(torch.cat([cond, t_emb], dim=-1))

        # Conv1d expects [B, C, L]
        x = noisy_actions.transpose(1, 2)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, cond_fused)

        x = self.output_proj(x)
        return x.transpose(1, 2)  # [B, H, A]
