"""Conditional UNet1D backbone (alternative to transformer)."""
import torch
import torch.nn as nn


class ConditionalUNet1D(nn.Module):
    """Simple 1D UNet with FiLM conditioning for diffusion."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        """x: [B, C, L], cond: [B, D] -> [B, C, L]"""
        c = self.cond_proj(cond).unsqueeze(-1)

        h1 = self.enc1(x)
        h1 = h1 + c
        h2 = self.enc2(h1)
        m = self.mid(h2)
        d2 = self.dec2(m)

        # Handle size mismatch from stride
        if d2.shape[-1] != h1.shape[-1]:
            d2 = d2[..., :h1.shape[-1]]

        d1 = self.dec1(torch.cat([d2, h1], dim=1))
        return d1
