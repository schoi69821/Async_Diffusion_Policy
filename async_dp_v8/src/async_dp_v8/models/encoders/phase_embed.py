"""Learned phase embedding."""
import torch
import torch.nn as nn

from async_dp_v8.constants import NUM_PHASES


class PhaseEmbedding(nn.Module):
    def __init__(self, num_phases: int = NUM_PHASES, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_phases, embed_dim)

    def forward(self, phase_ids: torch.Tensor) -> torch.Tensor:
        """phase_ids: [B] or [B, T] (long) -> [B, embed_dim] or [B, T, embed_dim]"""
        return self.embed(phase_ids)
