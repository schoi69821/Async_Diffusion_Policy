"""Phase classification head."""
import torch.nn as nn

from async_dp_v8.constants import NUM_PHASES


class PhaseHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_phases: int = NUM_PHASES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_phases),
        )

    def forward(self, h_last):
        """h_last: [B, D] -> [B, num_phases]"""
        return self.net(h_last)
