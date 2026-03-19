"""Contact prediction head (sigmoid output)."""
import torch.nn as nn


class ContactHead(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h_last):
        """h_last: [B, D] -> [B, 1]"""
        return self.net(h_last)
