"""Gripper 3-class token head (open/hold/close)."""
import torch.nn as nn

from async_dp_v8.constants import NUM_GRIP_TOKENS


class GripperHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_tokens: int = NUM_GRIP_TOKENS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_tokens),
        )

    def forward(self, h_last):
        """h_last: [B, D] -> [B, num_tokens]"""
        return self.net(h_last)
