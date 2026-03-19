"""Keypose prediction head (future keyposes for planning)."""
import torch
import torch.nn as nn


class KeyposeHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_keyposes: int = 3, pose_dim: int = 7):
        super().__init__()
        self.num_keyposes = num_keyposes
        self.pose_dim = pose_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_keyposes * pose_dim),
        )

    def forward(self, h_last):
        """h_last: [B, D] -> [B, K, 7]"""
        out = self.net(h_last)
        return out.reshape(-1, self.num_keyposes, self.pose_dim)
