"""Proprioceptive state encoder (qpos, qvel, ee_pose, gripper, current, pwm)."""
import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int = 39, out_dim: int = 128):
        super().__init__()
        # in_dim = 6(qpos) + 6(qvel) + 7(ee_pose) + 2(gripper) + 9(current) + 9(pwm) = 39
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, qpos, qvel, ee_pose, gripper, current, pwm):
        """Each input: [B, T, D_i] -> [B, T, out_dim]"""
        x = torch.cat([qpos, qvel, ee_pose, gripper, current, pwm], dim=-1)
        return self.net(x)
