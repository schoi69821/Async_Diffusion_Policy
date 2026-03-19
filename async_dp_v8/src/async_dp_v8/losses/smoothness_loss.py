"""Smoothness loss for action trajectories."""
import torch


def smoothness_loss(actions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Penalize large velocity in predicted action trajectories.

    actions: [B, H, A]
    mask: [B, H] optional
    """
    vel = actions[:, 1:] - actions[:, :-1]  # [B, H-1, A]
    loss = vel.abs()
    if mask is not None:
        vel_mask = mask[:, 1:].unsqueeze(-1)
        loss = (loss * vel_mask).sum() / vel_mask.sum().clamp(min=1) / actions.shape[-1]
    else:
        loss = loss.mean()
    return loss
