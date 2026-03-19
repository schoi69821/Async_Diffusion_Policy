"""Diffusion MSE loss for arm action prediction."""
import torch
import torch.nn.functional as F


def diffusion_mse_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE loss between predicted and true noise, optionally masked.

    pred_noise: [B, H, A]
    true_noise: [B, H, A]
    mask: [B, H] valid timesteps
    """
    loss = F.mse_loss(pred_noise, true_noise, reduction="none")  # [B, H, A]
    if mask is not None:
        mask = mask.unsqueeze(-1)  # [B, H, 1]
        loss = (loss * mask).sum() / mask.sum().clamp(min=1) / pred_noise.shape[-1]
    else:
        loss = loss.mean()
    return loss
