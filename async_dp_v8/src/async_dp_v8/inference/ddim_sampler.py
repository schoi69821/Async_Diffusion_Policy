"""DDIM (Denoising Diffusion Implicit Models) sampler for fast inference.

Reduces 100-step DDPM denoising to 10-20 steps with minimal quality loss.
"""
import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DDIMSampler:
    """DDIM deterministic sampler for fast action trajectory generation."""

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        eta: float = 0.0,
    ):
        self.alphas_cumprod = alphas_cumprod
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta  # 0 = deterministic DDIM, 1 = DDPM

        # Build sub-sequence of timesteps (evenly spaced)
        self.timesteps = self._build_timesteps()
        logger.info(
            f"DDIMSampler: {num_inference_steps} steps, "
            f"eta={eta}, timesteps={self.timesteps.tolist()}"
        )

    def _build_timesteps(self) -> torch.Tensor:
        """Create evenly-spaced timestep sub-sequence."""
        step_ratio = self.num_train_steps // self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps = np.flip(timesteps).copy()  # Reverse: high noise → low noise
        return torch.from_numpy(timesteps).long()

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    @torch.no_grad()
    def sample(
        self,
        model,
        obs_batch: dict,
        shape: tuple,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Run DDIM sampling to generate clean action trajectories.

        Args:
            model: HybridPolicyV8 model
            obs_batch: observation batch dict
            shape: (B, H, A) shape of action tensor
            device: torch device

        Returns:
            Denoised action tensor [B, H, A]
        """
        x = torch.randn(shape, device=device)

        for i, t in enumerate(self.timesteps):
            timestep = torch.full((shape[0],), t.item(), dtype=torch.long, device=device)

            out = model(obs_batch, noisy_actions=x, timestep=timestep)
            pred_noise = out["pred_noise"]

            # Current and previous alpha_cumprod
            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(self.timesteps):
                alpha_prev = self.alphas_cumprod[self.timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # Predict x0
            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

            # Clamp x0 for stability
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # DDIM update
            sigma = self.eta * (
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            ).sqrt()

            dir_xt = (1 - alpha_prev - sigma ** 2).sqrt() * pred_noise
            x = alpha_prev.sqrt() * x0_pred + dir_xt

            if sigma > 0:
                x = x + sigma * torch.randn_like(x)

        return x
