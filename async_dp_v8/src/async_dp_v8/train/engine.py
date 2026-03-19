"""Training engine for HybridPolicyV8."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Optional
import logging

from async_dp_v8.losses.multitask_loss import compute_losses
from .ema import EMAModel

logger = logging.getLogger(__name__)


class NoiseScheduler:
    """Simple linear noise scheduler for diffusion training."""

    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean actions according to timestep."""
        sqrt_alpha = self.alphas_cumprod[timestep].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[timestep]).sqrt()

        # Reshape for broadcasting
        while sqrt_alpha.dim() < clean.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * clean + sqrt_one_minus_alpha * noise

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


class TrainingEngine:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: NoiseScheduler,
        device: str = "cuda",
        grad_clip: float = 1.0,
        use_amp: bool = True,
        ema_decay: float = 0.995,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.ema = EMAModel(model, decay=ema_decay)
        self._prev_pred_x0 = None

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self._prev_pred_x0 = None
        meters = {}
        count = 0

        for batch in loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            target_actions = batch["target_arm_chunk"]
            B = target_actions.shape[0]
            mask = batch.get("mask_valid", None)

            # Sample random timesteps and noise
            timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)
            noise = torch.randn_like(target_actions)
            noisy_actions = self.scheduler.add_noise(target_actions, noise, timesteps)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(batch, noisy_actions=noisy_actions, timestep=timesteps)
                pred_noise = out["pred_noise"]

                # Reconstruct predicted clean actions (with gradients for smoothness regularization)
                alpha_t = self.scheduler.alphas_cumprod[timesteps]
                while alpha_t.dim() < noisy_actions.dim():
                    alpha_t = alpha_t.unsqueeze(-1)
                pred_x0 = (noisy_actions - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

                losses = compute_losses(
                    batch=batch,
                    out=out,
                    pred_noise=pred_noise,
                    true_noise=noise,
                    arm_chunk_pred=pred_x0,
                    mask=mask,
                    prev_chunk=self._prev_pred_x0,
                )

            self._prev_pred_x0 = pred_x0.detach()

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.ema.update()

            for k, v in losses.items():
                meters.setdefault(k, 0.0)
                meters[k] += float(v.detach().cpu())
            count += 1

        for k in meters:
            meters[k] /= max(count, 1)
        return meters

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.ema.apply_shadow()
        self.model.eval()
        meters = {}
        count = 0

        for batch in loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            target_actions = batch["target_arm_chunk"]
            B = target_actions.shape[0]
            mask = batch.get("mask_valid", None)

            timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)
            noise = torch.randn_like(target_actions)
            noisy_actions = self.scheduler.add_noise(target_actions, noise, timesteps)

            out = self.model(batch, noisy_actions=noisy_actions, timestep=timesteps)
            pred_noise = out["pred_noise"]

            # Reconstruct predicted clean actions for smoothness evaluation
            alpha_t = self.scheduler.alphas_cumprod[timesteps]
            while alpha_t.dim() < noisy_actions.dim():
                alpha_t = alpha_t.unsqueeze(-1)
            pred_x0 = (noisy_actions - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

            losses = compute_losses(
                batch=batch,
                out=out,
                pred_noise=pred_noise,
                true_noise=noise,
                arm_chunk_pred=pred_x0,
                mask=mask,
            )

            for k, v in losses.items():
                meters.setdefault(k, 0.0)
                meters[k] += float(v.detach().cpu())
            count += 1

        self.ema.restore()

        for k in meters:
            meters[k] /= max(count, 1)
        return meters
