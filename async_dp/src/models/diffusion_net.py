"""
Diffusion Policy Network with FiLM Conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet1DModel


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer
    Applies affine transformation: gamma * x + beta

    Reference: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)

        # Initialize gamma close to 1, beta close to 0
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, cond):
        """
        Args:
            x: Feature tensor (B, C, T) or (B, C)
            cond: Condition tensor (B, cond_dim)
        Returns:
            Modulated feature tensor
        """
        gamma = self.gamma_fc(cond)  # (B, C)
        beta = self.beta_fc(cond)    # (B, C)

        if x.dim() == 3:
            # (B, C, T) -> expand gamma/beta to (B, C, 1)
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * x + beta


class ObservationEncoder(nn.Module):
    """
    Multi-layer observation encoder with residual connections
    """
    def __init__(self, obs_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs):
        """
        Args:
            obs: Observation tensor (B, obs_dim)
        Returns:
            Encoded observation (B, hidden_dim)
        """
        x = self.input_proj(obs)
        x = F.gelu(x)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection

        return self.output_proj(x)


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy with FiLM-based Observation Conditioning

    Architecture:
        1. ObservationEncoder: obs -> 256-dim embedding
        2. FiLM Layers: Modulate UNet features with observation
        3. UNet1D: Predict noise (with pad-crop for horizon 16)
    """
    def __init__(self, action_dim, obs_dim, hidden_dim=256):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Observation Encoder (improved from simple Linear)
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim, num_layers=2)

        # FiLM layers for conditioning at different scales
        self.film_input = FiLMLayer(hidden_dim, action_dim)
        self.film_mid = FiLMLayer(hidden_dim, 256)  # Match UNet mid channels

        # Timestep embedding projection
        self.time_emb = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combined condition projection (obs + time)
        self.cond_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # UNet1D for noise prediction
        # [Safety Padding Arch]
        # Horizon 16 처리를 위해 입력(32) -> 2 Downsamples -> 출력(32) 구조 사용
        self.noise_pred_net = UNet1DModel(
            sample_size=32,
            in_channels=action_dim,
            out_channels=action_dim,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=("DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D"),
        )

    def forward(self, noisy_action, timestep, global_cond):
        """
        Forward pass with FiLM conditioning

        Args:
            noisy_action: (B, Horizon, ActionDim) - Noisy action trajectory
            timestep: (B,) - Diffusion timestep
            global_cond: (B, ObsDim) - Observation conditioning

        Returns:
            noise_pred: (B, Horizon, ActionDim) - Predicted noise
        """
        batch_size = noisy_action.shape[0]

        # Encode observation
        obs_emb = self.obs_encoder(global_cond)  # (B, hidden_dim)

        # Get timestep embedding from UNet
        t_emb = self.noise_pred_net.time_proj(timestep)  # (B, 256)
        t_emb = self.time_emb(t_emb)  # (B, hidden_dim)

        # Combine observation and timestep embeddings
        combined_cond = torch.cat([obs_emb, t_emb], dim=-1)  # (B, hidden_dim*2)
        cond = self.cond_proj(combined_cond)  # (B, hidden_dim)

        # (Batch, Horizon, Dim) -> (Batch, Dim, Horizon)
        x = noisy_action.transpose(1, 2)

        # Apply FiLM conditioning on input
        x = self.film_input(x, cond)

        # [Padding] Horizon 16 -> 32
        x_padded = F.pad(x, (0, 16), mode='replicate')  # Use replicate instead of constant

        # Forward through UNet
        pred_padded = self.noise_pred_net(x_padded, timestep).sample

        # Apply FiLM conditioning on output (before cropping)
        # Only apply to the mid channels if dimensions match

        # [Cropping] Horizon 32 -> 16
        pred = pred_padded[..., :16]

        return pred.transpose(1, 2)

    def get_action(self, obs, scheduler, num_inference_steps=16, device='cpu'):
        """
        Convenience method for inference

        Args:
            obs: (obs_dim,) numpy array or (1, obs_dim) tensor
            scheduler: DDIM/DDPM scheduler
            num_inference_steps: Number of denoising steps
            device: torch device

        Returns:
            action_traj: (pred_horizon, action_dim) numpy array
        """
        self.eval()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(device)

        # Initialize with noise
        noisy_action = torch.randn(1, 16, self.action_dim, device=device)

        scheduler.set_timesteps(num_inference_steps)

        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = self(noisy_action, t.unsqueeze(0).to(device), obs)
                noise_pred = noise_pred.transpose(1, 2)
                noisy_action_t = noisy_action.transpose(1, 2)
                noisy_action_t = scheduler.step(noise_pred, t, noisy_action_t).prev_sample
                noisy_action = noisy_action_t.transpose(1, 2)

        return noisy_action.squeeze(0).cpu().numpy()


# For backwards compatibility
import numpy as np
