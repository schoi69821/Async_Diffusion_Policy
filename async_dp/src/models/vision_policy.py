"""
Vision-based Diffusion Policy (v2)
Fixed architecture: observation conditioning injected at EVERY residual block
via global conditioning (obs + time → FiLM at each layer).

Based on Chi et al. "Diffusion Policy" (2023).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision import models


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    """1D ResBlock with FiLM conditioning from global condition (obs + time)."""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        # FiLM: condition → scale & shift
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x: (B, C, T)
        cond: (B, cond_dim) global condition
        """
        h = self.blocks[0](x)  # conv1
        h = self.blocks[1](h)  # groupnorm

        # FiLM conditioning
        params = self.cond_proj(cond)  # (B, out_channels*2)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        h = h * (1 + gamma) + beta

        h = self.blocks[2](h)  # mish
        h = self.blocks[3](h)  # conv2
        h = self.blocks[4](h)  # groupnorm
        h = self.blocks[5](h)  # mish

        return h + self.residual(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D UNet for diffusion policy.
    Global conditioning (obs + timestep) injected at EVERY ResBlock via FiLM.

    For channels=(256, 512):
      Encoder: [input→256, T] → skip0 → [512, T/2]
      Bottleneck: [512, T/2]
      Decoder: upsample → concat(skip0) → [256, T] → output
    """

    def __init__(self, input_dim, global_cond_dim, channels=(256, 512)):
        super().__init__()
        self.n_levels = len(channels)
        cond_dim = global_cond_dim * 2  # obs + time concatenated

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(global_cond_dim),
            nn.Linear(global_cond_dim, global_cond_dim * 4),
            nn.Mish(),
            nn.Linear(global_cond_dim * 4, global_cond_dim),
        )

        # Input: action_dim → channels[0]
        self.input_proj = nn.Conv1d(input_dim, channels[0], 1)

        # Encoder: each level has a ResBlock, then downsample (except last)
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.n_levels):
            self.enc_blocks.append(ConditionalResBlock1D(channels[i], channels[i], cond_dim))
            if i < self.n_levels - 1:
                self.downsamples.append(nn.Conv1d(channels[i], channels[i+1], 3, stride=2, padding=1))

        # Bottleneck
        self.mid_block = ConditionalResBlock1D(channels[-1], channels[-1], cond_dim)

        # Decoder: upsample, concat skip, ResBlock
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(self.n_levels - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose1d(channels[i], channels[i-1], 4, stride=2, padding=1))
            # After concat: channels[i-1] (upsampled) + channels[i-1] (skip) = channels[i-1]*2
            self.dec_blocks.append(ConditionalResBlock1D(channels[i-1] * 2, channels[i-1], cond_dim))

        # Output: channels[0] → action_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.Mish(),
            nn.Conv1d(channels[0], input_dim, 1),
        )

    def forward(self, x, timestep, global_cond):
        """
        x: (B, input_dim, T)
        timestep: (B,)
        global_cond: (B, global_cond_dim)
        Returns: (B, input_dim, T)
        """
        t_emb = self.time_emb(timestep)
        cond = torch.cat([global_cond, t_emb], dim=-1)

        h = self.input_proj(x)

        # Encoder: save skips from all levels except the deepest
        skips = []
        for i in range(self.n_levels):
            h = self.enc_blocks[i](h, cond)
            if i < self.n_levels - 1:
                skips.append(h)
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.mid_block(h, cond)

        # Decoder: upsample, concat skip (reverse order), block
        for upsample, block in zip(self.upsamples, self.dec_blocks):
            h = upsample(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.output_proj(h)


class VisionEncoder(nn.Module):
    """ResNet18-based image encoder."""

    def __init__(self, feature_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, feature_dim)

        # Freeze early layers, fine-tune last 2
        children = list(self.backbone.children())
        for child in children[:6]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, img):
        feat = self.backbone(img).flatten(1)
        return self.proj(feat)


class VisionDiffusionPolicy(nn.Module):
    """
    Diffusion Policy with Vision + Proprioception.

    Architecture (v2 - fixed):
        1. VisionEncoder: image → 256-dim
        2. QposEncoder: joints → 256-dim
        3. Fusion → 256-dim observation embedding
        4. ConditionalUNet1D: obs + time conditioning at EVERY layer
    """

    def __init__(self, action_dim=7, qpos_dim=7, hidden_dim=256, pred_horizon=16, obs_horizon=1):
        super().__init__()
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        # Image encoder
        self.vision_encoder = VisionEncoder(feature_dim=hidden_dim)

        # Joint position encoder
        self.qpos_encoder = nn.Sequential(
            nn.Linear(qpos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Progress encoder (0.0 → 1.0 scalar → hidden_dim)
        self.progress_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fuse (image + joints) * obs_horizon + progress → observation embedding
        fuse_input_dim = hidden_dim * 2 * obs_horizon + hidden_dim  # (img+qpos)*T + progress
        self.obs_fuse = nn.Sequential(
            nn.Linear(fuse_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Conditional UNet1D: obs conditioning at every ResBlock
        self.noise_pred_net = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=hidden_dim,
            channels=(256, 512),
        )

        # Image normalization (ImageNet stats)
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_image(self, img):
        """Normalize uint8 HWC image to float CHW tensor with ImageNet stats."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(self.img_mean.device)
        return (img - self.img_mean) / self.img_std

    def encode_obs(self, img, qpos, progress=None):
        """
        Encode observations. Supports obs_horizon >= 1.

        Args:
            img: (B, 3, H, W) for obs_horizon=1, or (B, T, 3, H, W) for obs_horizon>1
            qpos: (B, qpos_dim) for obs_horizon=1, or (B, T, qpos_dim) for obs_horizon>1
            progress: (B, 1)
        """
        if img.dim() == 4 and self.obs_horizon == 1:
            # Single frame: (B, 3, H, W) → encode directly
            img_feat = self.vision_encoder(img)
            qpos_feat = self.qpos_encoder(qpos)
            feats = [img_feat, qpos_feat]
        elif img.dim() == 5:
            # Multi-frame: (B, T, 3, H, W)
            B, T = img.shape[:2]
            img_flat = img.reshape(B * T, *img.shape[2:])
            img_feat = self.vision_encoder(img_flat).reshape(B, T * self.hidden_dim)
            qpos_flat = qpos.reshape(B * T, -1)
            qpos_feat = self.qpos_encoder(qpos_flat).reshape(B, T * self.hidden_dim)
            feats = [img_feat, qpos_feat]
        else:
            # Fallback for backward compat
            img_feat = self.vision_encoder(img)
            qpos_feat = self.qpos_encoder(qpos)
            feats = [img_feat, qpos_feat]

        if progress is None:
            progress = torch.zeros(img.shape[0], 1, device=img.device)
        progress_feat = self.progress_encoder(progress)
        feats.append(progress_feat)

        fused = torch.cat(feats, dim=-1)
        return self.obs_fuse(fused)

    def forward(self, noisy_action, timestep, img, qpos, progress=None):
        """
        Args:
            noisy_action: (B, pred_horizon, action_dim)
            timestep: (B,)
            img: (B, 3, 224, 224) normalized image
            qpos: (B, qpos_dim)
            progress: (B, 1) task progress 0.0 ~ 1.0
        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        obs_emb = self.encode_obs(img, qpos, progress)  # (B, hidden_dim)

        # (B, action_dim, horizon)
        x = noisy_action.transpose(1, 2)
        noise_pred = self.noise_pred_net(x, timestep, obs_emb)

        return noise_pred.transpose(1, 2)  # (B, pred_horizon, action_dim)

    @torch.no_grad()
    def get_action(self, img, qpos, scheduler, num_inference_steps=16,
                   device='cpu', progress=0.0, generator=None):
        """
        Inference: denoise random noise → action trajectory.

        Args:
            img: np.ndarray (H,W,3) or list of np.ndarrays for obs_horizon>1
            qpos: np.ndarray (qpos_dim,) or list of np.ndarrays for obs_horizon>1
            generator: optional torch.Generator for deterministic denoising
            progress: float 0.0~1.0 indicating task progress
        """
        self.eval()

        # Handle observation horizon
        if isinstance(img, list):
            # Multi-frame: list of np.ndarrays → (1, T, 3, H, W)
            imgs = [self.normalize_image(i).squeeze(0) for i in img]  # each (3,H,W)
            img_tensor = torch.stack(imgs).unsqueeze(0).to(device)    # (1, T, 3, H, W)
        elif isinstance(img, np.ndarray):
            img_tensor = self.normalize_image(img).to(device)         # (1, 3, H, W)
        else:
            img_tensor = img.unsqueeze(0) if img.dim() == 3 else img

        if isinstance(qpos, list):
            # Multi-frame: list of np.ndarrays → (1, T, qpos_dim)
            qps = [torch.from_numpy(q).float() if isinstance(q, np.ndarray) else q for q in qpos]
            qpos_tensor = torch.stack(qps).unsqueeze(0).to(device)   # (1, T, qpos_dim)
        elif isinstance(qpos, np.ndarray):
            qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
        else:
            qpos_tensor = qpos.unsqueeze(0).to(device) if qpos.dim() == 1 else qpos.to(device)

        prog_tensor = torch.tensor([[progress]], dtype=torch.float32, device=device)

        # Denoise with optional deterministic generator
        noisy_action = torch.randn(1, self.pred_horizon, self.action_dim,
                                   device=device, generator=generator)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            noise_pred = self(noisy_action, t.unsqueeze(0).to(device),
                            img_tensor, qpos_tensor, prog_tensor)
            noisy_action = scheduler.step(
                noise_pred.transpose(1, 2),
                t,
                noisy_action.transpose(1, 2)
            ).prev_sample.transpose(1, 2)

        return noisy_action.squeeze(0).cpu().numpy()
