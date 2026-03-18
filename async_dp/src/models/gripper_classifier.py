"""
Binary gripper classifier: open vs closed.
Separate from diffusion policy because gripper is bimodal (open/closed)
and diffusion models average the two modes.

Uses same observation (image + qpos) as the diffusion policy.
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class GripperClassifier(nn.Module):
    """
    Binary classifier: P(gripper should be closed | image, qpos, progress).

    Architecture:
        ResNet18 (frozen early layers) → 256-dim
        QPos encoder → 128-dim
        Progress → 64-dim
        Concat → FC → sigmoid
    """

    def __init__(self, qpos_dim=7, obs_horizon=1):
        super().__init__()
        self.obs_horizon = obs_horizon

        # Image encoder (lightweight, shared architecture with VisionEncoder)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, 256)

        # Freeze early layers
        for child in list(self.backbone.children())[:6]:
            for param in child.parameters():
                param.requires_grad = False

        # QPos encoder
        self.qpos_enc = nn.Sequential(
            nn.Linear(qpos_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Progress encoder
        self.progress_enc = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Classifier head
        feat_dim = (256 + 128) * obs_horizon + 64
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Image normalization
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_image(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(self.img_mean.device)
        return (img - self.img_mean) / self.img_std

    def forward(self, img, qpos, progress=None):
        """
        Args:
            img: (B, 3, H, W) or (B, T, 3, H, W)
            qpos: (B, qpos_dim) or (B, T, qpos_dim)
            progress: (B, 1)
        Returns:
            logits: (B, 1) — apply sigmoid for probability
        """
        if img.dim() == 5:
            B, T = img.shape[:2]
            img_flat = img.reshape(B * T, *img.shape[2:])
            img_feat = self.img_proj(self.backbone(img_flat).flatten(1))
            img_feat = img_feat.reshape(B, T * 256)
            qpos_flat = qpos.reshape(B * T, -1)
            qpos_feat = self.qpos_enc(qpos_flat).reshape(B, T * 128)
        else:
            img_feat = self.img_proj(self.backbone(img).flatten(1))
            qpos_feat = self.qpos_enc(qpos)

        if progress is None:
            progress = torch.zeros(img.shape[0], 1, device=img.device)
        prog_feat = self.progress_enc(progress)

        feat = torch.cat([img_feat, qpos_feat, prog_feat], dim=-1)
        return self.head(feat)

    @torch.no_grad()
    def predict(self, img, qpos, progress=0.0, device='cpu'):
        """
        Inference: returns probability of gripper closed.

        Args:
            img: np.ndarray or list of np.ndarrays
            qpos: np.ndarray or list of np.ndarrays (normalized)
        Returns:
            float: probability of closed (0.0 = open, 1.0 = closed)
        """
        self.eval()

        if isinstance(img, list):
            imgs = [self.normalize_image(i).squeeze(0) for i in img]
            img_t = torch.stack(imgs).unsqueeze(0).to(device)
        elif isinstance(img, np.ndarray):
            img_t = self.normalize_image(img).to(device)
        else:
            img_t = img.unsqueeze(0).to(device) if img.dim() == 3 else img.to(device)

        if isinstance(qpos, list):
            qps = [torch.from_numpy(q).float() if isinstance(q, np.ndarray) else q for q in qpos]
            qpos_t = torch.stack(qps).unsqueeze(0).to(device)
        elif isinstance(qpos, np.ndarray):
            qpos_t = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
        else:
            qpos_t = qpos.unsqueeze(0).to(device) if qpos.dim() == 1 else qpos.to(device)

        prog_t = torch.tensor([[progress]], dtype=torch.float32, device=device)
        logit = self(img_t, qpos_t, prog_t)
        return torch.sigmoid(logit).item()
