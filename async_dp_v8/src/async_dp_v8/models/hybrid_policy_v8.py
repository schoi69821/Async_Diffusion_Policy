"""HybridPolicyV8: multi-head policy with arm diffusion, phase, gripper, contact."""
import torch
import torch.nn as nn

from .encoders.vision_encoder import VisionEncoder
from .encoders.crop_encoder import CropEncoder
from .encoders.proprio_encoder import ProprioEncoder
from .backbones.temporal_transformer import TemporalBackbone
from .heads.phase_head import PhaseHead
from .heads.gripper_head import GripperHead
from .heads.contact_head import ContactHead
from .heads.arm_diffusion_head import ArmDiffusionHead

from async_dp_v8.constants import PRED_HORIZON, ARM_ACTION_DIM


class HybridPolicyV8(nn.Module):
    def __init__(
        self,
        pred_horizon: int = PRED_HORIZON,
        action_dim: int = ARM_ACTION_DIM,
        vision_out: int = 256,
        crop_out: int = 128,
        proprio_out: int = 128,
        backbone_dim: int = 512,
        backbone_heads: int = 8,
        backbone_depth: int = 4,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # Encoders
        self.vision = VisionEncoder(out_dim=vision_out)
        self.crop = CropEncoder(out_dim=crop_out)
        self.proprio = ProprioEncoder(out_dim=proprio_out)

        # Fusion
        self.fuse = nn.Linear(vision_out + crop_out + proprio_out, backbone_dim)

        # Backbone
        self.backbone = TemporalBackbone(
            d_model=backbone_dim,
            nhead=backbone_heads,
            depth=backbone_depth,
        )

        # Heads
        self.phase_head = PhaseHead(backbone_dim)
        self.gripper_head = GripperHead(backbone_dim)
        self.contact_head = ContactHead(backbone_dim)
        self.arm_head = ArmDiffusionHead(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            cond_dim=backbone_dim,
        )

    def encode_obs(self, batch):
        B, T, C, H, W = batch["obs_image_wrist"].shape

        # Vision encoding
        xg = batch["obs_image_wrist"].reshape(B * T, C, H, W)
        global_feat = self.vision(xg).reshape(B, T, -1)

        # Crop encoding
        _, _, C2, H2, W2 = batch["obs_image_crop"].shape
        xc = batch["obs_image_crop"].reshape(B * T, C2, H2, W2)
        crop_feat = self.crop(xc).reshape(B, T, -1)

        # Proprio encoding
        proprio_feat = self.proprio(
            batch["obs_qpos"],
            batch["obs_qvel"],
            batch["obs_ee_pose"],
            batch["obs_gripper"],
            batch["obs_current"],
            batch["obs_pwm"],
        )

        # Fuse and run through backbone
        fused = torch.cat([global_feat, crop_feat, proprio_feat], dim=-1)
        fused = self.fuse(fused)
        h = self.backbone(fused)
        return h

    def forward(self, batch, noisy_actions=None, timestep=None):
        h = self.encode_obs(batch)
        h_last = h[:, -1]  # Use last timestep features

        out = {
            "phase_logits": self.phase_head(h_last),
            "grip_logits": self.gripper_head(h_last),
            "contact_logit": self.contact_head(h_last),
        }

        if noisy_actions is not None and timestep is not None:
            out["pred_noise"] = self.arm_head(noisy_actions, h_last, timestep)

        return out
