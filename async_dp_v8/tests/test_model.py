"""Test model forward pass."""
import torch
from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8


def test_hybrid_policy_forward():
    model = HybridPolicyV8(pred_horizon=12, action_dim=6)
    B, T = 2, 3
    batch = {
        "obs_image_wrist": torch.randn(B, T, 3, 224, 224),
        "obs_image_crop": torch.randn(B, T, 3, 96, 96),
        "obs_qpos": torch.randn(B, T, 6),
        "obs_qvel": torch.randn(B, T, 6),
        "obs_ee_pose": torch.randn(B, T, 7),
        "obs_gripper": torch.randn(B, T, 2),
        "obs_current": torch.randn(B, T, 7),
        "obs_pwm": torch.randn(B, T, 7),
    }
    noisy = torch.randn(B, 12, 6)
    timestep = torch.randint(0, 100, (B,))

    out = model(batch, noisy_actions=noisy, timestep=timestep)
    assert "phase_logits" in out
    assert "grip_logits" in out
    assert "contact_logit" in out
    assert "pred_noise" in out
    assert out["phase_logits"].shape == (B, 6)
    assert out["grip_logits"].shape == (B, 3)
    assert out["contact_logit"].shape == (B, 1)
    assert out["pred_noise"].shape == (B, 12, 6)


def test_hybrid_policy_no_diffusion():
    model = HybridPolicyV8()
    B, T = 1, 3
    batch = {
        "obs_image_wrist": torch.randn(B, T, 3, 224, 224),
        "obs_image_crop": torch.randn(B, T, 3, 96, 96),
        "obs_qpos": torch.randn(B, T, 6),
        "obs_qvel": torch.randn(B, T, 6),
        "obs_ee_pose": torch.randn(B, T, 7),
        "obs_gripper": torch.randn(B, T, 2),
        "obs_current": torch.randn(B, T, 7),
        "obs_pwm": torch.randn(B, T, 7),
    }
    out = model(batch)
    assert "pred_noise" not in out
    assert out["phase_logits"].shape == (1, 6)
