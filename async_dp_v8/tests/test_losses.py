"""Test loss functions."""
import torch
from async_dp_v8.losses.diffusion_loss import diffusion_mse_loss
from async_dp_v8.losses.focal_loss import focal_cross_entropy
from async_dp_v8.losses.smoothness_loss import smoothness_loss
from async_dp_v8.losses.multitask_loss import compute_losses


def test_diffusion_mse_loss():
    pred = torch.randn(4, 12, 6)
    true = torch.randn(4, 12, 6)
    loss = diffusion_mse_loss(pred, true)
    assert loss.shape == ()
    assert loss > 0


def test_diffusion_mse_loss_masked():
    pred = torch.randn(4, 12, 6)
    true = torch.randn(4, 12, 6)
    mask = torch.ones(4, 12)
    mask[:, 8:] = 0
    loss = diffusion_mse_loss(pred, true, mask)
    assert loss.shape == ()


def test_focal_ce():
    logits = torch.randn(8, 6)
    targets = torch.randint(0, 6, (8,))
    loss = focal_cross_entropy(logits, targets)
    assert loss.shape == ()


def test_smoothness():
    actions = torch.randn(4, 12, 6)
    loss = smoothness_loss(actions)
    assert loss.shape == ()


def test_compute_losses():
    B = 4
    batch = {
        "target_phase_next": torch.randint(0, 6, (B,)),
        "target_grip_token": torch.randint(0, 3, (B,)),
        "target_contact": torch.rand(B),
    }
    out = {
        "phase_logits": torch.randn(B, 6),
        "grip_logits": torch.randn(B, 3),
        "contact_logit": torch.randn(B, 1),
    }
    pred_noise = torch.randn(B, 12, 6)
    true_noise = torch.randn(B, 12, 6)

    losses = compute_losses(batch, out, pred_noise, true_noise)
    assert "total" in losses
    assert "arm_diff" in losses
    assert losses["total"].shape == ()
