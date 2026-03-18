"""
Tests for Diffusion inference logic
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
from config.settings import Config


def test_diffusion_inference_shape():
    """Test that diffusion inference produces correct output shape"""
    device = torch.device('cpu')
    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)
    model.eval()

    scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=8)

    # Simulate inference
    obs = torch.randn(1, Config.OBS_DIM).to(device)
    noisy_action = torch.randn(1, Config.PRED_HORIZON, Config.ACTION_DIM).to(device)

    scheduler.set_timesteps(8)
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = model(noisy_action, t.unsqueeze(0).to(device), obs)
            noise_pred = noise_pred.transpose(1, 2)
            noisy_action_t = noisy_action.transpose(1, 2)
            noisy_action_t = scheduler.step(noise_pred, t, noisy_action_t).prev_sample
            noisy_action = noisy_action_t.transpose(1, 2)

    assert noisy_action.shape == (1, Config.PRED_HORIZON, Config.ACTION_DIM)
    print("[Test] Diffusion inference shape: PASSED")


def test_scheduler_timesteps():
    """Test DDIM scheduler timestep generation"""
    scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=16)
    scheduler.set_timesteps(16)

    assert len(scheduler.timesteps) == 16
    assert scheduler.timesteps[0] > scheduler.timesteps[-1]  # Descending order
    print("[Test] Scheduler timesteps: PASSED")


def test_action_clipping():
    """Test that actions are properly clipped to safety limits"""
    action = np.array([5.0, -5.0, 0.5, -0.5])
    clipped = np.clip(action, -Config.EMERGENCY_STOP_LIMIT, Config.EMERGENCY_STOP_LIMIT)

    assert np.all(clipped <= Config.EMERGENCY_STOP_LIMIT)
    assert np.all(clipped >= -Config.EMERGENCY_STOP_LIMIT)
    print("[Test] Action clipping: PASSED")


if __name__ == "__main__":
    test_diffusion_inference_shape()
    test_scheduler_timesteps()
    test_action_clipping()
    print("\nAll inference tests passed!")
