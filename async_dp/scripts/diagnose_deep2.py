"""
Quick check: scheduler fix + overfit test with more steps.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler
from src.utils.vision_dataset import VisionDiffusionDataset


def test_scheduler():
    print("=== Scheduler roundtrip (clip_sample=False) ===")
    scheduler = get_scheduler('ddpm', num_train_timesteps=100)
    ddim = get_scheduler('ddim', num_train_timesteps=100)

    action = torch.randn(1, 7, 16) * 0.5
    noise = torch.randn_like(action)
    noisy = scheduler.add_noise(action, noise, torch.tensor([99]))

    ddim.set_timesteps(50)
    sample = noisy.clone()
    for t in ddim.timesteps:
        sample = ddim.step(noise, t, sample).prev_sample

    err = torch.abs(sample - action).mean().item()
    print(f"  Error: {err:.6f} (should be near 0)")


def test_overfit(data_dir, device, steps=1000):
    print(f"\n=== Overfit single sample ({steps} steps) ===")
    dataset = VisionDiffusionDataset(data_dir, pred_horizon=16, action_dim=7)
    sample = dataset[len(dataset)//2]

    img = sample['image'].unsqueeze(0).to(device)
    qpos = sample['qpos'].unsqueeze(0).to(device)
    action = sample['action'].unsqueeze(0).to(device)

    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    noise_scheduler = get_scheduler('ddpm', num_train_timesteps=100)

    for step in range(steps):
        noise = torch.randn_like(action)
        t = torch.randint(0, 100, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(action, noise, t)
        pred = model(noisy, t, img, qpos)
        loss = nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 200 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}")

    # Test inference
    model.eval()
    scheduler = get_scheduler('ddim', num_train_timesteps=100)

    for n_steps in [50, 100]:
        torch.manual_seed(42)
        with torch.no_grad():
            noisy_action = torch.randn(1, 16, 7, device=device)
            scheduler.set_timesteps(n_steps)
            for t in scheduler.timesteps:
                noise_pred = model(noisy_action, t.unsqueeze(0).to(device), img, qpos)
                noise_pred = noise_pred.transpose(1, 2)
                noisy_t = noisy_action.transpose(1, 2)
                noisy_t = scheduler.step(noise_pred, t, noisy_t).prev_sample
                noisy_action = noisy_t.transpose(1, 2)

        pred_np = noisy_action.squeeze(0).cpu().numpy()
        gt_np = action.squeeze(0).cpu().numpy()

        pred_raw = dataset.unnormalize_action(pred_np)
        gt_raw = dataset.unnormalize_action(gt_np)

        err = np.abs(np.rad2deg(pred_raw[0] - gt_raw[0]))
        print(f"\n  Inference ({n_steps} steps):")
        print(f"    GT   (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in gt_raw[0])}]")
        print(f"    Pred (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in pred_raw[0])}]")
        print(f"    Error: mean={np.mean(err):.1f}  max={np.max(err):.1f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_scheduler()
    test_overfit("episodes/20260310_141707", device, steps=1000)
