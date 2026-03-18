"""Quick overfit test for v2 architecture."""
import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler
from src.utils.vision_dataset import VisionDiffusionDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = VisionDiffusionDataset("episodes/20260310_141707", pred_horizon=16, action_dim=7)
    sample = dataset[len(dataset)//2]
    img = sample['image'].unsqueeze(0).to(device)
    qpos = sample['qpos'].unsqueeze(0).to(device)
    action = sample['action'].unsqueeze(0).to(device)

    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    noise_scheduler = get_scheduler('ddpm', num_train_timesteps=100)

    print("\n=== Overfit single sample (2000 steps) ===")
    for step in range(2000):
        noise = torch.randn_like(action)
        t = torch.randint(0, 100, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(action, noise, t)
        pred = model(noisy, t, img, qpos)
        loss = nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step+1) % 200 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}")

    # Test inference
    model.eval()
    gt_raw = dataset.unnormalize_action(action.squeeze(0).cpu().numpy())

    for n_steps in [16, 50, 100]:
        scheduler = get_scheduler('ddim', num_train_timesteps=100)
        torch.manual_seed(42)
        with torch.no_grad():
            pred_norm = model.get_action(
                img.squeeze(0), qpos.squeeze(0), scheduler,
                num_inference_steps=n_steps, device=device
            )
        pred_raw = dataset.unnormalize_action(pred_norm)

        err = np.abs(np.rad2deg(pred_raw[0] - gt_raw[0]))
        print(f"\n  Inference ({n_steps} steps):")
        print(f"    GT   (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in gt_raw[0])}]")
        print(f"    Pred (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in pred_raw[0])}]")
        print(f"    Error: mean={np.mean(err):.1f}  max={np.max(err):.1f}")


if __name__ == "__main__":
    main()
