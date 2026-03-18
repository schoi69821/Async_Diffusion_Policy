"""
Deep diagnosis: test if diffusion inference pipeline actually works.
1. Test scheduler: add noise → denoise with KNOWN noise → should reconstruct
2. Test model with more inference steps (DDPM 100 vs DDIM 16)
3. Test model overfitting on a single sample

Usage:
    uv run python scripts/diagnose_deep.py --checkpoint checkpoints/vision_policy/best.pth --data-dir episodes/20260310_141707
"""
import numpy as np
import torch
import torch.nn as nn
import json
import h5py
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler
from src.utils.vision_dataset import VisionDiffusionDataset


def test_scheduler_roundtrip():
    """Test: add noise then denoise with perfect noise prediction → should reconstruct."""
    print("\n" + "="*60)
    print("  TEST 1: Scheduler roundtrip (no model)")
    print("="*60)

    scheduler = get_scheduler('ddpm', num_train_timesteps=100)
    ddim_scheduler = get_scheduler('ddim', num_train_timesteps=100)

    # Create a known action trajectory
    action = torch.randn(1, 7, 16) * 0.5  # small values in [-1, 1]

    # Add noise at timestep 99 (full noise)
    noise = torch.randn_like(action)
    t = torch.tensor([99])
    noisy = scheduler.add_noise(action, noise, t)

    # Try to reconstruct with DDIM using the KNOWN noise
    ddim_scheduler.set_timesteps(16)
    sample = noisy.clone()
    for t_step in ddim_scheduler.timesteps:
        # Perfect noise prediction = the actual noise
        # But scheduler.step expects noise prediction at CURRENT timestep
        # This is a simplified test
        result = ddim_scheduler.step(noise, t_step, sample)
        sample = result.prev_sample

    error = torch.abs(sample - action).mean().item()
    print(f"  Reconstruction error (perfect noise): {error:.6f}")
    if error < 0.1:
        print(f"  → OK: Scheduler roundtrip works")
    else:
        print(f"  → WARNING: Scheduler roundtrip has high error")
    return error


def test_inference_steps(model, img, qpos_norm, stats, device):
    """Test: compare DDIM-16 vs DDIM-50 vs DDIM-100 inference steps."""
    print("\n" + "="*60)
    print("  TEST 2: Inference steps comparison")
    print("="*60)

    for n_steps in [10, 16, 50, 100]:
        scheduler = get_scheduler('ddim', num_train_timesteps=100)
        torch.manual_seed(42)
        action_norm = model.get_action(img, qpos_norm, scheduler,
                                        num_inference_steps=n_steps, device=device)

        # Check action statistics
        mn = action_norm.min()
        mx = action_norm.max()
        std = action_norm.std()

        # Unnormalize
        action_mn = np.array(stats['action']['min'])
        action_mx = np.array(stats['action']['max'])
        action = (action_norm + 1) / 2 * (action_mx - action_mn) + action_mn

        print(f"  Steps={n_steps:3d} | norm range: [{mn:.2f}, {mx:.2f}] std={std:.2f} | "
              f"action[0] (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in action[0])}]")


def test_model_dimensions(model, device):
    """Test: verify model forward pass dimensions."""
    print("\n" + "="*60)
    print("  TEST 3: Model dimensions")
    print("="*60)

    B = 2
    img = torch.randn(B, 3, 224, 224, device=device)
    qpos = torch.randn(B, 7, device=device)
    noisy_action = torch.randn(B, 16, 7, device=device)
    timestep = torch.tensor([50, 50], device=device)

    # Test time_proj output
    t_proj = model.noise_pred_net.time_proj(timestep)
    print(f"  time_proj output shape: {t_proj.shape}")

    # Test time_emb input
    time_emb_input = model.time_emb[0].in_features
    print(f"  time_emb input features: {time_emb_input}")
    print(f"  MATCH: {t_proj.shape[-1] == time_emb_input}")

    # Full forward
    try:
        output = model(noisy_action, timestep, img, qpos)
        print(f"  Forward output shape: {output.shape} (expected: [{B}, 16, 7])")
        print(f"  → OK: Forward pass works")
    except Exception as e:
        print(f"  → ERROR: {e}")


def test_overfit_single(data_dir, device):
    """Test: can the model overfit a single training sample?"""
    print("\n" + "="*60)
    print("  TEST 4: Overfit single sample (fresh model, 200 steps)")
    print("="*60)

    dataset = VisionDiffusionDataset(data_dir, pred_horizon=16, action_dim=7)

    # Get one sample
    sample = dataset[len(dataset)//2]
    img = sample['image'].unsqueeze(0).to(device)
    qpos = sample['qpos'].unsqueeze(0).to(device)
    action = sample['action'].unsqueeze(0).to(device)

    # Fresh model
    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noise_scheduler = get_scheduler('ddpm', num_train_timesteps=100)

    losses = []
    for step in range(200):
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, 100, (1,), device=device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

        noise_pred = model(noisy_action, timesteps, img, qpos)
        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Loss: {losses[0]:.4f} → {losses[49]:.4f} → {losses[99]:.4f} → {losses[-1]:.4f}")

    # Test inference on the overfit sample
    model.eval()
    scheduler = get_scheduler('ddim', num_train_timesteps=100)

    # Unnormalize ground truth
    action_np = action.squeeze(0).cpu().numpy()
    gt_raw = dataset.unnormalize_action(action_np)

    # Predict
    img_np = sample['image'].numpy().transpose(1, 2, 0)  # CHW → HWC
    img_np = ((img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
    qpos_np = sample['qpos'].numpy()

    # Use normalized input directly
    torch.manual_seed(42)
    with torch.no_grad():
        noisy_action = torch.randn(1, 16, 7, device=device)
        scheduler.set_timesteps(50)
        for t in scheduler.timesteps:
            noise_pred = model(noisy_action, t.unsqueeze(0).to(device), img, qpos)
            noise_pred = noise_pred.transpose(1, 2)
            noisy_t = noisy_action.transpose(1, 2)
            noisy_t = scheduler.step(noise_pred, t, noisy_t).prev_sample
            noisy_action = noisy_t.transpose(1, 2)

    pred_np = noisy_action.squeeze(0).cpu().numpy()
    pred_raw = dataset.unnormalize_action(pred_np)

    err_deg = np.abs(np.rad2deg(pred_raw[0] - gt_raw[0]))
    mean_err = np.mean(err_deg)

    print(f"  GT action[0]  (deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in gt_raw[0])}]")
    print(f"  Pred action[0](deg): [{', '.join(f'{np.rad2deg(v):+.0f}' for v in pred_raw[0])}]")
    print(f"  Error (deg): mean={mean_err:.1f}")

    if mean_err < 10:
        print(f"  → OK: Model CAN overfit — architecture works, need more data/training")
    elif mean_err < 30:
        print(f"  → PARTIAL: Model partially overfits — might need more overfit steps or architecture tweaks")
    else:
        print(f"  → FAIL: Model CANNOT overfit single sample — architecture or inference bug")

    return mean_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load trained model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load stats
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "dataset_stats.json")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    for k in stats:
        for sk in stats[k]:
            stats[k][sk] = np.array(stats[k][sk], dtype=np.float32)

    # Load one sample for testing
    episode_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.hdf5')])
    ep_path = os.path.join(args.data_dir, episode_files[0])
    with h5py.File(ep_path, 'r') as f:
        img = f['observations/images'][100]
        qpos = f['observations/qpos'][100].astype(np.float32)

    qpos_mn = stats['qpos']['min']
    qpos_mx = stats['qpos']['max']
    rng = qpos_mx - qpos_mn
    rng[rng < 1e-6] = 1.0
    qpos_norm = (qpos - qpos_mn) / rng * 2 - 1

    # Run tests
    test_scheduler_roundtrip()
    test_model_dimensions(model, device)
    test_inference_steps(model, img, qpos_norm, stats, device)
    test_overfit_single(args.data_dir, device)

    print("\n" + "="*60)
    print("  DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
