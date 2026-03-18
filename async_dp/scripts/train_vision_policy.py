"""
Train vision-based diffusion policy.

Usage:
    uv run python scripts/train_vision_policy.py --data-dir episodes/20260310_123456
    uv run python scripts/train_vision_policy.py --data-dir episodes/pen_pick --epochs 200 --batch-size 64
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler
from src.utils.vision_dataset import VisionDiffusionDataset


def train_one_epoch(model, loader, optimizer, lr_scheduler, noise_scheduler, device, grad_clip):
    model.train()
    total_loss = 0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        img = batch['image'].to(device)       # (B, 3, 224, 224)
        qpos = batch['qpos'].to(device)       # (B, 7)
        action = batch['action'].to(device)   # (B, 16, 7)
        progress = batch['progress'].to(device)  # (B, 1)

        noise = torch.randn_like(action)
        timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

        noise_pred = model(noisy_action, timesteps, img, qpos, progress)
        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


@torch.no_grad()
def validate(model, loader, noise_scheduler, device):
    model.eval()
    total_loss = 0
    n = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        img = batch['image'].to(device)
        qpos = batch['qpos'].to(device)
        action = batch['action'].to(device)
        progress = batch['progress'].to(device)

        noise = torch.randn_like(action)
        timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

        noise_pred = model(noisy_action, timesteps, img, qpos, progress)
        loss = nn.functional.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        n += 1

    return total_loss / n


def main():
    parser = argparse.ArgumentParser(description="Train vision diffusion policy")
    parser.add_argument("--data-dir", required=True, help="Directory with episode HDF5 files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--obs-horizon", type=int, default=1, help="Observation horizon (1 or 2)")
    parser.add_argument("--temporal-stride", type=int, default=1, help="Action trajectory stride (3 for 50Hz→~15Hz)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    dataset = VisionDiffusionDataset(args.data_dir, pred_horizon=16, action_dim=7,
                                     obs_horizon=args.obs_horizon,
                                     temporal_stride=args.temporal_stride)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=use_cuda, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=use_cuda)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7, obs_horizon=args.obs_horizon).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params (trainable): {num_params:,}")

    noise_scheduler = get_scheduler('ddpm', num_train_timesteps=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.01 + 0.99 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "vision_policy"
        )
    os.makedirs(save_dir, exist_ok=True)

    # Save dataset stats (needed for inference)
    stats_path = os.path.join(save_dir, "dataset_stats.json")
    stats_serializable = {}
    for k, v in dataset.stats.items():
        stats_serializable[k] = {sk: sv.tolist() if hasattr(sv, 'tolist') else sv
                                  for sk, sv in v.items()}
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val = float('inf')
    patience = 0
    max_patience = 20

    # CSV log for monitoring
    log_path = os.path.join(save_dir, "train_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, 'a' if log_exists else 'w')
    if not log_exists:
        log_file.write("epoch,train_loss,val_loss,best_val,lr,time_s\n")
        log_file.flush()

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Log: {log_path}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler,
                                      noise_scheduler, device, grad_clip=1.0)
        val_loss = validate(model, val_loader, noise_scheduler, device)

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | {dt:.1f}s")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': {'obs_horizon': args.obs_horizon, 'action_dim': 7,
                           'qpos_dim': 7, 'pred_horizon': 16,
                           'temporal_stride': args.temporal_stride},
            }, os.path.join(save_dir, "best.pth"))
            print(f"  -> New best! ({val_loss:.4f})")
        else:
            patience += 1

        # Log to CSV
        log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{best_val:.6f},{lr:.2e},{dt:.1f}\n")
        log_file.flush()

        # Periodic save
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(save_dir, f"epoch_{epoch+1}.pth"))

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    log_file.close()
    print("=" * 60)
    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
