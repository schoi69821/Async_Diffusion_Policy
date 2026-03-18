"""
Train binary gripper classifier on existing episode data.
Labels: action gripper < threshold → closed (1), else open (0).

Usage:
    uv run python scripts/train_gripper_classifier.py --data-dir episodes/pen_fixed
    uv run python scripts/train_gripper_classifier.py --data-dir episodes/pen_fixed --obs-horizon 2 --temporal-stride 3
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gripper_classifier import GripperClassifier
from src.utils.vision_dataset import VisionDiffusionDataset

# Gripper threshold: below this = closed
GRIPPER_THRESHOLD_RAD = np.deg2rad(-55)  # -55° separates open(-35°) from closed(-71°)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--obs-horizon", type=int, default=1)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset (reuse vision dataset, we only need image + qpos + action for labels)
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

    # Count label distribution
    n_closed = 0
    n_total = 0
    for ep in dataset.episodes:
        grip = ep['action'][:, 6]  # gripper joint in action
        n_closed += np.sum(grip < GRIPPER_THRESHOLD_RAD)
        n_total += len(grip)
    ratio_closed = n_closed / n_total
    print(f"Label distribution: {ratio_closed:.1%} closed, {1-ratio_closed:.1%} open")
    print(f"Train: {train_size}, Val: {val_size}")

    # Positive weight for class imbalance
    pos_weight = torch.tensor([(1 - ratio_closed) / max(ratio_closed, 0.01)]).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    # Model
    model = GripperClassifier(qpos_dim=6, obs_horizon=args.obs_horizon).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classifier params: {num_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save directory
    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "gripper_classifier"
        )
    os.makedirs(args.save_dir, exist_ok=True)

    # Training
    best_val_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img = batch['image'].to(device)
            qpos = batch['qpos'].to(device)
            progress = batch['progress'].to(device)
            action = batch['action']  # (B, 16, 7)

            # Label: will gripper be closed at ANY step in the 16-step trajectory?
            # This predicts FUTURE intent, not current state
            mn = dataset.stats['action']['min'][6]
            mx = dataset.stats['action']['max'][6]
            grip_all = action[:, :, 6]  # (B, 16) normalized gripper across trajectory
            grip_raw = (grip_all + 1) / 2 * (mx - mn) + mn  # unnormalize
            labels = (grip_raw.min(dim=1).values < GRIPPER_THRESHOLD_RAD).float().unsqueeze(1).to(device)

            logits = model(img, qpos, progress)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.shape[0]

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                qpos = batch['qpos'].to(device)
                progress = batch['progress'].to(device)
                action = batch['action']

                action_grip = action[:, 0, 6]
                mn = dataset.stats['action']['min'][6]
                mx = dataset.stats['action']['max'][6]
                grip_raw = (action_grip + 1) / 2 * (mx - mn) + mn
                labels = (grip_raw < GRIPPER_THRESHOLD_RAD).float().unsqueeze(1).to(device)

                logits = model(img, qpos, progress)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:3d} | loss={train_loss/len(train_loader):.4f} "
              f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'obs_horizon': args.obs_horizon,
                    'threshold_rad': float(GRIPPER_THRESHOLD_RAD),
                },
            }, os.path.join(args.save_dir, "best.pth"))
            print(f"  -> New best! ({val_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
