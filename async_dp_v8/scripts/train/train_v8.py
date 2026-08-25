#!/usr/bin/env python3
"""Train HybridPolicyV8."""
import argparse
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import logging

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.data.dataset_v8 import AsyncDPv8Dataset
from async_dp_v8.data.collate import v8_collate_fn
from async_dp_v8.data.samplers import PhaseBalancedSampler
from async_dp_v8.train.engine import TrainingEngine, NoiseScheduler
from async_dp_v8.train.hooks import TrainingHooks
from async_dp_v8.utils.checkpointing import save_checkpoint
from async_dp_v8.utils.config import load_config, get_loss_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_warmup_schedule(warmup_epochs, total_epochs, min_lr_ratio):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--index", default="data/interim/episodes_index_train.parquet")
    parser.add_argument("--val-index", default="data/interim/episodes_index_val.parquet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stats", default="data/interim/stats.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/hybrid_policy_v8")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_dir)
    loss_weights = get_loss_weights(config)
    logger.info(f"Loss weights: {loss_weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Model
    model = HybridPolicyV8(pred_horizon=12, action_dim=6).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_scheduler = NoiseScheduler(num_steps=100)

    # LR scheduler: cosine annealing with warmup
    min_lr_ratio = args.min_lr / args.lr
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=cosine_warmup_schedule(args.warmup_epochs, args.epochs, min_lr_ratio),
    )
    logger.info(f"LR schedule: cosine warmup={args.warmup_epochs}, epochs={args.epochs}, "
                f"lr={args.lr} -> {args.min_lr}")

    # Data (with proprioceptive augmentation)
    aug_cfg = config.get("augmentation", {})
    action_noise_cfg = aug_cfg.get("action_noise", {})
    qpos_noise = action_noise_cfg.get("qpos_std", 0.0) if action_noise_cfg.get("enabled", False) else 0.0
    qvel_noise = action_noise_cfg.get("qvel_std", 0.0) if action_noise_cfg.get("enabled", False) else 0.0
    logger.info(f"Proprio augmentation: qpos_std={qpos_noise}, qvel_std={qvel_noise}")

    train_index = pd.read_parquet(args.index)
    train_ds = AsyncDPv8Dataset(
        data_dir=args.data_dir, index_df=train_index, stats_path=args.stats,
        qpos_noise_std=qpos_noise, qvel_noise_std=qvel_noise,
    )
    sampler = PhaseBalancedSampler(train_index)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=v8_collate_fn,
    )

    val_loader = None
    if Path(args.val_index).exists():
        val_index = pd.read_parquet(args.val_index)
        val_ds = AsyncDPv8Dataset(data_dir=args.data_dir, index_df=val_index, stats_path=args.stats)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=v8_collate_fn,
        )

    # Training
    engine = TrainingEngine(
        model, optimizer, noise_scheduler,
        device=device, ema_decay=0.99, loss_kwargs=loss_weights,
    )
    hooks = TrainingHooks()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = engine.train_one_epoch(train_loader)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics = None
        if val_loader and (epoch + 1) % args.val_every == 0:
            val_metrics = engine.validate(val_loader)
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    f"{args.checkpoint_dir}/best.pt",
                    ema_model=engine.ema,
                )
                logger.info(f"  New best val_loss={best_val_loss:.4f}")

        hooks.on_epoch_end(epoch, train_metrics, val_metrics)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} lr={current_lr:.2e} "
                        f"train_total={train_metrics['total']:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                f"{args.checkpoint_dir}/epoch_{epoch+1:04d}.pt",
            )

    hooks.close()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
