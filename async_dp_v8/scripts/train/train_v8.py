#!/usr/bin/env python3
"""Train HybridPolicyV8."""
import argparse
import torch
from torch.optim import AdamW
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--index", default="data/interim/episodes_index_train.parquet")
    parser.add_argument("--val-index", default="data/interim/episodes_index_val.parquet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/hybrid_policy_v8")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Model
    model = HybridPolicyV8(pred_horizon=12, action_dim=6).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = NoiseScheduler(num_steps=100)

    # Data
    train_index = pd.read_parquet(args.index)
    train_ds = AsyncDPv8Dataset(data_dir=args.data_dir, index_df=train_index)
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
        val_ds = AsyncDPv8Dataset(data_dir=args.data_dir, index_df=val_index)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=v8_collate_fn,
        )

    # Training
    engine = TrainingEngine(model, optimizer, scheduler, device=device)
    hooks = TrainingHooks()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = engine.train_one_epoch(train_loader)

        val_metrics = None
        if val_loader and (epoch + 1) % args.val_every == 0:
            val_metrics = engine.validate(val_loader)
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    f"{args.checkpoint_dir}/best.pt",
                    ema_model=None,
                )

        hooks.on_epoch_end(epoch, train_metrics, val_metrics)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                f"{args.checkpoint_dir}/epoch_{epoch+1:04d}.pt",
            )

    hooks.close()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
