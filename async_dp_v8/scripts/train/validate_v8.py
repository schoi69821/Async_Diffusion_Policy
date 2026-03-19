#!/usr/bin/env python3
"""Validate trained v8 model."""
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import logging

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.data.dataset_v8 import AsyncDPv8Dataset
from async_dp_v8.data.collate import v8_collate_fn
from async_dp_v8.train.engine import TrainingEngine, NoiseScheduler
from async_dp_v8.utils.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--index", default="data/interim/episodes_index_val.parquet")
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridPolicyV8().to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    val_index = pd.read_parquet(args.index)
    val_ds = AsyncDPv8Dataset(data_dir=args.data_dir, index_df=val_index)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=v8_collate_fn)

    scheduler = NoiseScheduler()
    optimizer = torch.optim.AdamW(model.parameters())
    engine = TrainingEngine(model, optimizer, scheduler, device=device)
    metrics = engine.validate(val_loader)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
