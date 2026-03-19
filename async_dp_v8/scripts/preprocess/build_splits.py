#!/usr/bin/env python3
"""Build train/val/test splits by episode."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--output-dir", default="data/interim")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet_files = sorted(Path(args.data_dir).glob("*.parquet"))
    episodes = [f.stem for f in parquet_files]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(episodes)

    n = len(episodes)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": episodes[:n_train],
        "val": episodes[n_train:n_train + n_val],
        "test": episodes[n_train + n_val:],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build index dataframes
    for split_name, ep_list in splits.items():
        rows = []
        for ep_id in ep_list:
            ep_path = Path(args.data_dir) / f"{ep_id}.parquet"
            if ep_path.exists():
                df = pd.read_parquet(ep_path)
                for _, row in df.iterrows():
                    rows.append({
                        "episode_id": ep_id,
                        "frame_idx": row.get("frame_idx", 0),
                        "phase": row.get("phase", 0),
                    })

        split_df = pd.DataFrame(rows)
        out_path = out_dir / f"episodes_index_{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(f"{split_name}: {len(ep_list)} episodes, {len(split_df)} frames -> {out_path}")


if __name__ == "__main__":
    main()
