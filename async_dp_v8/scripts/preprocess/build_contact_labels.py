#!/usr/bin/env python3
"""Build contact labels from gripper signals."""
import argparse
import pandas as pd
from pathlib import Path

from async_dp_v8.data.relabel_contact import ContactRelabeler, ContactLabelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--output", default="data/interim/contact_events.parquet")
    args = parser.parse_args()

    cfg = ContactLabelConfig()
    relabeler = ContactRelabeler(cfg)

    parquet_files = sorted(Path(args.data_dir).glob("*.parquet"))
    all_labels = []

    for f in parquet_files:
        df = pd.read_parquet(f)
        if "gripper_current" not in df.columns or "gripper_vel" not in df.columns:
            continue
        labeled = relabeler.label_episode(df)
        all_labels.append(labeled[["episode_id", "frame_idx", "contact_soft", "contact_hard"]])

    if all_labels:
        result = pd.concat(all_labels, ignore_index=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        print(f"Contact labels saved to {args.output}")


if __name__ == "__main__":
    main()
