#!/usr/bin/env python3
"""Compute dataset statistics for normalization."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from async_dp_v8.constants import NUM_JOINTS, NUM_MOTORS
from async_dp_v8.utils.normalization import Normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--output", default="data/interim/stats.json")
    args = parser.parse_args()

    parquet_files = sorted(Path(args.data_dir).glob("*.parquet"))
    print(f"Computing stats from {len(parquet_files)} files")

    all_dfs = [pd.read_parquet(f) for f in parquet_files]
    if not all_dfs:
        print("No data found")
        return
    df = pd.concat(all_dfs, ignore_index=True)

    field_groups = {
        "qpos": [f"qpos_{i}" for i in range(NUM_JOINTS)],
        "qvel": [f"qvel_{i}" for i in range(NUM_JOINTS)],
        "ee_pose": [f"ee_pose_{i}" for i in range(7)],
        "gripper": ["gripper_0", "gripper_1"],
        "current": [f"current_{i}" for i in range(NUM_MOTORS)],
        "pwm": [f"pwm_{i}" for i in range(NUM_MOTORS)],
        "action_arm": [f"action_arm_{i}" for i in range(NUM_JOINTS)],
    }

    data_dict = {}
    for key, cols in field_groups.items():
        available = [c for c in cols if c in df.columns]
        if available:
            data_dict[key] = df[available].to_numpy().astype(np.float32)

    normalizer = Normalizer.fit(data_dict)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    normalizer.to_json(args.output)
    print(f"Stats saved to {args.output}")


if __name__ == "__main__":
    main()
