#!/usr/bin/env python3
"""Build phase labels for all episodes."""
import argparse
import pandas as pd
from pathlib import Path

from async_dp_v8.data.relabel_phase import PhaseRelabeler, PhaseLabelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--output", default="data/interim/phase_labels.parquet")
    args = parser.parse_args()

    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)

    parquet_files = sorted(Path(args.data_dir).glob("*.parquet"))
    print(f"Labeling {len(parquet_files)} episodes")

    all_labels = []
    for f in parquet_files:
        df = pd.read_parquet(f)

        # Check required columns exist
        required = ["gripper_current", "gripper_vel", "target_xy_error_mm",
                     "ee_z_mm", "ee_vz_mm_s", "gripper_pos_rad",
                     "gripper_vel_rad_s", "home_dist_mm"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  Skipping {f.name}: missing columns {missing}")
            continue

        labeled = relabeler.label_episode(df)
        all_labels.append(labeled[["episode_id", "frame_idx", "phase", "contact_soft", "contact_hard"]])

    if all_labels:
        result = pd.concat(all_labels, ignore_index=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        print(f"Phase labels saved to {args.output} ({len(result)} frames)")
    else:
        print("No episodes labeled")


if __name__ == "__main__":
    main()
