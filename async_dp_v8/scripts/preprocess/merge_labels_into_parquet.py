#!/usr/bin/env python3
"""Merge phase/contact labels from interim into processed parquet files.

Joins phase_labels.parquet and contact_events.parquet into each
episode parquet in data/processed/{train,val,test}/ on (episode_id, frame_idx).

Also derives grip_token from gripper velocity:
  0=OPEN (opening), 1=HOLD (stationary), 2=CLOSE (closing)
"""
import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRIP_VEL_THRESHOLD = 0.005  # rad/s — below this is HOLD


def derive_grip_token(df: pd.DataFrame) -> pd.Series:
    """Derive grip token from gripper velocity."""
    if "gripper_vel_rad_s" in df.columns:
        vel = df["gripper_vel_rad_s"]
    elif "gripper_1" in df.columns:
        vel = df["gripper_1"]
    else:
        return pd.Series(1, index=df.index, dtype=int)  # default HOLD

    token = pd.Series(1, index=df.index, dtype=int)  # HOLD
    token[vel > GRIP_VEL_THRESHOLD] = 0   # OPEN (positive vel = opening)
    token[vel < -GRIP_VEL_THRESHOLD] = 2  # CLOSE (negative vel = closing)
    return token


def merge_labels(data_root: str = "data"):
    data_root = Path(data_root)
    interim = data_root / "interim"

    # Load label sources
    phase_path = interim / "phase_labels.parquet"
    contact_path = interim / "contact_events.parquet"

    if not phase_path.exists():
        logger.error(f"Phase labels not found: {phase_path}")
        return
    if not contact_path.exists():
        logger.error(f"Contact labels not found: {contact_path}")
        return

    phase_df = pd.read_parquet(phase_path)
    contact_df = pd.read_parquet(contact_path)

    # phase_labels already contains contact_soft/contact_hard, but use
    # contact_events as authoritative source for contact columns
    label_df = phase_df[["episode_id", "frame_idx", "phase"]].merge(
        contact_df[["episode_id", "frame_idx", "contact_soft", "contact_hard"]],
        on=["episode_id", "frame_idx"],
        how="outer",
    )
    logger.info(f"Label table: {len(label_df)} rows, columns={label_df.columns.tolist()}")

    # Process each split
    total_files = 0
    for split in ["train", "val", "test"]:
        split_dir = data_root / "processed" / split
        if not split_dir.exists():
            logger.info(f"Skip missing split: {split}")
            continue

        parquet_files = sorted(split_dir.glob("*.parquet"))
        logger.info(f"[{split}] {len(parquet_files)} files")

        for pf in parquet_files:
            df = pd.read_parquet(pf)

            # Drop old label columns if they exist (re-merge cleanly)
            for col in ["phase", "contact_soft", "contact_hard", "grip_token"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Merge labels
            df = df.merge(
                label_df,
                on=["episode_id", "frame_idx"],
                how="left",
            )

            # Fill NaN labels with defaults
            df["phase"] = df["phase"].fillna(0).astype(int)
            df["contact_soft"] = df["contact_soft"].fillna(0.0)
            df["contact_hard"] = df["contact_hard"].fillna(0).astype(int)

            # Derive grip_token
            df["grip_token"] = derive_grip_token(df)

            # Save back
            df.to_parquet(pf, index=False)
            total_files += 1

    logger.info(f"Done. Updated {total_files} parquet files.")

    # Verify
    sample = sorted((data_root / "processed" / "train").glob("*.parquet"))[0]
    vdf = pd.read_parquet(sample)
    logger.info(f"Verification — {sample.name}:")
    logger.info(f"  Columns: {sorted(vdf.columns.tolist())}")
    logger.info(f"  phase distribution: {vdf['phase'].value_counts().to_dict()}")
    logger.info(f"  grip_token distribution: {vdf['grip_token'].value_counts().to_dict()}")
    logger.info(f"  contact_hard distribution: {vdf['contact_hard'].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    merge_labels(args.data_root)
