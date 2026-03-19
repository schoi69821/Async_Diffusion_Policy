#!/usr/bin/env python3
"""Convert raw HDF5 episodes to v8 Parquet format."""
import argparse
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

from async_dp_v8.control.kinematics import qpos_to_ee_pose
from async_dp_v8.constants import NUM_JOINTS, NUM_MOTORS


def convert_episode(hdf5_path: str, output_dir: str) -> str:
    episode_id = Path(hdf5_path).stem

    with h5py.File(hdf5_path, "r") as f:
        n = len(f["timestamp"])
        records = []

        for i in range(n):
            rec = {
                "episode_id": episode_id,
                "frame_idx": i,
                "timestamp": float(f["timestamp"][i]),
            }

            # Arm joints
            qpos = f["qpos"][i] if "qpos" in f else np.zeros(NUM_JOINTS)
            qvel = f["qvel"][i] if "qvel" in f else np.zeros(NUM_JOINTS)
            for j in range(NUM_JOINTS):
                rec[f"qpos_{j}"] = float(qpos[j]) if j < len(qpos) else 0.0
                rec[f"qvel_{j}"] = float(qvel[j]) if j < len(qvel) else 0.0

            # EE pose
            ee = qpos_to_ee_pose(qpos[:NUM_JOINTS] if len(qpos) >= NUM_JOINTS else np.zeros(NUM_JOINTS))
            for j in range(7):
                rec[f"ee_pose_{j}"] = float(ee[j])

            # Gripper
            grip_pos = float(f["gripper_pos"][i]) if "gripper_pos" in f else 0.0
            grip_vel = float(f["gripper_vel"][i]) if "gripper_vel" in f else 0.0
            rec["gripper_0"] = grip_pos
            rec["gripper_1"] = grip_vel

            # Current and PWM
            cur = f["current"][i] if "current" in f else np.zeros(NUM_MOTORS)
            pwm = f["pwm"][i] if "pwm" in f else np.zeros(NUM_MOTORS)
            for j in range(NUM_MOTORS):
                rec[f"current_{j}"] = float(cur[j]) if j < len(cur) else 0.0
                rec[f"pwm_{j}"] = float(pwm[j]) if j < len(pwm) else 0.0

            # Action (next frame qpos as target)
            if i < n - 1:
                next_qpos = f["qpos"][i + 1] if "qpos" in f else np.zeros(NUM_JOINTS)
                for j in range(NUM_JOINTS):
                    rec[f"action_arm_{j}"] = float(next_qpos[j]) if j < len(next_qpos) else 0.0
            else:
                for j in range(NUM_JOINTS):
                    rec[f"action_arm_{j}"] = rec[f"qpos_{j}"]

            records.append(rec)

    df = pd.DataFrame(records)
    out_path = Path(output_dir) / f"{episode_id}.parquet"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Converted {hdf5_path} -> {out_path} ({n} frames)")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/pen_fixed_hdf5")
    parser.add_argument("--output-dir", default="data/processed/train")
    args = parser.parse_args()

    hdf5_files = sorted(Path(args.input_dir).glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")

    for f in hdf5_files:
        convert_episode(str(f), args.output_dir)


if __name__ == "__main__":
    main()
