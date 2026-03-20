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

            # Raw qpos/qvel may be 7D (6 arm + 1 gripper) or 6D (arm only)
            raw_qpos = np.array(f["qpos"][i]) if "qpos" in f else np.zeros(NUM_JOINTS + 1)
            raw_qvel = np.array(f["qvel"][i]) if "qvel" in f else np.zeros(NUM_JOINTS + 1)

            # Arm joints (first 6)
            qpos = raw_qpos[:NUM_JOINTS] if len(raw_qpos) >= NUM_JOINTS else np.zeros(NUM_JOINTS)
            qvel = raw_qvel[:NUM_JOINTS] if len(raw_qvel) >= NUM_JOINTS else np.zeros(NUM_JOINTS)
            for j in range(NUM_JOINTS):
                rec[f"qpos_{j}"] = float(qpos[j])
                rec[f"qvel_{j}"] = float(qvel[j])

            # EE pose
            ee = qpos_to_ee_pose(qpos)
            for j in range(7):
                rec[f"ee_pose_{j}"] = float(ee[j])

            # Gripper (index 6 if present, else check dedicated keys)
            if len(raw_qpos) > NUM_JOINTS:
                grip_pos = float(raw_qpos[NUM_JOINTS])
                grip_vel = float(raw_qvel[NUM_JOINTS]) if len(raw_qvel) > NUM_JOINTS else 0.0
            elif "gripper_pos" in f:
                gp = f["gripper_pos"][i]
                grip_pos = float(np.asarray(gp).flat[0])
                gv = f["gripper_vel"][i] if "gripper_vel" in f else 0.0
                grip_vel = float(np.asarray(gv).flat[0])
            else:
                grip_pos, grip_vel = 0.0, 0.0
            rec["gripper_0"] = grip_pos
            rec["gripper_1"] = grip_vel

            # Current and PWM
            cur = np.array(f["current"][i]) if "current" in f else np.zeros(NUM_MOTORS)
            pwm = np.array(f["pwm"][i]) if "pwm" in f else np.zeros(NUM_MOTORS)
            n_cur = min(len(cur), NUM_MOTORS)
            n_pwm = min(len(pwm), NUM_MOTORS)
            for j in range(NUM_MOTORS):
                rec[f"current_{j}"] = float(cur[j]) if j < n_cur else 0.0
                rec[f"pwm_{j}"] = float(pwm[j]) if j < n_pwm else 0.0

            # Action (next frame qpos as target, arm only)
            if i < n - 1:
                next_raw = np.array(f["qpos"][i + 1]) if "qpos" in f else np.zeros(NUM_JOINTS)
                next_qpos = next_raw[:NUM_JOINTS] if len(next_raw) >= NUM_JOINTS else np.zeros(NUM_JOINTS)
                for j in range(NUM_JOINTS):
                    rec[f"action_arm_{j}"] = float(next_qpos[j])
            else:
                for j in range(NUM_JOINTS):
                    rec[f"action_arm_{j}"] = rec[f"qpos_{j}"]

            records.append(rec)

    df = pd.DataFrame(records)

    # --- Derived columns for phase labeling ---
    # Gripper current: last current channel (gripper motor)
    # HDF5 current has 7 channels (matching qpos); gripper = current_6
    df["gripper_current"] = df["current_6"]
    df["gripper_vel"] = df["gripper_1"]
    df["gripper_pos_rad"] = df["gripper_0"]
    df["gripper_vel_rad_s"] = df["gripper_1"]

    # EE position in mm
    df["ee_z_mm"] = df["ee_pose_2"] * 1000.0

    # EE Z velocity (mm/s) via finite difference
    dt = df["timestamp"].diff().fillna(df["timestamp"].iloc[1] - df["timestamp"].iloc[0] if len(df) > 1 else 1.0 / 30.0)
    dt = dt.clip(lower=1e-6)
    df["ee_vz_mm_s"] = df["ee_z_mm"].diff().fillna(0.0) / dt

    # Target XY: use EE XY at the frame of minimum gripper position (grasp point)
    grasp_idx = df["gripper_0"].idxmin()
    target_x = df.loc[grasp_idx, "ee_pose_0"]
    target_y = df.loc[grasp_idx, "ee_pose_1"]
    df["target_xy_error_mm"] = np.sqrt(
        (df["ee_pose_0"] - target_x) ** 2 + (df["ee_pose_1"] - target_y) ** 2
    ) * 1000.0

    # Home distance: EE distance from frame 0 position
    home_x = df.iloc[0]["ee_pose_0"]
    home_y = df.iloc[0]["ee_pose_1"]
    home_z = df.iloc[0]["ee_pose_2"]
    df["home_dist_mm"] = np.sqrt(
        (df["ee_pose_0"] - home_x) ** 2
        + (df["ee_pose_1"] - home_y) ** 2
        + (df["ee_pose_2"] - home_z) ** 2
    ) * 1000.0

    out_path = Path(output_dir) / f"{episode_id}.parquet"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Converted {hdf5_path} -> {out_path} ({len(df)} frames)")
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
