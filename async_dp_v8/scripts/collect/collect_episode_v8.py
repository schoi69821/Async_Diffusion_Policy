#!/usr/bin/env python3
"""Collect demonstration episodes with v8 schema (qpos, qvel, current, pwm)."""
import argparse
import time
import h5py
import numpy as np
from pathlib import Path

from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE


def collect_episode(robot: RobotInterface, duration_s: float, hz: float = 30.0) -> dict:
    dt = 1.0 / hz
    frames = []
    print(f"Recording for {duration_s}s at {hz}Hz...")

    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        ts = time.monotonic()
        obs = robot.get_observation()
        obs["timestamp"] = time.monotonic() - t0
        frames.append(obs)

        elapsed = time.monotonic() - ts
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"Collected {len(frames)} frames")
    return {"frames": frames}


def save_hdf5(data: dict, path: str):
    with h5py.File(path, "w") as f:
        frames = data["frames"]
        n = len(frames)
        for key in frames[0]:
            if key == "timestamp":
                f.create_dataset(key, data=np.array([fr[key] for fr in frames]))
            else:
                vals = [fr[key] for fr in frames]
                f.create_dataset(key, data=np.stack(vals))
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--output-dir", default="data/raw/pen_fixed_hdf5")
    parser.add_argument("--episode-name", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        print("Failed to connect, using dummy mode")

    robot = RobotInterface(dxl)

    name = args.episode_name or f"episode_{int(time.time())}"
    data = collect_episode(robot, args.duration, args.hz)
    save_hdf5(data, str(out_dir / f"{name}.hdf5"))

    dxl.disconnect()


if __name__ == "__main__":
    main()
