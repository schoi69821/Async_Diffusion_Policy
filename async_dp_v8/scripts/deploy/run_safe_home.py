#!/usr/bin/env python3
"""Safely move robot to home position."""
import argparse
import time
import numpy as np

from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--speed", type=float, default=0.5, help="0-1 speed factor")
    args = parser.parse_args()

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        print("Connection failed")
        return

    robot = RobotInterface(dxl)
    obs = robot.get_observation()
    current = obs["qpos"]
    home = robot.home_qpos

    print(f"Current: {current}")
    print(f"Home:    {home}")
    print(f"Delta:   {home - current}")

    steps = int(50 / args.speed)
    for i in range(steps):
        t = (i + 1) / steps
        target = current + t * (home - current)
        for j, motor_id in enumerate(dxl.ids[:6]):
            dxl.goal_position_rad(motor_id, target[j])
        time.sleep(0.02)

    robot.open_gripper()
    print("At home position")
    dxl.disconnect()


if __name__ == "__main__":
    main()
