#!/usr/bin/env python3
"""Calibrate gripper min/max positions."""
import argparse
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    args = parser.parse_args()

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        print("Failed to connect")
        return

    print("Move gripper to FULLY CLOSED position, then press Enter")
    input()
    state = dxl.read_state()
    grip_min = int(state["present_position"][-1])
    print(f"  Closed position: {grip_min}")

    print("Move gripper to FULLY OPEN position, then press Enter")
    input()
    state = dxl.read_state()
    grip_max = int(state["present_position"][-1])
    print(f"  Open position: {grip_max}")

    print(f"\nGripper calibration:")
    print(f"  MIN (closed) = {grip_min}")
    print(f"  MAX (open)   = {grip_max}")

    dxl.disconnect()


if __name__ == "__main__":
    main()
