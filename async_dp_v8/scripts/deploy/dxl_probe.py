#!/usr/bin/env python3
"""Probe Dynamixel motors for diagnostics."""
import argparse
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    args = parser.parse_args()

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        print("Connection failed")
        return

    state = dxl.read_state()
    voltage = dxl.read_voltage()

    print(f"Port: {args.port}")
    print(f"Motors: {dxl.ids}")
    print(f"Voltage: {voltage:.1f}V")
    print(f"Positions: {state['present_position']}")
    print(f"Velocities: {state['present_velocity']}")
    print(f"Currents: {state['present_current']}")
    print(f"PWMs: {state['present_pwm']}")

    dxl.disconnect()


if __name__ == "__main__":
    main()
