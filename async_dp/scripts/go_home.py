"""
Move follower arm safely to home position via mid position.
Route: Current → Mid (arm raised) → Home (table rest)
This prevents collisions when returning from arbitrary task positions.

Usage:
    uv run python scripts/go_home.py
    uv run python scripts/go_home.py --mid-only   # Stop at mid position
"""
import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD
)

PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_JOINTS = 7
POS_TO_RAD = 0.001534

# From config/settings.py
HOME = np.array([+0.0199, +1.6682, -1.6176, -0.0430, -0.1319, -0.0568, -0.4065], dtype=np.float32)
MID  = np.array([-0.1227, +0.7148, -0.2094, +0.0828, +0.5891, -0.0261, -1.1766], dtype=np.float32)


class SimpleArm:
    def __init__(self, port_name="/dev/ttyDXL_puppet_right"):
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)
        self.port.openPort()
        self.port.setBaudRate(1000000)

        self._sr = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        self._sw = GroupSyncWrite(self.port, self.pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for mid in ALL_MOTOR_IDS:
            self._sr.addParam(mid)

    def read_joints(self):
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self._sr.txRxPacket() != COMM_SUCCESS:
            return joints
        for j, mids in enumerate(JOINT_MAP):
            pos = []
            for mid in mids:
                if self._sr.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    raw = self._sr.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    pos.append((raw - 2048) * POS_TO_RAD)
            if pos:
                joints[j] = np.mean(pos)
        return joints

    def write_joints(self, joints):
        self._sw.clearParam()
        for j, mids in enumerate(JOINT_MAP):
            raw = int(joints[j] / POS_TO_RAD + 2048)
            raw = max(0, min(4095, raw))
            param = [DXL_LOBYTE(DXL_LOWORD(raw)), DXL_HIBYTE(DXL_LOWORD(raw)),
                     DXL_LOBYTE(DXL_HIWORD(raw)), DXL_HIBYTE(DXL_HIWORD(raw))]
            for mid in mids:
                self._sw.addParam(mid, param)
        self._sw.txPacket()

    def set_torque(self, enable):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def set_profile(self, velocity=60, acceleration=30):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_VELOCITY, velocity)
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    def move_to(self, target, label="target", wait=3.0):
        print(f"  Moving to {label}...")
        self.write_joints(target)
        time.sleep(wait)
        actual = self.read_joints()
        error = np.max(np.abs(np.rad2deg(actual - target)))
        print(f"  Arrived. (max error: {error:.1f} deg)")

    def close(self):
        self.set_torque(False)
        self.port.closePort()


def safe_go_home(arm, mid_only=False):
    """Move arm safely: Current → Mid → Home."""
    current = arm.read_joints()
    print(f"Current (deg): [{', '.join(f'{np.rad2deg(v):+.1f}' for v in current)}]")

    # Step 1: Go to mid position (arm raised, safe)
    arm.move_to(MID, label="MID (safe waypoint)", wait=3.0)

    # Step 2: Go to home (table rest)
    if not mid_only:
        arm.move_to(HOME, label="HOME (rest)", wait=3.0)

    final = arm.read_joints()
    dest = "MID" if mid_only else "HOME"
    print(f"Final   (deg): [{', '.join(f'{np.rad2deg(v):+.1f}' for v in final)}]")
    print(f"Done. At {dest} position.")


def main():
    parser = argparse.ArgumentParser(description="Safe return to home via mid position")
    parser.add_argument("--mid-only", action="store_true", help="Stop at mid position")
    args = parser.parse_args()

    arm = SimpleArm()
    arm.set_torque(False)
    arm.set_profile(velocity=60, acceleration=30)

    current = arm.read_joints()
    arm.write_joints(current)
    time.sleep(0.05)
    arm.set_torque(True)
    time.sleep(0.3)

    try:
        safe_go_home(arm, mid_only=args.mid_only)
    finally:
        arm.set_torque(False)
        arm.port.closePort()
        print("Torque OFF.")


if __name__ == "__main__":
    main()
