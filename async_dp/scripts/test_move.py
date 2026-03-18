"""
Test: Move follower (execution) arm joint 0 (waist) by ~10 degrees, then return to original position.
Usage: uv run python scripts/test_move.py
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD
)

# Dynamixel Protocol 2.0
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
POS_TO_RAD = 0.001534  # 4096 units = 2*pi rad

MOVE_DEG = 10.0
MOVE_RAD = np.deg2rad(MOVE_DEG)
MOVE_JOINT = 0  # Waist


def main():
    port_name = "/dev/ttyDXL_puppet_right"
    port = PortHandler(port_name)
    pkt = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        print(f"Failed to open {port_name}")
        return
    if not port.setBaudRate(1000000):
        print("Failed to set baudrate")
        return

    print(f"Connected to {port_name} (Follower / Execution arm)")

    # Sync read setup
    sync_read = GroupSyncRead(port, pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for mid in ALL_MOTOR_IDS:
        sync_read.addParam(mid)

    # Sync write setup
    sync_write = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def read_joints():
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        result = sync_read.txRxPacket()
        if result != COMM_SUCCESS:
            print(f"Sync read failed: {result}")
            return joints
        for j, motor_ids in enumerate(JOINT_MAP):
            positions = []
            for mid in motor_ids:
                if sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    raw = sync_read.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    positions.append((raw - 2048) * POS_TO_RAD)
            if positions:
                joints[j] = np.mean(positions)
        return joints

    def write_joints(joints):
        sync_write.clearParam()
        for j, motor_ids in enumerate(JOINT_MAP):
            raw = int(joints[j] / POS_TO_RAD + 2048)
            raw = max(0, min(4095, raw))
            param = [
                DXL_LOBYTE(DXL_LOWORD(raw)),
                DXL_HIBYTE(DXL_LOWORD(raw)),
                DXL_LOBYTE(DXL_HIWORD(raw)),
                DXL_HIBYTE(DXL_HIWORD(raw))
            ]
            for mid in motor_ids:
                sync_write.addParam(mid, param)
        sync_write.txPacket()

    def set_torque(enable):
        val = 1 if enable else 0
        for mid in ALL_MOTOR_IDS:
            pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)
        print(f"Torque {'ON' if enable else 'OFF'}")

    def set_profile(velocity=100, acceleration=50):
        for mid in ALL_MOTOR_IDS:
            pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_VELOCITY, velocity)
            pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    try:
        # 1. Read current position
        original = read_joints()
        print(f"\nCurrent position (rad): [{', '.join(f'{v:.4f}' for v in original)}]")
        print(f"Current position (deg): [{', '.join(f'{np.rad2deg(v):.1f}' for v in original)}]")

        # 2. Prepare target: move joint 0 by +10 degrees
        target = original.copy()
        target[MOVE_JOINT] += MOVE_RAD
        print(f"\nTarget: Joint {MOVE_JOINT} (Waist) += {MOVE_DEG:.0f} deg")
        print(f"  {np.rad2deg(original[MOVE_JOINT]):.1f} deg -> {np.rad2deg(target[MOVE_JOINT]):.1f} deg")

        # 3. Enable torque with slow profile
        set_torque(False)
        set_profile(velocity=80, acceleration=40)
        # Set goal to current position first to prevent jump
        write_joints(original)
        time.sleep(0.05)
        set_torque(True)
        time.sleep(0.3)

        # 4. Move to target
        print("\nMoving to target...")
        write_joints(target)
        time.sleep(2.0)

        actual = read_joints()
        print(f"Arrived at (deg): [{', '.join(f'{np.rad2deg(v):.1f}' for v in actual)}]")

        # 5. Return to original
        print("\nReturning to original position...")
        write_joints(original)
        time.sleep(2.0)

        final = read_joints()
        print(f"Final position (deg): [{', '.join(f'{np.rad2deg(v):.1f}' for v in final)}]")

        # 6. Done
        error = np.rad2deg(np.abs(final - original))
        print(f"\nPosition error (deg): [{', '.join(f'{v:.2f}' for v in error)}]")
        print("Test complete!")

    finally:
        set_torque(False)
        port.closePort()
        print("Port closed.")


if __name__ == "__main__":
    main()
