"""
Register mid (safe waypoint) position interactively.
Hold the arm where you want, press Enter → position saved to config/settings.py and go_home.py.

Usage:
    uv run python scripts/register_mid.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, COMM_SUCCESS
)

PROTOCOL_VERSION = 2.0
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_JOINTS = 7
POS_TO_RAD = 0.001534


def read_joints(port, pkt, sync_read):
    joints = np.zeros(NUM_JOINTS, dtype=np.float32)
    if sync_read.txRxPacket() != COMM_SUCCESS:
        return joints
    for j, mids in enumerate(JOINT_MAP):
        pos = []
        for mid in mids:
            if sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                raw = sync_read.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                pos.append((raw - 2048) * POS_TO_RAD)
        if pos:
            joints[j] = np.mean(pos)
    return joints


def update_settings_file(mid_array):
    """Update MID_FOLLOWER in config/settings.py."""
    settings_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "settings.py"
    )
    with open(settings_path, 'r') as f:
        content = f.read()

    fmt = ", ".join(f"{v:+.4f}" for v in mid_array)
    new_line = f"    MID_FOLLOWER = np.array([{fmt}], dtype=np.float32)"

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'MID_FOLLOWER' in line and 'np.array' in line:
            lines[i] = new_line
            break

    with open(settings_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Updated: {settings_path}")


def update_go_home_file(mid_array):
    """Update MID in scripts/go_home.py."""
    go_home_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "go_home.py"
    )
    if not os.path.exists(go_home_path):
        print(f"  Skipped (not found): {go_home_path}")
        return

    with open(go_home_path, 'r') as f:
        content = f.read()

    fmt = ", ".join(f"{v:+.4f}" for v in mid_array)
    new_line = f"MID  = np.array([{fmt}], dtype=np.float32)"

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('MID') and 'np.array' in line:
            lines[i] = new_line
            break

    with open(go_home_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Updated: {go_home_path}")


def main():
    port_name = "/dev/ttyDXL_puppet_right"
    port = PortHandler(port_name)
    pkt = PacketHandler(PROTOCOL_VERSION)
    port.openPort()
    port.setBaudRate(1000000)

    sync_read = GroupSyncRead(port, pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for mid in ALL_MOTOR_IDS:
        sync_read.addParam(mid)

    # Torque OFF so user can move arm freely
    for mid in ALL_MOTOR_IDS:
        pkt.write1ByteTxRx(port, mid, 64, 0)

    print("\n  팔을 원하는 중간 위치로 잡고 있으세요.")
    print("  준비되면 Enter를 누르세요.\n")

    input("  >>> Enter를 누르면 현재 위치를 MID로 등록합니다...")

    joints = read_joints(port, pkt, sync_read)
    deg = [np.rad2deg(v) for v in joints]

    print(f"\n  읽은 위치 (rad): [{', '.join(f'{v:+.4f}' for v in joints)}]")
    print(f"  읽은 위치 (deg): [{', '.join(f'{v:+.1f}' for v in deg)}]")

    update_settings_file(joints)
    update_go_home_file(joints)

    print("\n  MID 위치 등록 완료!")

    port.closePort()


if __name__ == "__main__":
    main()
