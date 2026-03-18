"""
Fine-tune gripper calibration.
Step 1: Measure follower gripper exact closed position
Step 2: Live test — leader gripper controls follower gripper with mapping

Usage:
    uv run python scripts/calibrate_gripper_fine.py
"""
import sys
import os
import time
import threading
import numpy as np

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
GRIPPER_ID = 9
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
POS_TO_RAD = 0.001534

# Current calibration
GRIPPER_LEADER_MIN   = 1545
GRIPPER_LEADER_MAX   = 2187
GRIPPER_FOLLOWER_MIN = 1050
GRIPPER_FOLLOWER_MAX = 1965


def open_arm(port_name, name):
    port = PortHandler(port_name)
    pkt = PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        raise RuntimeError(f"Failed to open {port_name}")
    port.setBaudRate(1000000)
    print(f"  [{name}] Connected on {port_name}")
    return port, pkt


def set_torque_all(port, pkt, enable):
    val = 1 if enable else 0
    for mid in ALL_MOTOR_IDS:
        pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)


def set_torque_one(port, pkt, motor_id, enable):
    pkt.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)


def read_raw(port, pkt, motor_id):
    raw, res, err = pkt.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
    if res != COMM_SUCCESS:
        return None
    return raw


def write_raw(port, pkt, motor_id, raw_pos):
    raw_pos = max(0, min(4095, raw_pos))
    pkt.write4ByteTxRx(port, motor_id, ADDR_GOAL_POSITION, raw_pos)


def wait_for_enter():
    """Block until Enter is pressed. Returns immediately."""
    stop = threading.Event()
    def _wait():
        input()
        stop.set()
    t = threading.Thread(target=_wait, daemon=True)
    t.start()
    return stop


def main():
    f_port, f_pkt = open_arm("/dev/ttyDXL_puppet_right", "Follower")
    l_port, l_pkt = open_arm("/dev/ttyDXL_master_right", "Leader")

    # Torque off both arms
    set_torque_all(f_port, f_pkt, False)
    set_torque_all(l_port, l_pkt, False)

    # ── Step 1: Measure follower closed position ──────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Measure follower gripper CLOSED position")
    print("  Push the follower gripper FULLY closed (as hard as possible)")
    print("  Press ENTER to start, ENTER again to finish.")
    print("=" * 60)

    input("  Press ENTER to start measuring...")
    print("  >>> Measuring! Push gripper closed. Press ENTER to stop.\n")

    stop1 = wait_for_enter()
    min_raw = 999999

    while not stop1.is_set():
        raw = read_raw(f_port, f_pkt, GRIPPER_ID)
        if raw is not None:
            if raw < min_raw:
                min_raw = raw
            rad = (raw - 2048) * POS_TO_RAD
            min_rad = (min_raw - 2048) * POS_TO_RAD
            print(f"\r  Current: {raw:>5d} ({rad*57.2958:>6.1f} deg)  |  "
                  f"Min: {min_raw:>5d} ({min_rad*57.2958:>6.1f} deg)  ", end="", flush=True)
        time.sleep(0.03)

    new_follower_min = min_raw
    # Use the lower of measured value and hardcoded min (allow commanding past physical limit)
    effective_min = min(new_follower_min, GRIPPER_FOLLOWER_MIN)
    print(f"\n\n  Result: Follower closed = {new_follower_min} (measured)")
    print(f"  Effective MIN for mapping = {effective_min} (allows commanding past physical limit)")

    # ── Step 2: Full arm mirroring + gripper mapping test ────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Full arm mirroring + gripper mapping test")
    print(f"  Follower MIN={effective_min}, MAX={GRIPPER_FOLLOWER_MAX}")
    print("  Move LEADER arm freely. Follower follows all joints.")
    print("  Gripper uses calibrated mapping.")
    print("  Press ENTER to start, ENTER again to finish.")
    print("=" * 60)

    JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
    GRIPPER_JOINT = 6  # index in JOINT_MAP

    # Prepare gripper mapping (raw units)
    gl_min_raw = GRIPPER_LEADER_MIN
    gl_max_raw = GRIPPER_LEADER_MAX
    gf_min_raw = effective_min
    gf_max_raw = GRIPPER_FOLLOWER_MAX
    gl_range = gl_max_raw - gl_min_raw
    gf_range = gf_max_raw - gf_min_raw

    def map_gripper_raw(leader_raw):
        t = (leader_raw - gl_min_raw) / gl_range
        t = max(0.0, min(1.0, t))
        return int(gf_min_raw + t * gf_range)

    # Setup SyncRead for leader (all motors)
    l_sync_read = GroupSyncRead(l_port, l_pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for mid in ALL_MOTOR_IDS:
        l_sync_read.addParam(mid)

    # Setup SyncWrite for follower (all motors)
    f_sync_write = GroupSyncWrite(f_port, f_pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def read_all_raw(sync_read):
        """Read all motor raw positions."""
        result = sync_read.txRxPacket()
        if result != COMM_SUCCESS:
            return None
        raw_positions = {}
        for mid in ALL_MOTOR_IDS:
            if sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                raw_positions[mid] = sync_read.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        return raw_positions

    def write_all_raw(sync_write, raw_positions):
        """Write all motor raw positions."""
        sync_write.clearParam()
        for mid, raw in raw_positions.items():
            raw = max(0, min(4095, raw))
            param = [
                DXL_LOBYTE(DXL_LOWORD(raw)),
                DXL_HIBYTE(DXL_LOWORD(raw)),
                DXL_LOBYTE(DXL_HIWORD(raw)),
                DXL_HIBYTE(DXL_HIWORD(raw))
            ]
            sync_write.addParam(mid, param)
        sync_write.txPacket()

    # Read leader position, write to follower, then enable torque
    set_torque_all(f_port, f_pkt, False)
    for mid in ALL_MOTOR_IDS:
        f_pkt.write4ByteTxRx(f_port, mid, ADDR_PROFILE_VELOCITY, 150)
        f_pkt.write4ByteTxRx(f_port, mid, ADDR_PROFILE_ACCELERATION, 80)

    leader_raw = read_all_raw(l_sync_read)
    if leader_raw:
        # Apply gripper mapping before first write
        gripper_motor_id = JOINT_MAP[GRIPPER_JOINT][0]  # ID 9
        leader_raw_copy = dict(leader_raw)
        leader_raw_copy[gripper_motor_id] = map_gripper_raw(leader_raw[gripper_motor_id])
        write_all_raw(f_sync_write, leader_raw_copy)
    time.sleep(0.1)
    set_torque_all(f_port, f_pkt, True)

    input("  Press ENTER to start live test...")
    print("  >>> Running! Move leader arm + gripper. Press ENTER to stop.\n")

    stop2 = wait_for_enter()

    while not stop2.is_set():
        leader_raw = read_all_raw(l_sync_read)
        if leader_raw is None:
            continue

        # Copy all positions, apply gripper mapping
        follower_raw = dict(leader_raw)
        grip_id = JOINT_MAP[GRIPPER_JOINT][0]
        l_grip = leader_raw[grip_id]
        f_grip = map_gripper_raw(l_grip)
        follower_raw[grip_id] = f_grip

        write_all_raw(f_sync_write, follower_raw)

        # Display
        l_pct = max(0, min(100, (l_grip - gl_min_raw) / gl_range * 100))
        f_pct = max(0, min(100, (f_grip - gf_min_raw) / gf_range * 100))
        print(f"\r  Gripper: Leader {l_grip:>5d} ({l_pct:>5.1f}%) → "
              f"Follower {f_grip:>5d} ({f_pct:>5.1f}%)  |  "
              f"All joints mirroring  ", end="", flush=True)

        time.sleep(0.02)

    # Cleanup
    set_torque_all(f_port, f_pkt, False)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  FINAL VALUES")
    print("=" * 60)
    print(f"    GRIPPER_LEADER_MIN   = {GRIPPER_LEADER_MIN}")
    print(f"    GRIPPER_LEADER_MAX   = {GRIPPER_LEADER_MAX}")
    print(f"    GRIPPER_FOLLOWER_MIN = {new_follower_min}  {'(changed!)' if new_follower_min != GRIPPER_FOLLOWER_MIN else '(unchanged)'}")
    print(f"    GRIPPER_FOLLOWER_MAX = {GRIPPER_FOLLOWER_MAX}")

    l_port.closePort()
    f_port.closePort()
    print("\n  Done.")


if __name__ == "__main__":
    main()
