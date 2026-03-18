"""
Gripper Calibration Script
Reads gripper position from both arms in real-time.
Move each gripper by hand to fully open/closed to find the range.

Usage:
    uv run python scripts/calibrate_gripper.py
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
GRIPPER_ID = 9
POS_TO_RAD = 0.001534


class GripperReader:
    def __init__(self, port_name: str, name: str):
        self.name = name
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open {port_name}")
        self.port.setBaudRate(1000000)

        # Torque OFF so gripper can be moved by hand
        self.pkt.write1ByteTxRx(self.port, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0)

        self.min_raw = 999999
        self.max_raw = 0

    def read(self):
        raw, res, err = self.pkt.read4ByteTxRx(self.port, GRIPPER_ID, ADDR_PRESENT_POSITION)
        if res != COMM_SUCCESS:
            return None, None
        rad = (raw - 2048) * POS_TO_RAD
        if raw < self.min_raw:
            self.min_raw = raw
        if raw > self.max_raw:
            self.max_raw = raw
        return raw, rad

    def close(self):
        self.port.closePort()


def main():
    leader = GripperReader("/dev/ttyDXL_master_right", "Leader (teaching)")
    follower = GripperReader("/dev/ttyDXL_puppet_right", "Follower (execution)")

    print("=" * 70)
    print("  GRIPPER CALIBRATION")
    print("  Move each gripper by hand: fully OPEN and fully CLOSED")
    print("  Press ENTER when done.")
    print("=" * 70)
    print()
    print(f"  {'':>10}  {'Leader (teaching)':>25}  {'Follower (execution)':>25}")
    print(f"  {'':>10}  {'raw':>8} {'rad':>8} {'deg':>8}  {'raw':>8} {'rad':>8} {'deg':>8}")
    print("  " + "-" * 66)

    stop = False

    def wait_for_enter():
        nonlocal stop
        input()
        stop = True

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    try:
        while not stop:
            l_raw, l_rad = leader.read()
            f_raw, f_rad = follower.read()

            if l_raw is not None and f_raw is not None:
                l_deg = l_rad * 57.2958
                f_deg = f_rad * 57.2958
                print(f"\r  {'Current':>10}  {l_raw:>8d} {l_rad:>8.4f} {l_deg:>8.1f}  "
                      f"{f_raw:>8d} {f_rad:>8.4f} {f_deg:>8.1f}  ", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\n")
    print("  " + "=" * 66)
    print("  RESULTS")
    print("  " + "=" * 66)
    print()
    print(f"  {'':>12}  {'Leader (teaching)':>25}  {'Follower (execution)':>25}")
    print(f"  {'':>12}  {'raw':>8} {'rad':>8} {'deg':>8}  {'raw':>8} {'rad':>8} {'deg':>8}")
    print("  " + "-" * 66)

    for label, l_raw, f_raw in [
        ("Min (closed)", leader.min_raw, follower.min_raw),
        ("Max (open)", leader.max_raw, follower.max_raw),
    ]:
        l_rad = (l_raw - 2048) * POS_TO_RAD
        f_rad = (f_raw - 2048) * POS_TO_RAD
        print(f"  {label:>12}  {l_raw:>8d} {l_rad:>8.4f} {l_rad*57.2958:>8.1f}  "
              f"{f_raw:>8d} {f_rad:>8.4f} {f_rad*57.2958:>8.1f}")

    l_range = leader.max_raw - leader.min_raw
    f_range = follower.max_raw - follower.min_raw
    print()
    print(f"  Leader range:   {l_range} ticks ({l_range * POS_TO_RAD:.4f} rad, {l_range * POS_TO_RAD * 57.2958:.1f} deg)")
    print(f"  Follower range: {f_range} ticks ({f_range * POS_TO_RAD:.4f} rad, {f_range * POS_TO_RAD * 57.2958:.1f} deg)")

    print()
    print("  Copy these values to config/settings.py:")
    print(f"    GRIPPER_LEADER_MIN  = {leader.min_raw}  # closed (raw)")
    print(f"    GRIPPER_LEADER_MAX  = {leader.max_raw}  # open (raw)")
    print(f"    GRIPPER_FOLLOWER_MIN = {follower.min_raw}  # closed (raw)")
    print(f"    GRIPPER_FOLLOWER_MAX = {follower.max_raw}  # open (raw)")

    leader.close()
    follower.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
