#!/usr/bin/env python3
"""Test if leader arm locks up on read commands.

Step 1: Open port only — check if arm is free to move
Step 2: Send a single position read — check if arm locks
Step 3: If OK, send continuous reads for 3 seconds

Press Ctrl+C at any time to abort.
"""
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, COMM_SUCCESS
import time
import numpy as np

PORT = "/dev/ttyDXL_master_right"
BAUDRATE = 1_000_000
PROTOCOL = 2.0
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
POS_TO_RAD = 2 * np.pi / 4096


def main():
    port = PortHandler(PORT)
    pkt = PacketHandler(PROTOCOL)

    if not port.openPort():
        print(f"FAIL: Cannot open {PORT}")
        return
    if not port.setBaudRate(BAUDRATE):
        print(f"FAIL: Cannot set baudrate")
        port.closePort()
        return

    print(f"Port opened: {PORT} @ {BAUDRATE}")
    print()

    # Step 1: port open only
    input("STEP 1: Port is open. Move the leader arm around.\n"
          "  Is it FREE to move? Press ENTER to continue to read test...")
    print()

    # Step 2: single read
    print("STEP 2: Sending single read (read4ByteTxRx on motor 1)...")
    val, result, err = pkt.read4ByteTxRx(port, 1, ADDR_PRESENT_POSITION)
    if result == COMM_SUCCESS:
        rad = (val - 2048) * POS_TO_RAD
        print(f"  Motor 1 position: {val} raw, {rad:.3f} rad — READ OK")
    else:
        print(f"  Read failed: result={result}, err={err}")

    input("\n  Is the arm still FREE? Or did it LOCK?\n"
          "  Press ENTER to continue to SyncRead test (or Ctrl+C to abort)...")
    print()

    # Step 3: SyncRead all motors
    print("STEP 3: SyncRead all 9 motors, 3 seconds continuous...")
    sync_read = GroupSyncRead(port, pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for mid in ALL_MOTOR_IDS:
        sync_read.addParam(mid)

    t_start = time.monotonic()
    count = 0
    try:
        while time.monotonic() - t_start < 3.0:
            result = sync_read.txRxPacket()
            if result != COMM_SUCCESS:
                print(f"  SyncRead failed at count={count}")
                break
            count += 1

            if count % 30 == 0:
                positions = []
                for mid in ALL_MOTOR_IDS:
                    if sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                        raw = sync_read.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                        positions.append(round((raw - 2048) * POS_TO_RAD, 3))
                elapsed = time.monotonic() - t_start
                hz = count / elapsed
                print(f"  t={elapsed:.1f}s count={count} ({hz:.0f}Hz) pos={positions}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  Interrupted")

    elapsed = time.monotonic() - t_start
    print(f"\n  Done: {count} reads in {elapsed:.1f}s ({count/max(elapsed,0.01):.0f} Hz)")

    input("\n  Is the arm still FREE? Press ENTER to close port...")
    port.closePort()
    print("Port closed.")


if __name__ == "__main__":
    main()
