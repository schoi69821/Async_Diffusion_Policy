"""
Teach and Replay Script
- Phase 1 (Teaching): Move the leader arm, follower arm mirrors in real-time. Trajectory is recorded.
- Phase 2 (Replay): Follower arm replays the recorded trajectory.

Usage:
    uv run python scripts/teach_and_replay.py
    uv run python scripts/teach_and_replay.py --freq 100 --save trajectory.npy
"""
import numpy as np
import time
import signal
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamixel Protocol 2.0
PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

# VX300s joint map: 9 motors -> 7 joints
# Joint 0: Waist       [ID 1]
# Joint 1: Shoulder    [ID 2, 3] (dual motor)
# Joint 2: Elbow       [ID 4, 5] (dual motor)
# Joint 3: Forearm     [ID 6]
# Joint 4: Wrist Angle [ID 7]
# Joint 5: Wrist Rot   [ID 8]
# Joint 6: Gripper     [ID 9]
JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_JOINTS = 7

# Position conversion
POS_TO_RAD = 0.001534  # 4096 units = 2*pi rad


class ArmDriver:
    """Low-level driver for a single VX300s arm."""

    def __init__(self, port_name: str, name: str = "arm", read_only: bool = False):
        self.name = name
        self.read_only = read_only
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)
        self._sync_read = None
        self._sync_write = None

        if not self.port.openPort():
            raise RuntimeError(f"[{name}] Failed to open port: {port_name}")
        if not self.port.setBaudRate(1000000):
            raise RuntimeError(f"[{name}] Failed to set baudrate")

        # Init sync read (always needed)
        self._sync_read = GroupSyncRead(
            self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for mid in ALL_MOTOR_IDS:
            self._sync_read.addParam(mid)

        # Init sync write (only if not read-only)
        if not read_only:
            self._sync_write = GroupSyncWrite(
                self.port, self.pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
            )

        mode_str = "READ-ONLY" if read_only else "READ/WRITE"
        logger.info(f"[{name}] Connected on {port_name} ({mode_str})")

    def set_torque(self, enable: bool):
        """Enable or disable torque on all motors."""
        val = 1 if enable else 0
        for mid in ALL_MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, val)
        logger.info(f"[{self.name}] Torque {'ON' if enable else 'OFF'}")

    def set_profile(self, velocity: int = 100, acceleration: int = 50):
        """Set motion profile for smooth movement."""
        for mid in ALL_MOTOR_IDS:
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_VELOCITY, velocity)
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    def read_joints(self) -> np.ndarray:
        """Read joint positions in radians (7 joints from 9 motors)."""
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)

        result = self._sync_read.txRxPacket()
        if result != COMM_SUCCESS:
            logger.warning(f"[{self.name}] Sync read failed: {result}")
            return joints

        for j, motor_ids in enumerate(JOINT_MAP):
            positions = []
            for mid in motor_ids:
                if self._sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    raw = self._sync_read.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    positions.append((raw - 2048) * POS_TO_RAD)
            if positions:
                joints[j] = np.mean(positions)

        return joints

    def write_joints(self, joints: np.ndarray):
        """Write joint positions in radians (7 joints -> 9 motors)."""
        self._sync_write.clearParam()

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
                self._sync_write.addParam(mid, param)

        self._sync_write.txPacket()

    def close(self):
        if not self.read_only:
            self.set_torque(False)
        self.port.closePort()
        logger.info(f"[{self.name}] Closed")


def run_teaching(leader: ArmDriver, follower: ArmDriver, freq: float) -> list:
    """Phase 1: Mirror leader -> follower in real-time and record trajectory."""
    print("\n" + "=" * 60)
    print("  TEACHING MODE")
    print("  Move the teaching arm. The execution arm will follow.")
    print("  Press Ctrl+C to stop and proceed to replay.")
    print("=" * 60 + "\n")

    # Leader: read-only mode, no commands sent (torque already OFF from power cycle)

    # Read leader's current position BEFORE enabling follower torque
    initial_joints = leader.read_joints()
    print(f"  Initial leader position: [{', '.join(f'{v:.2f}' for v in initial_joints)}]")

    # Follower: set goal to leader's current position first, then enable torque
    follower.set_torque(False)
    follower.set_profile(velocity=150, acceleration=80)
    follower.write_joints(initial_joints)
    time.sleep(0.1)
    follower.set_torque(True)

    trajectory = []
    dt = 1.0 / freq
    stop = False

    def on_stop(sig, frame):
        nonlocal stop
        stop = True

    prev_handler = signal.signal(signal.SIGINT, on_stop)

    step = 0
    try:
        while not stop:
            t0 = time.perf_counter()

            # Read leader joint positions
            leader_joints = leader.read_joints()

            # Send to follower
            follower.write_joints(leader_joints)

            # Record
            trajectory.append(leader_joints.copy())

            step += 1
            if step % int(freq) == 0:
                print(f"  Recording... {len(trajectory)} frames "
                      f"({len(trajectory)/freq:.1f}s) | "
                      f"joints: [{', '.join(f'{v:.2f}' for v in leader_joints)}]")

            # Maintain frequency
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    finally:
        signal.signal(signal.SIGINT, prev_handler)

    print(f"\n  Recording complete: {len(trajectory)} frames ({len(trajectory)/freq:.1f}s)")
    return trajectory


def run_replay(follower: ArmDriver, trajectory: list, freq: float):
    """Phase 2: Replay recorded trajectory on the follower arm."""
    if not trajectory:
        print("  No trajectory recorded!")
        return

    print("\n" + "=" * 60)
    print("  REPLAY MODE")
    print(f"  Replaying {len(trajectory)} frames ({len(trajectory)/freq:.1f}s)")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    follower.set_torque(True)
    follower.set_profile(velocity=150, acceleration=80)

    dt = 1.0 / freq
    stop = False

    def on_stop(sig, frame):
        nonlocal stop
        stop = True

    prev_handler = signal.signal(signal.SIGINT, on_stop)

    try:
        for i, joints in enumerate(trajectory):
            if stop:
                print("  Replay interrupted!")
                break

            t0 = time.perf_counter()

            follower.write_joints(joints)

            if (i + 1) % int(freq) == 0:
                progress = (i + 1) / len(trajectory) * 100
                print(f"  Replaying... {progress:.0f}% ({i+1}/{len(trajectory)})")

            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

        if not stop:
            print("  Replay complete!")

    finally:
        signal.signal(signal.SIGINT, prev_handler)


def main():
    parser = argparse.ArgumentParser(description="Teach and Replay for VX300s dual arm")
    parser.add_argument("--port-follower", default="/dev/ttyDXL_puppet_right", help="Follower (execution) arm port")
    parser.add_argument("--port-leader", default="/dev/ttyDXL_master_right", help="Leader (teaching) arm port")
    parser.add_argument("--freq", type=float, default=100.0, help="Control frequency in Hz")
    parser.add_argument("--save", type=str, default=None, help="Save trajectory to .npy file")
    parser.add_argument("--load", type=str, default=None, help="Load trajectory from .npy and replay (skip teaching)")
    parser.add_argument("--replay-count", type=int, default=1, help="Number of replay repetitions")
    args = parser.parse_args()

    leader = None
    follower = None

    try:
        # Connect arms (leader is read-only: no write commands to avoid locking)
        follower = ArmDriver(args.port_follower, "Follower", read_only=False)
        leader = ArmDriver(args.port_leader, "Leader", read_only=True)

        if args.load:
            # Load and replay
            trajectory = list(np.load(args.load))
            print(f"Loaded trajectory: {len(trajectory)} frames from {args.load}")
        else:
            # Teaching phase
            trajectory = run_teaching(leader, follower, args.freq)

            # Save if requested
            if args.save and trajectory:
                np.save(args.save, np.array(trajectory))
                print(f"  Trajectory saved to {args.save}")

        # Replay phase
        for rep in range(args.replay_count):
            if args.replay_count > 1:
                print(f"\n--- Replay {rep + 1}/{args.replay_count} ---")
            run_replay(follower, trajectory, args.freq)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    finally:
        print("\nShutting down...")
        if follower:
            follower.close()
        if leader:
            leader.close()
        print("Done.")


if __name__ == "__main__":
    main()
