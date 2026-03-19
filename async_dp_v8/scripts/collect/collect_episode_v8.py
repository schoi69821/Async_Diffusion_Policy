#!/usr/bin/env python3
"""Collect demonstration episodes with v8 schema.

Leader arm (teaching) → Follower arm (execution) teleoperation.
Records: qpos, qvel, current, pwm, gripper, camera frames.

Usage:
    uv run python scripts/collect/collect_episode_v8.py
    uv run python scripts/collect/collect_episode_v8.py --freq 50 --episodes 10
    uv run python scripts/collect/collect_episode_v8.py --no-camera  # joint-only
"""
import argparse
import time
import threading
import h5py
import cv2
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000

# Ports (udev symlinks)
FOLLOWER_PORT = "/dev/ttyDXL_puppet_right"
LEADER_PORT = "/dev/ttyDXL_master_right"
WRIST_CAM = "/dev/CAM_RIGHT_WRIST"
FRONT_CAM = "/dev/CAM_Front"

# Motor config
JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_JOINTS = 7  # 6 arm + 1 gripper
GRIPPER_IDX = 6

# Register addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

# Extended SyncRead: PWM(124,2) + Current(126,2) + Velocity(128,4) + Position(132,4) = 12 bytes
SYNC_READ_ADDR = 124
SYNC_READ_LEN = 12

# Gripper calibration
GRIPPER_LEADER_MIN = 1545   # closed
GRIPPER_LEADER_MAX = 2187   # open
GRIPPER_FOLLOWER_MIN = 1050  # closed
GRIPPER_FOLLOWER_MAX = 1965  # open

POS_TO_RAD = 2 * np.pi / 4096

_GL_MIN = (GRIPPER_LEADER_MIN - 2048) * POS_TO_RAD
_GL_MAX = (GRIPPER_LEADER_MAX - 2048) * POS_TO_RAD
_GF_MIN = (GRIPPER_FOLLOWER_MIN - 2048) * POS_TO_RAD
_GF_MAX = (GRIPPER_FOLLOWER_MAX - 2048) * POS_TO_RAD

# Joint safety limits (radians)
JOINT_LIMITS_RAD = np.array([
    [-1.5708, +1.5708],  # waist
    [-1.5708, +1.7453],  # shoulder
    [-1.7453, +1.6580],  # elbow
    [-2.0944, +2.0944],  # forearm_roll
    [-1.7453, +1.9199],  # wrist_angle
    [-1.5708, +1.5708],  # wrist_rotate
    [-1.5708, +0.1745],  # gripper (must cover FOLLOWER_MIN=1050 → -1.53 rad)
], dtype=np.float32)

IMG_SIZE = 224


def map_gripper_leader_to_follower(leader_rad: float) -> float:
    t = (leader_rad - _GL_MIN) / (_GL_MAX - _GL_MIN)
    t = max(0.0, min(1.0, t))
    return _GF_MIN + t * (_GF_MAX - _GF_MIN)


def clamp_joints(joints: np.ndarray) -> np.ndarray:
    return np.clip(joints, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def _to_signed16(val: int) -> int:
    return val - 0x10000 if val > 0x7FFF else val


def _to_signed32(val: int) -> int:
    return val - 0x100000000 if val > 0x7FFFFFFF else val


# ─── Arm Driver (extended for v8: reads current, velocity, pwm) ──────────────

class ArmDriver:
    def __init__(self, port_name: str, name: str = "arm", read_only: bool = False):
        self.name = name
        self.read_only = read_only
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"[{name}] Failed to open port: {port_name}")
        if not self.port.setBaudRate(BAUDRATE):
            raise RuntimeError(f"[{name}] Failed to set baudrate")

        # Position-only SyncRead (for leader)
        self._sync_read_pos = GroupSyncRead(
            self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for mid in ALL_MOTOR_IDS:
            self._sync_read_pos.addParam(mid)

        # Extended SyncRead: position + velocity + current + pwm (for follower)
        self._sync_read_ext = GroupSyncRead(
            self.port, self.pkt, SYNC_READ_ADDR, SYNC_READ_LEN
        )
        for mid in ALL_MOTOR_IDS:
            self._sync_read_ext.addParam(mid)

        if not read_only:
            self._sync_write = GroupSyncWrite(
                self.port, self.pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
            )

        logger.info(f"[{name}] Connected on {port_name} ({'READ-ONLY' if read_only else 'R/W'})")

    def set_torque(self, enable: bool):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, int(enable))

    def set_profile(self, velocity: int = 100, acceleration: int = 50):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_VELOCITY, velocity)
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    def check_hardware_errors(self) -> dict:
        """Read Hardware Error Status (addr 70) for all motors. Returns {motor_id: error_byte}."""
        ADDR_HW_ERROR = 70
        errors = {}
        for mid in ALL_MOTOR_IDS:
            val, result, _ = self.pkt.read1ByteTxRx(self.port, mid, ADDR_HW_ERROR)
            if result == COMM_SUCCESS and val != 0:
                errors[mid] = val
                err_names = []
                if val & 0x01: err_names.append("InputVoltage")
                if val & 0x04: err_names.append("Overheating")
                if val & 0x08: err_names.append("MotorEncoder")
                if val & 0x10: err_names.append("ElectricalShock")
                if val & 0x20: err_names.append("Overload")
                logger.warning(f"[{self.name}] Motor {mid} HW error: {'+'.join(err_names)} (0x{val:02X})")
        return errors

    def clear_errors_and_reboot(self, motor_ids: list = None):
        """Reboot motors to clear hardware errors. Torque will be off after reboot."""
        targets = motor_ids or ALL_MOTOR_IDS
        for mid in targets:
            self.pkt.reboot(self.port, mid)
            logger.info(f"[{self.name}] Rebooted motor {mid}")
        time.sleep(0.5)  # Wait for reboot

    def read_joints(self) -> np.ndarray:
        """Read 7 joint positions (radians) via position-only SyncRead."""
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        result = self._sync_read_pos.txRxPacket()
        if result != COMM_SUCCESS:
            return joints
        for j, motor_ids in enumerate(JOINT_MAP):
            positions = []
            for mid in motor_ids:
                if self._sync_read_pos.isAvailable(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    raw = self._sync_read_pos.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    positions.append((_to_signed32(raw) - 2048) * POS_TO_RAD)
            if positions:
                joints[j] = np.mean(positions)
        return joints

    def read_full_state(self) -> dict:
        """Read extended state: position, velocity, current, pwm for all motors."""
        n = len(ALL_MOTOR_IDS)
        state = {
            "present_position": np.zeros(n, dtype=np.float32),
            "present_velocity": np.zeros(n, dtype=np.float32),
            "present_current": np.zeros(n, dtype=np.float32),
            "present_pwm": np.zeros(n, dtype=np.float32),
        }

        result = self._sync_read_ext.txRxPacket()
        if result != COMM_SUCCESS:
            logger.warning(f"[{self.name}] SyncRead failed")
            return state

        for i, mid in enumerate(ALL_MOTOR_IDS):
            if not self._sync_read_ext.isAvailable(mid, SYNC_READ_ADDR, SYNC_READ_LEN):
                continue
            pwm = self._sync_read_ext.getData(mid, 124, 2)
            cur = self._sync_read_ext.getData(mid, 126, 2)
            vel = self._sync_read_ext.getData(mid, 128, 4)
            pos = self._sync_read_ext.getData(mid, 132, 4)

            state["present_position"][i] = (_to_signed32(pos) - 2048) * POS_TO_RAD
            state["present_velocity"][i] = _to_signed32(vel) * 0.229 * (2 * np.pi / 60)
            state["present_current"][i] = float(_to_signed16(cur))
            state["present_pwm"][i] = float(_to_signed16(pwm))

        return state

    def motors_to_joints(self, motor_vals: np.ndarray) -> np.ndarray:
        """Map 9 motor values to 7 logical joints (average dual-motor joints)."""
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        idx = 0
        for j, motor_ids in enumerate(JOINT_MAP):
            n = len(motor_ids)
            if idx + n <= len(motor_vals):
                joints[j] = np.mean(motor_vals[idx:idx + n])
            elif idx < len(motor_vals):
                joints[j] = motor_vals[idx]
            idx += n
        return joints

    def write_joints(self, joints: np.ndarray):
        self._sync_write.clearParam()
        for j, motor_ids in enumerate(JOINT_MAP):
            raw = int(joints[j] / POS_TO_RAD + 2048)
            raw = max(0, min(4095, raw))
            param = [
                DXL_LOBYTE(DXL_LOWORD(raw)),
                DXL_HIBYTE(DXL_LOWORD(raw)),
                DXL_LOBYTE(DXL_HIWORD(raw)),
                DXL_HIBYTE(DXL_HIWORD(raw)),
            ]
            for mid in motor_ids:
                self._sync_write.addParam(mid, param)
        self._sync_write.txPacket()

    def close(self):
        if not self.read_only:
            self.set_torque(False)
        self.port.closePort()


# ─── Camera Grabber (threaded, non-blocking) ─────────────────────────────────

class CameraGrabber:
    """Grabs camera frames in a background thread."""

    def __init__(self, device: str, size: int = IMG_SIZE):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")
        self.size = size
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.cap.release()

    def grab(self) -> np.ndarray:
        """Get latest frame as [H, W, 3] RGB uint8."""
        with self._lock:
            if self._frame is None:
                return np.zeros((self.size, self.size, 3), dtype=np.uint8)
            return self._frame.copy()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.size, self.size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = frame


# ─── Episode Collection ──────────────────────────────────────────────────────

def collect_episode(
    leader: ArmDriver,
    follower: ArmDriver,
    camera: CameraGrabber = None,
    freq: float = 30.0,
) -> dict:
    """Collect one teaching episode.

    Returns dict with all v8 fields.
    """
    dt = 1.0 / freq

    # Check and clear any hardware errors before starting
    errors = follower.check_hardware_errors()
    if errors:
        logger.warning(f"Hardware errors detected on motors {list(errors.keys())}, rebooting...")
        follower.clear_errors_and_reboot(list(errors.keys()))
        # Re-check after reboot
        errors = follower.check_hardware_errors()
        if errors:
            logger.error(f"Errors persist after reboot: {errors}")
            raise RuntimeError("Cannot start episode: motor hardware errors")

    # Read leader position, set follower
    initial = leader.read_joints()
    follower.set_torque(False)
    follower.set_profile(velocity=150, acceleration=80)
    follower.write_joints(initial)
    time.sleep(0.1)
    follower.set_torque(True)

    input("\n  Press ENTER to start recording...")
    print("  >>> Recording! Press ENTER to stop.\n")

    # Data lists
    timestamps = []
    qpos_list = []       # [T, 7] follower joints (6 arm + gripper)
    qvel_list = []       # [T, 7]
    current_list = []    # [T, 7]
    pwm_list = []        # [T, 7]
    image_list = []      # [T, H, W, 3]

    stop = False

    def wait_enter():
        nonlocal stop
        input()
        stop = True

    stop_thread = threading.Thread(target=wait_enter, daemon=True)
    stop_thread.start()
    t_start = time.perf_counter()

    try:
        while not stop:
            t0 = time.perf_counter()

            # 1. Read leader
            leader_joints = leader.read_joints()

            # 2. Map gripper + clamp
            follower_joints = leader_joints.copy()
            follower_joints[GRIPPER_IDX] = map_gripper_leader_to_follower(
                leader_joints[GRIPPER_IDX]
            )
            follower_joints = clamp_joints(follower_joints)

            # 3. Write to follower
            follower.write_joints(follower_joints)

            # 4. Read follower extended state (position, velocity, current, pwm)
            full_state = follower.read_full_state()
            joints_pos = follower.motors_to_joints(full_state["present_position"])
            joints_vel = follower.motors_to_joints(full_state["present_velocity"])
            joints_cur = follower.motors_to_joints(full_state["present_current"])
            joints_pwm = follower.motors_to_joints(full_state["present_pwm"])

            # 5. Record
            t_now = time.perf_counter() - t_start
            timestamps.append(t_now)
            qpos_list.append(joints_pos)
            qvel_list.append(joints_vel)
            current_list.append(joints_cur)
            pwm_list.append(joints_pwm)

            # 6. Camera frame
            if camera is not None:
                image_list.append(camera.grab())

            # Progress display + gripper health check
            step = len(timestamps)
            if step % 30 == 0:
                grip_cmd = follower_joints[GRIPPER_IDX]
                grip_actual = joints_pos[GRIPPER_IDX]
                grip_cur = joints_cur[GRIPPER_IDX]
                grip_err = abs(grip_cmd - grip_actual)
                print(f"\r  Step {step:4d} | t={t_now:.1f}s | "
                      f"grip_cmd={grip_cmd:.3f} grip_act={grip_actual:.3f} "
                      f"err={grip_err:.3f} cur={grip_cur:.0f}mA", end="", flush=True)

                # Warn if gripper isn't tracking commands
                if grip_err > 0.3 and step > 60:
                    logger.warning(f"\n  Gripper tracking error high: {grip_err:.3f} rad. "
                                   f"Possible motor error. Checking...")
                    errors = follower.check_hardware_errors()
                    if errors:
                        logger.warning(f"  Motor errors: {errors}. Rebooting affected motors...")
                        follower.set_torque(False)
                        follower.clear_errors_and_reboot(list(errors.keys()))
                        follower.set_profile(velocity=150, acceleration=80)
                        follower.set_torque(True)

            # Rate control
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass

    duration = timestamps[-1] if timestamps else 0
    print(f"\n  Recording done: {len(timestamps)} frames, {duration:.1f}s")

    return {
        "timestamps": np.array(timestamps, dtype=np.float64),
        "qpos": np.array(qpos_list, dtype=np.float32),
        "qvel": np.array(qvel_list, dtype=np.float32),
        "current": np.array(current_list, dtype=np.float32),
        "pwm": np.array(pwm_list, dtype=np.float32),
        "images": np.array(image_list, dtype=np.uint8) if image_list else None,
    }


def save_episode(data: dict, path: str):
    """Save episode to HDF5."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp", data=data["timestamps"])
        f.create_dataset("qpos", data=data["qpos"])
        f.create_dataset("qvel", data=data["qvel"])
        f.create_dataset("current", data=data["current"])
        f.create_dataset("pwm", data=data["pwm"])

        if data["images"] is not None:
            f.create_dataset(
                "images", data=data["images"],
                chunks=(1, *data["images"].shape[1:]),
                compression="gzip", compression_opts=4,
            )

        f.attrs["num_timesteps"] = len(data["timestamps"])
        f.attrs["num_joints"] = data["qpos"].shape[1]
        f.attrs["has_images"] = data["images"] is not None
        f.attrs["timestamp"] = datetime.now().isoformat()

    size_mb = Path(path).stat().st_size / 1024 / 1024
    logger.info(f"Saved {path} ({size_mb:.1f}MB, {len(data['timestamps'])} frames)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect v8 teaching episodes")
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--cam", default=WRIST_CAM, help="Camera device")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera")
    parser.add_argument("--freq", type=float, default=30.0, help="Recording Hz")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--output-dir", default="data/raw/pen_fixed_hdf5")
    parser.add_argument("--prefix", default="episode", help="Episode filename prefix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ASYNC DP V8 - EPISODE COLLECTION")
    print(f"  Leader:   {args.leader_port}")
    print(f"  Follower: {args.follower_port}")
    print(f"  Camera:   {'OFF' if args.no_camera else args.cam}")
    print(f"  Freq:     {args.freq} Hz")
    print(f"  Episodes: {args.episodes}")
    print("=" * 60)

    # Connect arms
    leader = ArmDriver(args.leader_port, "leader", read_only=True)
    follower = ArmDriver(args.follower_port, "follower", read_only=False)

    # Camera
    camera = None
    if not args.no_camera:
        try:
            camera = CameraGrabber(args.cam)
            camera.start()
            logger.info(f"Camera started: {args.cam}")
        except RuntimeError as e:
            logger.warning(f"Camera failed: {e}. Continuing without camera.")
            camera = None

    try:
        for ep_idx in range(args.episodes):
            print(f"\n{'─'*60}")
            print(f"  Episode {ep_idx + 1}/{args.episodes}")
            print(f"{'─'*60}")

            data = collect_episode(leader, follower, camera, args.freq)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{args.prefix}_{ts}_{ep_idx:03d}"
            save_episode(data, str(output_dir / f"{name}.hdf5"))

            if ep_idx < args.episodes - 1:
                input("\n  Reset the scene, then press ENTER for next episode...")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        if camera:
            camera.stop()
        follower.close()
        leader.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
