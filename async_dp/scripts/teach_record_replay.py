"""
Teach, Record & Replay Test
- Phase 1 (Teaching): Move leader arm → follower mirrors real-time + wrist camera records video & joint trajectory
- Phase 2 (Replay): Follower replays recorded trajectory + video playback for comparison

Camera runs in a separate thread so it never blocks the control loop.

Usage:
    uv run python scripts/teach_record_replay.py
    uv run python scripts/teach_record_replay.py --freq 50 --cam /dev/CAM_Front
    uv run python scripts/teach_record_replay.py --replay-only recordings/20260310_123456  # skip teaching
"""
import numpy as np
import cv2
import time
import threading
import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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
GRIPPER_IDX = 6  # Joint index for gripper
POS_TO_RAD = 0.001534

# Gripper calibration (raw units) — from calibrate_gripper.py
GRIPPER_LEADER_MIN   = 1545  # closed
GRIPPER_LEADER_MAX   = 2187  # open
GRIPPER_FOLLOWER_MIN = 1050  # closed (past physical limit for firm grip)
GRIPPER_FOLLOWER_MAX = 1965  # open

# Pre-compute rad ranges
_GL_MIN = (GRIPPER_LEADER_MIN - 2048) * POS_TO_RAD
_GL_MAX = (GRIPPER_LEADER_MAX - 2048) * POS_TO_RAD
_GF_MIN = (GRIPPER_FOLLOWER_MIN - 2048) * POS_TO_RAD
_GF_MAX = (GRIPPER_FOLLOWER_MAX - 2048) * POS_TO_RAD


def map_gripper_leader_to_follower(leader_rad: float) -> float:
    """Map leader gripper position to follower gripper position (linear interpolation)."""
    # Normalize to 0..1 (0=closed, 1=open)
    t = (leader_rad - _GL_MIN) / (_GL_MAX - _GL_MIN)
    t = max(0.0, min(1.0, t))
    return _GF_MIN + t * (_GF_MAX - _GF_MIN)


# Joint safety limits (radians)
JOINT_LIMITS_RAD = np.array([
    [-1.5708, +1.5708],  # waist:       ±90 deg
    [-1.5708, +1.7453],  # shoulder:    -90 to +100 deg
    [-1.7453, +1.6580],  # elbow:       -100 to +95 deg
    [-2.0944, +2.0944],  # forearm:     ±120 deg
    [-1.7453, +1.9199],  # wrist_angle: -100 to +110 deg
    [-1.5708, +1.5708],  # wrist_rot:   ±90 deg
    [-1.3963, +0.1745],  # gripper:     -80 to +10 deg
], dtype=np.float32)


def clamp_joints(joints):
    """Clamp joint positions to safe limits."""
    return np.clip(joints, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


# Safe positions (from config/settings.py)
HOME = np.array([+0.0199, +1.6682, -1.6176, -0.0430, -0.1319, -0.0568, -0.4065], dtype=np.float32)
MID  = np.array([-0.1227, +0.7148, -0.2094, +0.0828, +0.5891, -0.0261, -1.1766], dtype=np.float32)


def _safe_go_home(arm):
    """Move arm safely: Current → MID → HOME, then torque off."""
    try:
        print("  Returning to home via safe waypoint...")
        arm.set_profile(velocity=60, acceleration=30)
        arm.set_torque(True)
        arm.write_joints(MID)
        time.sleep(3.0)
        arm.write_joints(HOME)
        time.sleep(3.0)
        print("  At HOME position.")
    except Exception as e:
        print(f"  Warning: safe_go_home failed: {e}")


class ArmDriver:
    """Low-level driver for a single VX300s arm."""

    def __init__(self, port_name: str, name: str = "arm", read_only: bool = False):
        self.name = name
        self.read_only = read_only
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"[{name}] Failed to open port: {port_name}")
        if not self.port.setBaudRate(1000000):
            raise RuntimeError(f"[{name}] Failed to set baudrate")

        self._sync_read = GroupSyncRead(
            self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for mid in ALL_MOTOR_IDS:
            self._sync_read.addParam(mid)

        if not read_only:
            self._sync_write = GroupSyncWrite(
                self.port, self.pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
            )

        logger.info(f"[{name}] Connected on {port_name} ({'READ-ONLY' if read_only else 'READ/WRITE'})")

    def set_torque(self, enable: bool):
        val = 1 if enable else 0
        for mid in ALL_MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, val)
        logger.info(f"[{self.name}] Torque {'ON' if enable else 'OFF'}")

    def set_profile(self, velocity: int = 100, acceleration: int = 50):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_VELOCITY, velocity)
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    def read_joints(self) -> np.ndarray:
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        result = self._sync_read.txRxPacket()
        if result != COMM_SUCCESS:
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


# ─── Camera Recorder (separate thread) ───────────────────────────────────────

class CameraRecorder:
    """Records camera to video file in a background thread, never blocking the caller."""

    def __init__(self, cam: cv2.VideoCapture, video_path: str):
        self.cam = cam
        self.cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cam_fps = cam.get(cv2.CAP_PROP_FPS) or 30.0

        self.writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.cam_fps,
            (self.cam_w, self.cam_h)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {video_path}")

        self.frame_count = 0
        self._running = False
        self._thread = None

    def start(self):
        """Start recording in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop recording and return frame count."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.writer.release()
        return self.frame_count

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cam.read()
            if ret:
                self.writer.write(frame)
                self.frame_count += 1


# ─── Phase 1: Teaching + Recording ───────────────────────────────────────────

def run_teaching(leader: ArmDriver, follower: ArmDriver, cam: cv2.VideoCapture,
                 freq: float, save_dir: str) -> dict:
    """Mirror leader → follower, record joint trajectory + camera video.
    Camera runs in separate thread — control loop is never blocked by camera."""

    video_path = os.path.join(save_dir, "recording.mp4")
    recorder = CameraRecorder(cam, video_path)

    print("\n" + "=" * 60)
    print("  PHASE 1: TEACHING + RECORDING")
    print("  Move the teaching arm. The execution arm will follow.")
    print("  Wrist camera is recording.")
    print("=" * 60)

    # Read leader position before enabling follower
    initial_joints = leader.read_joints()
    print(f"  Initial position (deg): [{', '.join(f'{np.rad2deg(v):.1f}' for v in initial_joints)}]")

    # Set follower to leader's position, then enable torque
    follower.set_torque(False)
    follower.set_profile(velocity=150, acceleration=80)
    follower.write_joints(initial_joints)
    time.sleep(0.1)
    follower.set_torque(True)

    input("\n  Press ENTER to start teaching...")
    print("  >>> Recording started! Press ENTER again to stop.\n")

    trajectory = []
    timestamps = []
    dt = 1.0 / freq
    stop = False
    loop_times = []

    # Start camera in background
    recorder.start()

    # Background thread waits for Enter key
    def wait_for_enter():
        nonlocal stop
        input()
        stop = True

    stop_thread = threading.Thread(target=wait_for_enter, daemon=True)
    stop_thread.start()
    t_start = time.perf_counter()

    try:
        while not stop:
            t0 = time.perf_counter()

            # Read leader → map gripper → clamp → write to follower (no camera blocking!)
            leader_joints = leader.read_joints()
            follower_joints = leader_joints.copy()
            follower_joints[GRIPPER_IDX] = map_gripper_leader_to_follower(leader_joints[GRIPPER_IDX])
            follower_joints = clamp_joints(follower_joints)
            follower.write_joints(follower_joints)

            # Record follower trajectory (with mapped gripper)
            t_now = time.perf_counter() - t_start
            trajectory.append(follower_joints.copy())
            timestamps.append(t_now)

            # Track loop performance
            loop_time = time.perf_counter() - t0
            loop_times.append(loop_time)

            step = len(trajectory)
            if step % int(freq) == 0:
                avg_hz = 1.0 / (sum(loop_times[-int(freq):]) / len(loop_times[-int(freq):]))
                grip_l = np.rad2deg(leader_joints[GRIPPER_IDX])
                grip_f = np.rad2deg(follower_joints[GRIPPER_IDX])
                print(f"  Recording... {step} frames ({t_now:.1f}s) | "
                      f"control: {avg_hz:.0f} Hz | video: {recorder.frame_count} frames | "
                      f"gripper: L={grip_l:.1f} -> F={grip_f:.1f} deg")

            # Maintain frequency
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    finally:
        frame_count = recorder.stop()

    duration = timestamps[-1] if timestamps else 0
    avg_loop = np.mean(loop_times) * 1000 if loop_times else 0
    actual_hz = len(trajectory) / duration if duration > 0 else 0

    print(f"\n  Recording complete:")
    print(f"    Joint trajectory: {len(trajectory)} frames ({duration:.1f}s, {actual_hz:.1f} Hz)")
    print(f"    Video: {frame_count} frames → {video_path}")
    print(f"    Avg loop time: {avg_loop:.1f} ms (target: {dt*1000:.1f} ms)")

    # Save trajectory + timestamps
    traj_path = os.path.join(save_dir, "trajectory.npy")
    ts_path = os.path.join(save_dir, "timestamps.npy")
    np.save(traj_path, np.array(trajectory))
    np.save(ts_path, np.array(timestamps))
    print(f"    Trajectory saved → {traj_path}")

    # Save metadata
    meta = {
        'freq': freq,
        'actual_hz': actual_hz,
        'num_frames': len(trajectory),
        'video_frames': frame_count,
        'duration': duration,
        'avg_loop_ms': avg_loop,
        'cam_resolution': f'{recorder.cam_w}x{recorder.cam_h}',
        'cam_fps': recorder.cam_fps,
    }
    meta_path = os.path.join(save_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    return {
        'trajectory': trajectory,
        'timestamps': timestamps,
        'video_path': video_path,
        'duration': duration,
    }


# ─── Phase 2: Replay ─────────────────────────────────────────────────────────

def run_replay(follower: ArmDriver, cam: cv2.VideoCapture,
               recording: dict, save_dir: str):
    """Replay recorded trajectory on follower + record video for comparison.
    Camera runs in separate thread — control loop is never blocked by camera."""

    trajectory = recording['trajectory']
    timestamps = recording['timestamps']
    duration = recording['duration']

    if not trajectory:
        print("  No trajectory to replay!")
        return

    replay_video_path = os.path.join(save_dir, "replay.mp4")
    recorder = CameraRecorder(cam, replay_video_path)

    print("\n" + "=" * 60)
    print("  PHASE 2: REPLAY")
    print(f"  Replaying {len(trajectory)} frames ({duration:.1f}s)")
    print("  Wrist camera is recording replay for comparison.")
    print("  Press ENTER to stop early.")
    print("=" * 60 + "\n")

    # Move to start position smoothly
    follower.set_torque(True)
    follower.set_profile(velocity=80, acceleration=40)
    follower.write_joints(trajectory[0])
    time.sleep(1.5)

    # Switch to recording profile
    follower.set_profile(velocity=150, acceleration=80)

    stop = False

    # Background thread waits for Enter key to stop
    def wait_for_enter():
        nonlocal stop
        input()
        stop = True

    stop_thread = threading.Thread(target=wait_for_enter, daemon=True)
    stop_thread.start()

    # Start camera in background
    recorder.start()
    t_start = time.perf_counter()

    try:
        for i in range(len(trajectory)):
            if stop:
                print("  Replay interrupted!")
                break

            # Write joint command (no camera blocking!)
            follower.write_joints(trajectory[i])

            # Progress
            if (i + 1) % max(1, len(trajectory) // 10) == 0:
                progress = (i + 1) / len(trajectory) * 100
                actual = follower.read_joints()
                error = np.rad2deg(np.abs(actual - trajectory[i]))
                max_err = np.max(error)
                print(f"  Replaying... {progress:.0f}% ({i+1}/{len(trajectory)}) | "
                      f"max error: {max_err:.2f} deg | video: {recorder.frame_count} frames")

            # Timing: follow original timestamps
            if i + 1 < len(trajectory):
                target_time = timestamps[i + 1]
                elapsed = time.perf_counter() - t_start
                wait = target_time - elapsed
                if wait > 0:
                    time.sleep(wait)

        if not stop:
            print("  Replay complete!")

    finally:
        frame_count = recorder.stop()

    print(f"    Replay video: {frame_count} frames → {replay_video_path}")
    print(f"    All data saved → {save_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Teach, Record & Replay for VX300s")
    parser.add_argument("--port-follower", default="/dev/ttyDXL_puppet_right",
                        help="Follower (execution) arm port")
    parser.add_argument("--port-leader", default="/dev/ttyDXL_master_right",
                        help="Leader (teaching) arm port")
    parser.add_argument("--cam", default="/dev/CAM_RIGHT_WRIST",
                        help="Camera device (wrist camera on execution arm)")
    parser.add_argument("--freq", type=float, default=50.0,
                        help="Control/recording frequency in Hz")
    parser.add_argument("--replay-only", type=str, default=None,
                        help="Path to recording dir to replay (skip teaching)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory (default: recordings/<timestamp>)")
    args = parser.parse_args()

    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "recordings", timestamp
        )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    leader = None
    follower = None
    cam = None

    try:
        # Open camera
        cam = cv2.VideoCapture(args.cam)
        if not cam.isOpened():
            raise RuntimeError(f"Failed to open camera: {args.cam}")
        cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {args.cam} ({cam_w}x{cam_h})")

        if args.replay_only:
            # Replay-only mode: load existing recording
            rec_dir = args.replay_only
            trajectory = list(np.load(os.path.join(rec_dir, "trajectory.npy")))
            timestamps = list(np.load(os.path.join(rec_dir, "timestamps.npy")))
            duration = timestamps[-1] if timestamps else 0
            recording = {
                'trajectory': trajectory,
                'timestamps': timestamps,
                'video_path': os.path.join(rec_dir, "recording.mp4"),
                'duration': duration,
            }
            print(f"Loaded recording: {len(trajectory)} frames ({duration:.1f}s) from {rec_dir}")

            # Only need follower for replay
            follower = ArmDriver(args.port_follower, "Follower", read_only=False)
        else:
            # Full mode: teaching + replay
            follower = ArmDriver(args.port_follower, "Follower", read_only=False)
            leader = ArmDriver(args.port_leader, "Leader", read_only=True)

            # Phase 1: Teaching + Recording
            recording = run_teaching(leader, follower, cam, args.freq, save_dir)

        # Phase 2: Replay
        input("\n  Press Enter to start replay (or Ctrl+C to skip)...")
        run_replay(follower, cam, recording, save_dir)

    except KeyboardInterrupt:
        print("\n  Skipped.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        print("\nShutting down...")
        if follower:
            _safe_go_home(follower)
            follower.close()
        if leader:
            leader.close()
        if cam:
            cam.release()
        print("Done.")


if __name__ == "__main__":
    main()
