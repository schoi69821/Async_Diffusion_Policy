"""
Collect demonstration episodes for vision-based diffusion policy training.
Each episode saves joint positions + camera images to HDF5.

Usage:
    uv run python scripts/collect_episodes.py
    uv run python scripts/collect_episodes.py --num-episodes 20 --cam /dev/CAM_Front
    uv run python scripts/collect_episodes.py --save-dir episodes/pen_pick
"""
import numpy as np
import cv2
import h5py
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
GRIPPER_IDX = 6
POS_TO_RAD = 0.001534

# Gripper calibration
GRIPPER_LEADER_MIN   = 1545
GRIPPER_LEADER_MAX   = 2187
GRIPPER_FOLLOWER_MIN = 1050
GRIPPER_FOLLOWER_MAX = 1965

_GL_MIN = (GRIPPER_LEADER_MIN - 2048) * POS_TO_RAD
_GL_MAX = (GRIPPER_LEADER_MAX - 2048) * POS_TO_RAD
_GF_MIN = (GRIPPER_FOLLOWER_MIN - 2048) * POS_TO_RAD
_GF_MAX = (GRIPPER_FOLLOWER_MAX - 2048) * POS_TO_RAD

IMG_SIZE = 224  # ResNet input size

# Joint safety limits (radians) — prevents wrist twisting etc.
JOINT_LIMITS_RAD = np.array([
    [-1.5708, +1.5708],  # waist:       ±90 deg
    [-1.5708, +1.7453],  # shoulder:    -90 to +100 deg
    [-1.7453, +1.6580],  # elbow:       -100 to +95 deg
    [-2.0944, +2.0944],  # forearm:     ±120 deg
    [-1.7453, +1.9199],  # wrist_angle: -100 to +110 deg
    [-1.5708, +1.5708],  # wrist_rot:   ±90 deg
    [-1.3963, +0.1745],  # gripper:     -80 to +10 deg
], dtype=np.float32)


def map_gripper(leader_rad):
    t = (leader_rad - _GL_MIN) / (_GL_MAX - _GL_MIN)
    t = max(0.0, min(1.0, t))
    return _GF_MIN + t * (_GF_MAX - _GF_MIN)


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
    def __init__(self, port_name, name="arm", read_only=False):
        self.name = name
        self.read_only = read_only
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"[{name}] Failed to open {port_name}")
        self.port.setBaudRate(1000000)

        self._sync_read = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for mid in ALL_MOTOR_IDS:
            self._sync_read.addParam(mid)

        if not read_only:
            self._sync_write = GroupSyncWrite(self.port, self.pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        logger.info(f"[{name}] Connected on {port_name}")

    def set_torque(self, enable):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def set_profile(self, velocity=100, acceleration=50):
        for mid in ALL_MOTOR_IDS:
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_VELOCITY, velocity)
            self.pkt.write4ByteTxRx(self.port, mid, ADDR_PROFILE_ACCELERATION, acceleration)

    def read_joints(self):
        joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self._sync_read.txRxPacket() != COMM_SUCCESS:
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

    def write_joints(self, joints):
        self._sync_write.clearParam()
        for j, motor_ids in enumerate(JOINT_MAP):
            raw = int(joints[j] / POS_TO_RAD + 2048)
            raw = max(0, min(4095, raw))
            param = [DXL_LOBYTE(DXL_LOWORD(raw)), DXL_HIBYTE(DXL_LOWORD(raw)),
                     DXL_LOBYTE(DXL_HIWORD(raw)), DXL_HIBYTE(DXL_HIWORD(raw))]
            for mid in motor_ids:
                self._sync_write.addParam(mid, param)
        self._sync_write.txPacket()

    def close(self):
        if not self.read_only:
            self.set_torque(False)
        self.port.closePort()


class CameraGrabber:
    """Grabs latest camera frame in background thread (non-blocking)."""

    def __init__(self, cam, img_size=IMG_SIZE):
        self.cam = cam
        self.img_size = img_size
        self.latest_frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_frame(self):
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _loop(self):
        while self._running:
            ret, frame = self.cam.read()
            if ret:
                resized = cv2.resize(frame, (self.img_size, self.img_size))
                with self._lock:
                    self.latest_frame = resized

    def stop(self):
        self._running = False


def wait_for_enter():
    stop = threading.Event()
    def _wait():
        input()
        stop.set()
    threading.Thread(target=_wait, daemon=True).start()
    return stop


def collect_one_episode(leader, follower, grabber, freq):
    """Collect a single demonstration episode. Returns (qpos_list, action_list, images_list)."""

    # Read leader, sync follower (with safety clamp)
    initial = leader.read_joints()
    follower_joints = initial.copy()
    follower_joints[GRIPPER_IDX] = map_gripper(initial[GRIPPER_IDX])
    follower_joints = clamp_joints(follower_joints)
    follower.set_torque(False)
    follower.set_profile(velocity=150, acceleration=80)
    follower.write_joints(follower_joints)
    time.sleep(0.1)
    follower.set_torque(True)

    input("  Press ENTER to start recording...")
    print("  >>> Recording! Press ENTER to stop.\n")

    qpos_list = []
    action_list = []
    images_list = []
    dt = 1.0 / freq

    stop = wait_for_enter()
    t_start = time.perf_counter()

    while not stop.is_set():
        t0 = time.perf_counter()

        # Read leader, map gripper, clamp to safe limits, write to follower
        leader_joints = leader.read_joints()
        follower_joints = leader_joints.copy()
        follower_joints[GRIPPER_IDX] = map_gripper(leader_joints[GRIPPER_IDX])
        follower_joints = clamp_joints(follower_joints)
        follower.write_joints(follower_joints)

        # Read follower actual position (observation)
        follower_actual = follower.read_joints()

        # Grab camera frame
        frame = grabber.get_frame()

        # Store: qpos = current follower state, action = commanded follower position
        if frame is not None:
            qpos_list.append(follower_actual.copy())
            action_list.append(follower_joints.copy())
            images_list.append(frame)

        elapsed = time.perf_counter() - t0
        step = len(qpos_list)
        if step > 0 and step % int(freq) == 0:
            t_now = time.perf_counter() - t_start
            hz = step / t_now if t_now > 0 else 0
            print(f"  Recording... {step} frames ({t_now:.1f}s, {hz:.0f} Hz)")

        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    duration = time.perf_counter() - t_start
    print(f"  Episode done: {len(qpos_list)} frames ({duration:.1f}s)")
    return qpos_list, action_list, images_list


def save_episode_hdf5(save_path, qpos_list, action_list, images_list):
    """Save episode to HDF5 file."""
    qpos = np.array(qpos_list, dtype=np.float32)    # (T, 7)
    action = np.array(action_list, dtype=np.float32)  # (T, 7)
    images = np.array(images_list, dtype=np.uint8)     # (T, 224, 224, 3)

    with h5py.File(save_path, 'w') as f:
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos)
        obs.create_dataset('images', data=images, chunks=(1, *images.shape[1:]),
                           compression='gzip', compression_opts=4)
        f.create_dataset('action', data=action)
        f.attrs['num_timesteps'] = len(qpos)
        f.attrs['timestamp'] = datetime.now().isoformat()

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"  Saved: {save_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration episodes")
    parser.add_argument("--port-follower", default="/dev/ttyDXL_puppet_right")
    parser.add_argument("--port-leader", default="/dev/ttyDXL_master_right")
    parser.add_argument("--cam", default="/dev/CAM_RIGHT_WRIST")
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "episodes", timestamp
        )
    os.makedirs(save_dir, exist_ok=True)

    leader = None
    follower = None
    cam = None

    try:
        cam = cv2.VideoCapture(args.cam)
        if not cam.isOpened():
            raise RuntimeError(f"Failed to open camera: {args.cam}")

        follower = ArmDriver(args.port_follower, "Follower", read_only=False)
        leader = ArmDriver(args.port_leader, "Leader", read_only=True)

        grabber = CameraGrabber(cam)
        grabber.start()
        time.sleep(0.5)  # Let camera warm up

        print(f"\n{'='*60}")
        print(f"  EPISODE COLLECTION")
        print(f"  Target: {args.num_episodes} episodes")
        print(f"  Save dir: {save_dir}")
        print(f"  Camera: {args.cam} → {IMG_SIZE}x{IMG_SIZE}")
        print(f"{'='*60}\n")

        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
            print("  Position the pen, then press ENTER when ready.")

            qpos, action, images = collect_one_episode(leader, follower, grabber, args.freq)

            if len(qpos) < 10:
                print("  Too short, skipping.")
                continue

            ep_path = os.path.join(save_dir, f"episode_{ep:04d}.hdf5")
            save_episode_hdf5(ep_path, qpos, action, images)

        grabber.stop()

        # Summary
        files = [f for f in os.listdir(save_dir) if f.endswith('.hdf5')]
        print(f"\n{'='*60}")
        print(f"  COLLECTION COMPLETE")
        print(f"  {len(files)} episodes saved to {save_dir}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("\nShutting down...")
        if follower:
            _safe_go_home(follower)
            follower.close()
        if leader:
            leader.close()
        if cam:
            cam.release()


if __name__ == "__main__":
    main()
