"""
Run trained vision diffusion policy on the robot.
Implements Chi et al. Diffusion Policy inference pipeline:
  - Temporal ensemble: weighted average of overlapping predictions
  - Action chunking: execute subset, blend remainder with next prediction
  - Deterministic denoising: fixed generator for consistent predictions
  - obs_horizon support: multi-frame observation buffer

Usage:
    uv run python scripts/run_vision_policy.py --checkpoint checkpoints/vision_policy_v5/best.pth
    uv run python scripts/run_vision_policy.py --checkpoint checkpoints/vision_policy_v5/best.pth --chunk-size 8
"""
import numpy as np
import cv2
import torch
import json
import time
import threading
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamixel_sdk import (
    PortHandler, PacketHandler,
    GroupSyncRead, GroupSyncWrite,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
    DXL_LOWORD, DXL_HIWORD
)
from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler

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
POS_TO_RAD = 0.001534
IMG_SIZE = 224

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

# Max joint movement per step (radians)
MAX_STEP_RAD = 0.15  # ~8.6 deg/step at 15Hz → max ~130 deg/s


def clamp_joints(joints):
    """Clamp joint positions to safe limits."""
    return np.clip(joints, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def safe_step(target, current):
    """Limit per-step movement and clamp to joint limits. Emergency stop if read failed."""
    if np.all(np.abs(current) < 1e-6):
        return None
    delta = target - current
    delta = np.clip(delta, -MAX_STEP_RAD, MAX_STEP_RAD)
    result = current + delta
    return clamp_joints(result)


class TemporalEnsemble:
    """
    Temporal ensemble from Chi et al. Diffusion Policy.

    Maintains overlapping predictions. For each execution timestep,
    computes exponentially-weighted average across all predictions covering it.

    Weight for step i within a prediction: exp(-k * i)
    → earlier steps (near-future) weighted more heavily.
    """

    def __init__(self, action_dim=7, k=0.01):
        self.action_dim = action_dim
        self.k = k
        self.predictions = []  # list of (global_start_step, trajectory_array)
        self.global_step = 0

    def add_prediction(self, trajectory):
        """Add a new prediction starting at current global step."""
        self.predictions.append((self.global_step, trajectory.copy()))
        self._prune()

    def get_action(self, step_offset=0):
        """Get ensembled action for global_step + step_offset."""
        target = self.global_step + step_offset
        weighted_sum = np.zeros(self.action_dim, dtype=np.float64)
        weight_sum = 0.0

        for start, traj in self.predictions:
            idx = target - start
            if 0 <= idx < len(traj):
                w = np.exp(-self.k * idx)
                weighted_sum += w * traj[idx]
                weight_sum += w

        if weight_sum < 1e-8:
            return None
        return (weighted_sum / weight_sum).astype(np.float32)

    def advance(self, n_steps=1):
        """Advance global step counter after executing n steps."""
        self.global_step += n_steps
        self._prune()

    def _prune(self):
        """Remove predictions that no longer cover any future steps."""
        self.predictions = [
            (s, t) for s, t in self.predictions
            if s + len(t) > self.global_step
        ]


# Safe positions (from config/settings.py)
HOME = np.array([+0.0199, +1.6682, -1.6176, -0.0430, -0.1319, -0.0568, -0.4065], dtype=np.float32)
MID  = np.array([-0.1227, +0.7148, -0.2094, +0.0828, +0.5891, -0.0261, -1.1766], dtype=np.float32)


def _safe_go_home(arm):
    """Move arm safely: Current → MID → HOME."""
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
    def __init__(self, port_name, name="arm"):
        self.name = name
        self.port = PortHandler(port_name)
        self.pkt = PacketHandler(PROTOCOL_VERSION)
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open {port_name}")
        self.port.setBaudRate(1000000)

        self._sync_read = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for mid in ALL_MOTOR_IDS:
            self._sync_read.addParam(mid)
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
        result = self._sync_read.txRxPacket()
        if result != COMM_SUCCESS:
            for j, motor_ids in enumerate(JOINT_MAP):
                positions = []
                for mid in motor_ids:
                    raw, _, _ = self.pkt.read4ByteTxRx(self.port, mid, ADDR_PRESENT_POSITION)
                    if raw != 0:
                        positions.append((raw - 2048) * POS_TO_RAD)
                if positions:
                    joints[j] = np.mean(positions)
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
        self.set_torque(False)
        self.port.closePort()


class CameraGrabber:
    def __init__(self, cam, img_size=IMG_SIZE):
        self.cam = cam
        self.img_size = img_size
        self.latest_frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

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


def load_model(checkpoint_path, device):
    """Load trained model and dataset stats. Auto-detect obs_horizon from checkpoint."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.json")

    with open(stats_path, 'r') as f:
        stats = json.load(f)
    for k in stats:
        for sk in stats[k]:
            stats[k][sk] = np.array(stats[k][sk], dtype=np.float32)

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    obs_horizon = config.get('obs_horizon', 1)

    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7, obs_horizon=obs_horizon).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} params, val_loss={ckpt.get('val_loss', '?')}, obs_horizon={obs_horizon}")

    return model, stats, obs_horizon


def normalize_qpos(qpos, stats):
    mn, mx = stats['qpos']['min'], stats['qpos']['max']
    rng = mx - mn
    rng[rng < 1e-6] = 1.0
    return (qpos - mn) / rng * 2 - 1


def unnormalize_action(action, stats):
    mn, mx = stats['action']['min'], stats['action']['max']
    return (action + 1) / 2 * (mx - mn) + mn


def wait_for_enter():
    stop = threading.Event()
    def _w():
        input()
        stop.set()
    threading.Thread(target=_w, daemon=True).start()
    return stop


def main():
    parser = argparse.ArgumentParser(description="Run vision diffusion policy")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--port", default="/dev/ttyDXL_puppet_right")
    parser.add_argument("--cam", default="/dev/CAM_RIGHT_WRIST")
    parser.add_argument("--exec-hz", type=float, default=15.0)
    parser.add_argument("--chunk-size", type=int, default=8, help="Action chunking: execute N of 16 steps per prediction")
    parser.add_argument("--ensemble-k", type=float, default=0.01, help="Temporal ensemble decay factor (Chi et al. use 0.01)")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--duration", type=float, default=45.0, help="Expected task duration (seconds)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model (auto-detect obs_horizon)
    model, stats, obs_horizon = load_model(args.checkpoint, device)
    scheduler = get_scheduler('ddim', num_train_timesteps=100)

    # Deterministic denoising: fixed generator for consistent predictions
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    # Connect hardware
    cam = cv2.VideoCapture(args.cam)
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.cam}")

    arm = ArmDriver(args.port, "Follower")
    grabber = CameraGrabber(cam)
    grabber.start()
    time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"  VISION POLICY EXECUTION (Chi et al. pipeline)")
    print(f"  obs_horizon={obs_horizon}, chunk_size={args.chunk_size}, ensemble_k={args.ensemble_k}")
    print(f"  Execution: {args.exec_hz} Hz, deterministic denoising")
    print(f"  Press ENTER to start, ENTER again to stop.")
    print(f"{'='*60}")

    # Enable follower
    arm.set_torque(False)
    arm.set_profile(velocity=60, acceleration=30)

    # Verify read
    print("  Verifying joint read...")
    current = arm.read_joints()
    if np.all(np.abs(current) < 1e-6):
        print("  ERROR: Cannot read joint positions! Aborting for safety.")
        arm.close()
        cam.release()
        grabber.stop()
        return
    print(f"  Joint read OK: [{', '.join(f'{v:.2f}' for v in current)}]")

    arm.write_joints(current)
    time.sleep(0.05)
    arm.set_torque(True)

    if not args.auto_start:
        input("\n  Press ENTER to start policy execution...")
    print("  >>> Running! Press Ctrl+C to stop.\n")

    if not args.auto_start:
        stop = wait_for_enter()
    else:
        stop = threading.Event()

    dt = 1.0 / args.exec_hz
    step = 0
    task_start = time.perf_counter()

    # Temporal ensemble
    ensemble = TemporalEnsemble(action_dim=NUM_JOINTS, k=args.ensemble_k)

    # Observation buffer for obs_horizon > 1
    obs_imgs = []
    obs_qpos = []

    try:
        while not stop.is_set():
            t0 = time.perf_counter()

            # Progress
            elapsed = time.perf_counter() - task_start
            progress = min(elapsed / args.duration, 1.0)

            # Observe
            img = grabber.get_frame()
            qpos = arm.read_joints()

            if np.all(np.abs(qpos) < 1e-6):
                logger.warning("Joint read failed, skipping step")
                time.sleep(0.05)
                continue
            if img is None:
                time.sleep(0.01)
                continue

            qpos_norm = normalize_qpos(qpos, stats)

            # Update observation buffer
            obs_imgs.append(img)
            obs_qpos.append(qpos_norm)
            if len(obs_imgs) > obs_horizon:
                obs_imgs.pop(0)
                obs_qpos.pop(0)
            # Pad if not enough history
            while len(obs_imgs) < obs_horizon:
                obs_imgs.insert(0, obs_imgs[0].copy())
                obs_qpos.insert(0, obs_qpos[0].copy())

            # Predict action trajectory (16 steps) with deterministic denoising
            t_infer = time.perf_counter()
            if obs_horizon == 1:
                action_norm = model.get_action(obs_imgs[-1], obs_qpos[-1], scheduler,
                                                num_inference_steps=16, device=device,
                                                progress=progress, generator=gen)
            else:
                action_norm = model.get_action(obs_imgs, obs_qpos, scheduler,
                                                num_inference_steps=16, device=device,
                                                progress=progress, generator=gen)
            infer_time = (time.perf_counter() - t_infer) * 1000

            # Unnormalize
            action_traj = unnormalize_action(action_norm, stats)

            # Add to temporal ensemble
            ensemble.add_prediction(action_traj)

            # Execute chunk_size steps from ensemble
            current_pos = qpos.copy()
            for i in range(args.chunk_size):
                if stop.is_set():
                    break
                ensembled = ensemble.get_action(step_offset=i)
                if ensembled is None:
                    break
                safe_cmd = safe_step(ensembled, current_pos)
                if safe_cmd is None:
                    logger.warning("Safety: blocked due to invalid state")
                    break
                arm.write_joints(safe_cmd)
                current_pos = safe_cmd.copy()
                time.sleep(dt)

            # Advance ensemble by executed steps
            ensemble.advance(args.chunk_size)

            step += 1
            if step % 5 == 0:
                pred_deg = [np.rad2deg(v) for v in action_traj[0]]
                qpos_deg = [np.rad2deg(v) for v in qpos]
                delta_deg = [p - q for p, q in zip(pred_deg, qpos_deg)]
                n_preds = len(ensemble.predictions)
                print(f"  Step {step} | {infer_time:.0f}ms | prog={progress:.2f} | ens={n_preds} | "
                      f"qpos: [{', '.join(f'{v:+.0f}' for v in qpos_deg)}] | "
                      f"pred: [{', '.join(f'{v:+.0f}' for v in pred_deg)}] | "
                      f"delta: [{', '.join(f'{v:+.1f}' for v in delta_deg)}]")

    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        _safe_go_home(arm)
        arm.close()
        cam.release()
        print("\nDone.")


if __name__ == "__main__":
    main()
