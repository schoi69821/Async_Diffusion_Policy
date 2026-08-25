#!/usr/bin/env python3
"""Deploy v8 policy on real robot.

Synchronous control loop at 15 Hz:
  1. Read robot state + capture camera
  2. Build observation batch (3-frame history, normalized)
  3. Run DDIM inference (10 steps ~175ms on RTX 3060)
  4. Send arm command with safety clamping
  5. Log state machine transitions
"""
import argparse
import time
import signal
import sys
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from collections import deque

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.robot.safety import SafetyGuard
from async_dp_v8.inference.policy_runner import PolicyRunnerV8
from async_dp_v8.inference.ddim_sampler import DDIMSampler
from async_dp_v8.types import InferenceConfig
from async_dp_v8.train.engine import NoiseScheduler
from async_dp_v8.utils.checkpointing import load_checkpoint
from async_dp_v8.utils.normalization import Normalizer
from async_dp_v8.constants import (
    FOLLOWER_PORT, DXL_BAUDRATE, OBS_HORIZON, IMAGE_SIZE, CROP_SIZE,
    NUM_JOINTS, NUM_MOTORS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WRIST_CAM_DEV = "/dev/CAM_RIGHT_WRIST"
FRONT_CAM_DEV = "/dev/CAM_Front"


class ObservationBuilder:
    """Maintains observation history and builds batched tensor inputs."""

    def __init__(self, normalizer, device, obs_horizon=OBS_HORIZON):
        self.normalizer = normalizer
        self.device = device
        self.obs_horizon = obs_horizon
        self.history = deque(maxlen=obs_horizon)

    def _normalize(self, key, arr):
        if self.normalizer is not None:
            return self.normalizer.normalize(key, arr)
        return arr

    def _capture_image(self, cap, size):
        if cap is None or not cap.isOpened():
            return np.zeros((size[0], size[1], 3), dtype=np.uint8)
        ret, frame = cap.read()
        if not ret:
            return np.zeros((size[0], size[1], 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        return frame

    def update(self, robot_obs, wrist_cap, front_cap=None):
        """Add new observation frame to history."""
        wrist_img = self._capture_image(wrist_cap, IMAGE_SIZE)
        crop_img = self._capture_image(front_cap, CROP_SIZE) if front_cap else np.zeros((*CROP_SIZE, 3), dtype=np.uint8)

        frame = {
            "image_wrist": torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0,
            "image_crop": torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0,
            "qpos": self._normalize("qpos", robot_obs["qpos"]),
            "qvel": self._normalize("qvel", robot_obs["qvel"]),
            "ee_pose": self._normalize("ee_pose", robot_obs["ee_pose"]),
            "gripper": self._normalize("gripper", np.array([
                robot_obs["gripper_pos"][0], robot_obs["gripper_vel"][0]
            ])),
            "current": self._normalize("current", robot_obs["current"]),
            "pwm": self._normalize("pwm", robot_obs["pwm"]),
        }
        self.history.append(frame)

    def build_batch(self):
        """Build batched tensor dict [1, T, ...] from history."""
        # Pad history if not enough frames
        while len(self.history) < self.obs_horizon:
            self.history.appendleft(self.history[0])

        frames = list(self.history)

        def stack_key(key):
            tensors = [f[key] if isinstance(f[key], torch.Tensor)
                       else torch.from_numpy(np.array(f[key])).float()
                       for f in frames]
            return torch.stack(tensors).unsqueeze(0).to(self.device)  # [1, T, ...]

        return {
            "obs_image_wrist": stack_key("image_wrist"),
            "obs_image_crop": stack_key("image_crop"),
            "obs_qpos": stack_key("qpos"),
            "obs_qvel": stack_key("qvel"),
            "obs_ee_pose": stack_key("ee_pose"),
            "obs_gripper": stack_key("gripper"),
            "obs_current": stack_key("current"),
            "obs_pwm": stack_key("pwm"),
        }


def main():
    parser = argparse.ArgumentParser(description="Deploy v8 policy on real robot")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--stats", default="data/interim/stats.json")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max inference steps before auto-stop")
    parser.add_argument("--no-torque", action="store_true",
                        help="Observe only, do not send commands")
    parser.add_argument("--open-loop", action="store_true",
                        help="Bypass state machine, directly execute diffusion actions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load model ----
    logger.info(f"Loading model from {args.checkpoint}")
    model = HybridPolicyV8(pred_horizon=12, action_dim=6).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ---- Noise scheduler + DDIM ----
    scheduler = NoiseScheduler(num_steps=100).to(device)

    # ---- Normalizer ----
    normalizer = None
    if Path(args.stats).exists():
        normalizer = Normalizer.from_json(args.stats)
        logger.info(f"Loaded stats from {args.stats}")

    # ---- Robot ----
    logger.info(f"Connecting to robot at {args.port}")
    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        logger.error("Failed to connect to robot")
        sys.exit(1)

    safety = SafetyGuard(dxl_client=dxl)
    robot = RobotInterface(dxl, safety_guard=safety)

    # ---- Enable torque (unless observe-only) ----
    if not args.no_torque:
        robot.enable_torque()

    # ---- Policy runner ----
    cfg = InferenceConfig()
    runner = PolicyRunnerV8(model, robot, cfg, safety_guard=safety)
    runner.set_noise_scheduler(scheduler, ddim_steps=args.ddim_steps)
    if normalizer:
        runner.set_normalizer(normalizer)

    # ---- Cameras ----
    logger.info("Opening cameras...")
    wrist_cap = cv2.VideoCapture(WRIST_CAM_DEV)
    front_cap = cv2.VideoCapture(FRONT_CAM_DEV)
    if not wrist_cap.isOpened():
        logger.warning(f"Wrist camera not available at {WRIST_CAM_DEV}")
    if not front_cap.isOpened():
        logger.warning(f"Front camera not available at {FRONT_CAM_DEV}")

    # ---- Observation builder ----
    obs_builder = ObservationBuilder(normalizer, device)

    # ---- Graceful shutdown ----
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        logger.info("Shutdown requested (Ctrl+C)")
    signal.signal(signal.SIGINT, signal_handler)

    # ---- Main loop ----
    print(f"\n{'='*60}")
    print(f"ASYNC DP V8 - POLICY ROLLOUT")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"DDIM steps: {args.ddim_steps}")
    print(f"Max steps: {args.max_steps}")
    mode_str = "OPEN-LOOP" if args.open_loop else "STATE-MACHINE"
    print(f"Torque: {'OFF (observe only)' if args.no_torque else 'ON'}")
    print(f"Mode: {mode_str}")
    print(f"{'='*60}")
    print(f"Press Ctrl+C to stop\n")

    step = 0
    prev_state = None
    loop_times = deque(maxlen=50)

    # Open-loop state
    ol_chunk = None
    ol_chunk_step = 0

    try:
        while running and step < args.max_steps:
            t_loop = time.monotonic()

            # 1. Read robot state
            robot_obs = robot.get_observation()

            # 2. Capture cameras + build observation batch
            obs_builder.update(robot_obs, wrist_cap, front_cap)
            obs_batch = obs_builder.build_batch()

            if args.open_loop:
                # ---- Open-loop: bypass state machine, run diffusion directly ----
                t_infer = time.monotonic()

                # Replan every execute_horizon steps
                if ol_chunk is None or ol_chunk_step >= cfg.execute_horizon:
                    B = 1
                    H = cfg.pred_horizon
                    A = 6
                    device_t = obs_batch["obs_qpos"].device

                    ddim = DDIMSampler(
                        scheduler.alphas_cumprod,
                        num_train_steps=100,
                        num_inference_steps=args.ddim_steps,
                    ).to(device)

                    arm_chunk = ddim.sample(model, obs_batch, shape=(B, H, A), device=device_t)
                    arm_np = arm_chunk[0].cpu().numpy()

                    # Denormalize
                    if normalizer:
                        arm_np = normalizer.denormalize("action_arm", arm_np)

                    ol_chunk = arm_np
                    ol_chunk_step = 0

                infer_ms = (time.monotonic() - t_infer) * 1000

                # Get current action
                action = ol_chunk[ol_chunk_step]
                ol_chunk_step += 1

                # Safety clamp
                current_qpos = robot_obs["qpos"]
                from async_dp_v8.control.action_postprocess import clip_joint_step
                action = clip_joint_step(
                    action.reshape(1, -1), current_qpos,
                    max_step_rad=cfg.max_joint_step_rad,
                )[0]

                # Send command
                if not args.no_torque:
                    robot.send_command({"arm_actions": action, "gripper": "hold"})

                # Log
                loop_ms = (time.monotonic() - t_loop) * 1000
                loop_times.append(loop_ms)

                if step % 15 == 0:
                    avg_loop = np.mean(loop_times) if loop_times else 0
                    hz = 1000 / avg_loop if avg_loop > 0 else 0
                    logger.info(
                        f"step={step:4d} | OPEN-LOOP | "
                        f"chunk={ol_chunk_step}/{cfg.execute_horizon} | "
                        f"infer={infer_ms:5.0f}ms | loop={avg_loop:5.1f}ms ({hz:.0f}Hz) | "
                        f"qpos={current_qpos.round(3)} | action={action.round(3)}"
                    )

            else:
                # ---- State machine mode ----
                t_infer = time.monotonic()
                cmd, info = runner.step(obs_batch)
                infer_ms = (time.monotonic() - t_infer) * 1000

                state = info.get("state", "?")
                if state != prev_state:
                    logger.info(f"State: {prev_state} -> {state}")
                    prev_state = state

                if not args.no_torque and cmd.get("arm_actions") is not None:
                    robot.send_command(cmd)

                loop_ms = (time.monotonic() - t_loop) * 1000
                loop_times.append(loop_ms)

                if step % 15 == 0:
                    avg_loop = np.mean(loop_times) if loop_times else 0
                    hz = 1000 / avg_loop if avg_loop > 0 else 0
                    contact_p = info.get("contact_prob", 0)
                    phase_prob = info.get("phase_prob", [])
                    phase_str = ""
                    if len(phase_prob) > 0:
                        top_phase = int(np.argmax(phase_prob))
                        phase_names = {0: "REACH", 1: "ALIGN", 2: "CLOSE", 3: "LIFT", 4: "PLACE", 5: "RETURN"}
                        phase_str = f"phase={phase_names.get(top_phase, '?')}({phase_prob[top_phase]:.2f})"

                    logger.info(
                        f"step={step:4d} | state={state:12s} | "
                        f"infer={infer_ms:5.0f}ms | loop={avg_loop:5.1f}ms ({hz:.0f}Hz) | "
                        f"contact={contact_p:.2f} | {phase_str}"
                    )

                if state in ("DONE", "E_STOP"):
                    logger.info(f"Exiting: state={state}")
                    break

            step += 1

    finally:
        # Cleanup
        logger.info(f"Stopping after {step} steps")
        if not args.no_torque:
            robot.disable_torque()
        wrist_cap.release()
        front_cap.release()
        dxl.disconnect()
        logger.info("Cleanup complete")

    print(f"\n{'='*60}")
    print(f"ROLLOUT COMPLETE: {step} steps, final_state={prev_state or mode_str}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
