#!/usr/bin/env python3
"""Benchmark policy rollout on real robot.

Runs N rollouts, records success/failure metrics, saves logs and videos.
Supports variable conditions (lighting, pen position, start pose).

Usage:
    python rollout_benchmark.py --checkpoint best.pt --num-rollouts 20
    python rollout_benchmark.py --checkpoint best.pt --num-rollouts 5 --condition lighting_dim
"""
import argparse
import time
import json
import sys
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.robot.safety import SafetyGuard
from async_dp_v8.inference.policy_runner import PolicyRunnerV8
from async_dp_v8.inference.state_machine import State
from async_dp_v8.types import InferenceConfig
from async_dp_v8.utils.checkpointing import load_checkpoint
from async_dp_v8.utils.normalization import Normalizer
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE, NUM_JOINTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RolloutMetrics:
    rollout_id: int = 0
    condition: str = "default"
    start_time: str = ""
    duration_s: float = 0.0
    total_steps: int = 0
    final_state: str = ""

    # Success criteria
    reach_success: bool = False
    close_triggered: bool = False
    contact_confirmed: bool = False
    lift_success: bool = False
    place_success: bool = False
    return_success: bool = False
    end_to_end_success: bool = False

    # Safety
    e_stop_triggered: bool = False
    recovery_count: int = 0
    max_joint_step: float = 0.0

    # Quality
    jitter_rms: float = 0.0
    state_trace: List[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    checkpoint: str = ""
    timestamp: str = ""
    num_rollouts: int = 0
    condition: str = "default"
    rollouts: List[RolloutMetrics] = field(default_factory=list)

    def compute_summary(self) -> dict:
        n = max(len(self.rollouts), 1)
        return {
            "reach_success_rate": sum(r.reach_success for r in self.rollouts) / n,
            "close_recall": sum(r.close_triggered for r in self.rollouts) / n,
            "contact_confirmed_rate": sum(r.contact_confirmed for r in self.rollouts) / n,
            "lift_success_rate": sum(r.lift_success for r in self.rollouts) / n,
            "place_success_rate": sum(r.place_success for r in self.rollouts) / n,
            "return_success_rate": sum(r.return_success for r in self.rollouts) / n,
            "end_to_end_success_rate": sum(r.end_to_end_success for r in self.rollouts) / n,
            "e_stop_rate": sum(r.e_stop_triggered for r in self.rollouts) / n,
            "avg_jitter_rms": np.mean([r.jitter_rms for r in self.rollouts]),
            "avg_duration_s": np.mean([r.duration_s for r in self.rollouts]),
            "false_close_rate": sum(
                r.close_triggered and not r.contact_confirmed for r in self.rollouts
            ) / n,
        }


def run_single_rollout(
    runner: PolicyRunnerV8,
    robot: RobotInterface,
    rollout_id: int,
    condition: str,
    max_steps: int = 1000,
    device: str = "cpu",
) -> RolloutMetrics:
    """Execute a single rollout and collect metrics."""
    metrics = RolloutMetrics(
        rollout_id=rollout_id,
        condition=condition,
        start_time=datetime.now().isoformat(),
    )

    runner.reset()
    joint_history = []
    state_trace = []

    logger.info(f"Rollout {rollout_id} starting...")

    # Move to home first
    robot.safe_go_home()
    time.sleep(1.0)

    t_start = time.monotonic()

    for step in range(max_steps):
        try:
            obs = robot.get_observation()
            joint_history.append(obs["qpos"].copy())

            # Build observation batch (single-frame for now, would need history for real use)
            obs_batch = _obs_to_batch(obs, device)

            cmd, info = runner.step(obs_batch)
            state_name = info.get("state", "UNKNOWN")
            state_trace.append(state_name)

            # Track state transitions for success criteria
            if state_name == "REACH":
                metrics.reach_success = True
            elif state_name == "CLOSE_COMMIT":
                metrics.close_triggered = True
            elif state_name == "LIFT":
                metrics.contact_confirmed = True
            elif state_name == "PLACE":
                metrics.lift_success = True
            elif state_name == "RELEASE":
                metrics.place_success = True
            elif state_name == "DONE":
                metrics.return_success = True
                metrics.end_to_end_success = True
                break
            elif state_name == "E_STOP":
                metrics.e_stop_triggered = True
                break
            elif state_name == "RECOVERY":
                metrics.recovery_count += 1

            # Send command
            robot.send_command(cmd)

            # Control rate
            time.sleep(1.0 / 15)  # 15 Hz inference

        except Exception as e:
            logger.error(f"Rollout {rollout_id} error at step {step}: {e}")
            metrics.e_stop_triggered = True
            break

    metrics.duration_s = time.monotonic() - t_start
    metrics.total_steps = len(state_trace)
    metrics.final_state = state_trace[-1] if state_trace else "UNKNOWN"
    metrics.state_trace = state_trace

    # Compute jitter
    if len(joint_history) > 2:
        joints = np.array(joint_history)
        vel = np.diff(joints, axis=0)
        acc = np.diff(vel, axis=0)
        metrics.jitter_rms = float(np.sqrt(np.mean(acc ** 2)))
        metrics.max_joint_step = float(np.max(np.abs(vel)))

    # Return to home
    try:
        robot.safe_go_home()
    except Exception:
        pass

    status = "SUCCESS" if metrics.end_to_end_success else "FAIL"
    logger.info(
        f"Rollout {rollout_id} {status}: "
        f"final_state={metrics.final_state}, "
        f"steps={metrics.total_steps}, "
        f"duration={metrics.duration_s:.1f}s"
    )

    return metrics


def _obs_to_batch(obs: dict, device: str) -> dict:
    """Convert single observation to batched tensor format.

    NOTE: This creates a single-frame observation. For proper inference,
    you need to maintain an observation history buffer of obs_horizon frames.
    """
    T = 1  # Single frame (simplified)
    B = 1

    def to_t(arr, expected_dim=None):
        t = torch.from_numpy(np.array(arr)).float()
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        return t.to(device)

    return {
        "obs_image_wrist": torch.zeros(B, T, 3, 224, 224, device=device),
        "obs_image_crop": torch.zeros(B, T, 3, 96, 96, device=device),
        "obs_qpos": to_t(obs["qpos"]),
        "obs_qvel": to_t(obs["qvel"]),
        "obs_ee_pose": to_t(obs["ee_pose"]),
        "obs_gripper": to_t(np.concatenate([obs["gripper_pos"], obs["gripper_vel"]])),
        "obs_current": to_t(obs["current"]),
        "obs_pwm": to_t(obs["pwm"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Rollout benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--stats", default="data/interim/stats.json")
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--condition", default="default",
                        help="Test condition label (e.g., lighting_dim, pen_pos_2)")
    parser.add_argument("--output-dir", default="data/artifacts/rollout_logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = HybridPolicyV8().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Connect robot
    dxl = DxlClient(args.port, DXL_BAUDRATE)
    if not dxl.connect():
        logger.error("Failed to connect to robot")
        sys.exit(1)

    safety = SafetyGuard(dxl_client=dxl)
    robot = RobotInterface(dxl, safety_guard=safety)
    cfg = InferenceConfig()
    runner = PolicyRunnerV8(model, robot, cfg, safety_guard=safety)

    # Load normalizer
    if Path(args.stats).exists():
        normalizer = Normalizer.from_json(args.stats)
        runner.set_normalizer(normalizer)

    # Run benchmark
    report = BenchmarkReport(
        checkpoint=args.checkpoint,
        timestamp=datetime.now().isoformat(),
        num_rollouts=args.num_rollouts,
        condition=args.condition,
    )

    print(f"\n{'='*60}")
    print(f"ROLLOUT BENCHMARK: {args.num_rollouts} rollouts, condition={args.condition}")
    print(f"{'='*60}\n")

    for i in range(args.num_rollouts):
        metrics = run_single_rollout(
            runner, robot, i, args.condition,
            max_steps=args.max_steps, device=device,
        )
        report.rollouts.append(metrics)

        # Pause between rollouts for reset
        if i < args.num_rollouts - 1:
            input(f"\n>>> Reset pen position for rollout {i+1}, then press Enter...")

    # Summary
    summary = report.compute_summary()
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({args.num_rollouts} rollouts, {args.condition})")
    print(f"{'='*60}")
    for key, val in summary.items():
        if isinstance(val, float):
            if "rate" in key:
                print(f"  {key}: {val*100:.1f}%")
            else:
                print(f"  {key}: {val:.4f}")
    print(f"{'='*60}")

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"rollout_report_{args.condition}_{ts}.json"
    with open(report_path, "w") as f:
        json.dump({
            "checkpoint": report.checkpoint,
            "timestamp": report.timestamp,
            "num_rollouts": report.num_rollouts,
            "condition": report.condition,
            "summary": summary,
            "rollouts": [asdict(r) for r in report.rollouts],
        }, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    # Check pass criteria
    targets = {
        "reach_success_rate": 0.95,
        "close_recall": 0.95,
        "contact_confirmed_rate": 0.90,
        "lift_success_rate": 0.85,
        "e_stop_rate": 0.0,  # Should be 0
    }

    print(f"\nPASS CRITERIA CHECK:")
    all_pass = True
    for metric, target in targets.items():
        actual = summary[metric]
        if metric == "e_stop_rate":
            passed = actual == target
        else:
            passed = actual >= target
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {metric}: {actual*100:.1f}% (target: {target*100:.0f}%)")
        if not passed:
            all_pass = False

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    dxl.disconnect()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
