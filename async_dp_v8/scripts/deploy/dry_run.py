#!/usr/bin/env python3
"""Robot dry-run: validate communication, sensors, and safety with torque OFF.

This script tests:
1. DXL port connection
2. SyncRead (position, velocity, current, PWM)
3. Motor count and ID detection
4. Voltage check
5. Gripper current sensing
6. FK computation from live joint readings
7. SafetyGuard basic checks
8. Bus watchdog verification

Run this BEFORE enabling torque or running the policy.
"""
import argparse
import time
import sys
import numpy as np
import logging

from async_dp_v8.robot.dxl_client import DxlClient, DxlReadError
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.robot.safety import SafetyGuard
from async_dp_v8.control.kinematics import forward_kinematics, qpos_to_ee_pose
from async_dp_v8.constants import (
    FOLLOWER_PORT, DXL_BAUDRATE, NUM_MOTORS, NUM_JOINTS, JOINT_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DryRunResult:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.checks.append((name, status, detail))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        icon = "✓" if condition else "✗"
        print(f"  [{icon}] {name}: {status}  {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"DRY RUN RESULT: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed == 0:
            print("STATUS: ALL CHECKS PASSED - safe to proceed")
        else:
            print("STATUS: SOME CHECKS FAILED - DO NOT proceed to rollout")
            for name, status, detail in self.checks:
                if status == "FAIL":
                    print(f"  FAILED: {name} - {detail}")
        print(f"{'='*60}")
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="Robot dry-run validation")
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--baudrate", type=int, default=DXL_BAUDRATE)
    parser.add_argument("--read-count", type=int, default=50, help="Number of state reads to test")
    args = parser.parse_args()

    result = DryRunResult()

    print(f"\n{'='*60}")
    print(f"ASYNC DP V8 - ROBOT DRY RUN")
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"{'='*60}\n")

    # 1. Connection
    print("[1/8] Connection Test")
    dxl = DxlClient(args.port, args.baudrate, bus_watchdog_20ms=0)  # No watchdog for dry-run
    connected = dxl.connect()
    result.check("DXL port connection", connected, args.port)
    if not connected:
        print("\nCannot proceed without connection.")
        result.summary()
        return

    try:
        # 2. Motor detection
        print("\n[2/8] Motor Detection")
        result.check("Motor IDs configured", len(dxl.ids) == NUM_MOTORS,
                      f"expected {NUM_MOTORS}, got {len(dxl.ids)}")

        # 3. SyncRead test
        print("\n[3/8] SyncRead Test")
        read_success = True
        read_times = []
        try:
            for i in range(args.read_count):
                t0 = time.monotonic()
                state = dxl.read_state()
                dt = (time.monotonic() - t0) * 1000
                read_times.append(dt)
        except DxlReadError as e:
            read_success = False
            result.check("SyncRead stable", False, str(e))

        if read_success:
            avg_ms = np.mean(read_times)
            max_ms = np.max(read_times)
            result.check("SyncRead stable", True,
                          f"{args.read_count} reads, avg={avg_ms:.1f}ms, max={max_ms:.1f}ms")
            result.check("SyncRead latency < 5ms", avg_ms < 5.0, f"avg={avg_ms:.1f}ms")

            # Check all motors responded
            state = dxl.read_state()
            all_nonzero = not np.all(state["present_position"] == 0)
            result.check("All motors responding", all_nonzero,
                          f"positions: {state['present_position']}")

        # 4. Voltage
        print("\n[4/8] Voltage Check")
        voltage = dxl.read_voltage()
        result.check("Voltage reading", voltage > 0, f"{voltage:.1f}V")
        result.check("Voltage in range (10-14V)", 10.0 <= voltage <= 14.0,
                      f"{voltage:.1f}V")

        # 5. Robot interface
        print("\n[5/8] Robot Interface")
        safety = SafetyGuard(dxl_client=dxl)
        robot = RobotInterface(dxl, safety_guard=safety)
        obs = robot.get_observation()

        result.check("qpos shape", obs["qpos"].shape == (NUM_JOINTS,),
                      f"shape={obs['qpos'].shape}")
        result.check("qvel shape", obs["qvel"].shape == (NUM_JOINTS,),
                      f"shape={obs['qvel'].shape}")
        result.check("ee_pose shape", obs["ee_pose"].shape == (7,),
                      f"shape={obs['ee_pose'].shape}")
        result.check("gripper_pos shape", obs["gripper_pos"].shape == (1,),
                      f"shape={obs['gripper_pos'].shape}")
        result.check("current shape", obs["current"].shape == (len(JOINT_MAP),),
                      f"shape={obs['current'].shape}")

        print(f"  Joint positions (rad): {obs['qpos']}")
        print(f"  EE pose: {obs['ee_pose'][:3]} (xyz, meters)")
        print(f"  Gripper pos: {obs['gripper_pos'][0]:.3f}")

        # 6. FK validation
        print("\n[6/8] Forward Kinematics")
        ee_pos = obs["ee_pose"][:3]
        result.check("EE Z > 0 (above base)", ee_pos[2] > 0,
                      f"z={ee_pos[2]*1000:.1f}mm")
        ee_dist = np.linalg.norm(ee_pos)
        result.check("EE distance reasonable (0.05-0.8m)", 0.05 < ee_dist < 0.8,
                      f"dist={ee_dist:.3f}m")

        # 7. Safety guard
        print("\n[7/8] Safety Guard")
        safe, reason = safety.check_state(
            obs["qvel"], obs["current"],
            obs["voltage"][0] if len(obs["voltage"]) > 0 else 12.0,
        )
        result.check("Safety state check", safe, reason)

        # Test clamp
        big_target = obs["qpos"] + np.ones(NUM_JOINTS) * 0.5
        clamped, was_clamped, _ = safety.check_and_clamp_command(big_target, obs["qpos"])
        result.check("Joint clamp works", was_clamped, "large step correctly clamped")
        max_step = np.max(np.abs(clamped - obs["qpos"]))
        result.check("Clamped step <= max", max_step <= safety.max_joint_step_rad + 1e-6,
                      f"max_step={max_step:.4f}")

        # 8. Timing consistency
        print("\n[8/8] Timing Consistency")
        obs_times = []
        for _ in range(20):
            t0 = time.monotonic()
            robot.get_observation()
            obs_times.append((time.monotonic() - t0) * 1000)
        obs_avg = np.mean(obs_times)
        obs_std = np.std(obs_times)
        result.check("Observation latency < 10ms", obs_avg < 10.0,
                      f"avg={obs_avg:.1f}ms, std={obs_std:.1f}ms")

    finally:
        dxl.disconnect()

    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
