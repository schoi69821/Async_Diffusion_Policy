"""High-level robot interface for observation and command."""
import numpy as np
import logging
from typing import Dict, Optional

from .dxl_client import DxlClient
from async_dp_v8.constants import (
    JOINT_MAP, NUM_JOINTS, NUM_MOTORS,
    FOLLOWER_GRIP_MIN, FOLLOWER_GRIP_MAX,
)
from async_dp_v8.control.kinematics import qpos_to_ee_pose

logger = logging.getLogger(__name__)


class RobotInterface:
    TICKS_PER_RAD = 4096 / (2 * np.pi)
    CENTER_TICK = 2048

    def __init__(self, dxl_client: DxlClient, home_qpos: Optional[np.ndarray] = None,
                 safety_guard=None, contact_threshold_ma: float = 80.0):
        self.dxl = dxl_client
        self.home_qpos = home_qpos if home_qpos is not None else np.zeros(NUM_JOINTS)
        self.safety = safety_guard
        self.contact_threshold_ma = contact_threshold_ma
        self._last_state: Dict[str, np.ndarray] = {}

    def get_observation(self) -> Dict[str, np.ndarray]:
        raw = self.dxl.read_state()
        voltage = self.dxl.read_voltage()

        # Convert raw ticks to radians
        pos_rad = (raw["present_position"] - self.CENTER_TICK) / self.TICKS_PER_RAD
        vel_rad = raw["present_velocity"] * 0.229 * (2 * np.pi / 60)

        # Map 9 motors to 7 logical joints (avg dual-motor joints)
        all_joints_pos = self._motors_to_joints(pos_rad)
        all_joints_vel = self._motors_to_joints(vel_rad)

        # Arm = first 6 joints, gripper = 7th joint
        qpos = all_joints_pos[:NUM_JOINTS]
        qvel = all_joints_vel[:NUM_JOINTS]
        gripper_pos = all_joints_pos[NUM_JOINTS] if len(all_joints_pos) > NUM_JOINTS else 0.0
        gripper_vel = all_joints_vel[NUM_JOINTS] if len(all_joints_vel) > NUM_JOINTS else 0.0

        # EE pose from FK
        ee_pose = qpos_to_ee_pose(qpos)

        # Current and PWM - keep all 9 motor values for contact detection
        current_joints = self._motors_to_joints(raw["present_current"])
        pwm_joints = self._motors_to_joints(raw["present_pwm"])

        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pose": ee_pose,
            "gripper_pos": np.array([gripper_pos]),
            "gripper_vel": np.array([gripper_vel]),
            "current": current_joints,  # 7 logical joints
            "pwm": pwm_joints,          # 7 logical joints
            "voltage": np.array([voltage]),
        }
        self._last_state = obs
        return obs

    def send_command(self, cmd: dict):
        """Send a single-step command to the robot.

        For arm_actions, only the FIRST action in the array is sent per call.
        The caller (control loop) should call this at control_freq.
        """
        if self.safety and self.safety.is_estopped:
            return

        if cmd.get("arm_actions") is not None:
            actions = cmd["arm_actions"]
            # Take only the first action step
            if len(actions.shape) > 1:
                target_joints = actions[0]
            else:
                target_joints = actions

            # Safety check and clamp
            current_qpos = self._last_state.get("qpos", np.zeros(NUM_JOINTS))
            if self.safety:
                target_joints, was_clamped, reason = self.safety.check_and_clamp_command(
                    target_joints[:NUM_JOINTS], current_qpos
                )

            # Convert joints to motor positions and send
            motor_positions = self._joints_to_motors(target_joints)
            for i, motor_id in enumerate(self.dxl.ids[:len(motor_positions)]):
                self.dxl.goal_position_rad(motor_id, motor_positions[i])

        # Handle gripper
        gripper = cmd.get("gripper", "hold")
        if gripper == "close":
            self.close_gripper()
        elif gripper == "open":
            self.open_gripper()

        # Handle primitives
        if cmd.get("primitive") == "lift_z":
            self._execute_lift_primitive(cmd.get("delta_z_mm", 20.0))

    def close_gripper(self):
        gripper_id = self.dxl.ids[-1]  # Last motor is gripper
        self.dxl.goal_position(gripper_id, FOLLOWER_GRIP_MIN)

    def open_gripper(self):
        gripper_id = self.dxl.ids[-1]
        self.dxl.goal_position(gripper_id, FOLLOWER_GRIP_MAX)

    def gripper_contact_confirmed(self, threshold_ma: float = None) -> bool:
        if threshold_ma is None:
            threshold_ma = self.contact_threshold_ma
        if not self._last_state:
            return False
        current = self._last_state.get("current", np.zeros(len(JOINT_MAP)))
        gripper_idx = len(JOINT_MAP) - 1
        gripper_current = abs(current[gripper_idx]) if len(current) > gripper_idx else 0
        return gripper_current > threshold_ma

    def is_at_home(self, thresh_rad: float = 0.05) -> bool:
        if not self._last_state:
            return False
        qpos = self._last_state.get("qpos", np.zeros(NUM_JOINTS))
        return np.max(np.abs(qpos - self.home_qpos)) < thresh_rad

    def ee_z_above(self, z_mm: float) -> bool:
        if not self._last_state:
            return False
        ee_pose = self._last_state.get("ee_pose", np.zeros(7))
        return ee_pose[2] * 1000 > z_mm  # Convert m to mm

    def safe_go_home(self, steps: int = 100, dt: float = 0.02):
        """Safely interpolate to home position."""
        import time
        obs = self.get_observation()
        current = obs["qpos"]

        for i in range(steps):
            t = (i + 1) / steps
            target = current + t * (self.home_qpos - current)

            if self.safety:
                target, _, _ = self.safety.check_and_clamp_command(target, current)

            motor_positions = self._joints_to_motors(target)
            for j, motor_id in enumerate(self.dxl.ids[:len(motor_positions)]):
                self.dxl.goal_position_rad(motor_id, motor_positions[j])

            current = target
            time.sleep(dt)

        self.open_gripper()

    def _execute_lift_primitive(self, delta_z_mm: float, steps: int = 30, dt: float = 0.02):
        """Lift the end-effector by delta_z_mm using differential IK.

        Breaks the motion into small steps for safety.
        """
        import time
        from async_dp_v8.control.kinematics import ik_delta_z

        obs = self.get_observation()
        current_qpos = obs["qpos"].copy()

        # Convert mm to meters, divide into steps
        delta_z_m = delta_z_mm / 1000.0
        step_z = delta_z_m / steps

        for i in range(steps):
            # Compute joint delta for this small Z step
            dq = ik_delta_z(current_qpos, step_z)
            target = current_qpos + dq

            # Safety clamp
            if self.safety:
                target, _, _ = self.safety.check_and_clamp_command(target, current_qpos)

            # Send to motors
            motor_positions = self._joints_to_motors(target)
            for j, motor_id in enumerate(self.dxl.ids[:len(motor_positions)]):
                self.dxl.goal_position_rad(motor_id, motor_positions[j])

            current_qpos = target
            time.sleep(dt)

        logger.info(f"Lift primitive completed: {delta_z_mm}mm in {steps} steps")

    def _motors_to_joints(self, motor_vals: np.ndarray) -> np.ndarray:
        """Convert motor-level values to joint-level using JOINT_MAP.

        Dual-motor joints are averaged.
        """
        joints = np.zeros(len(JOINT_MAP))
        idx = 0
        for j, motor_ids in enumerate(JOINT_MAP):
            n_motors = len(motor_ids)
            if idx + n_motors <= len(motor_vals):
                joints[j] = np.mean(motor_vals[idx:idx + n_motors])
            elif idx < len(motor_vals):
                joints[j] = motor_vals[idx]
            idx += n_motors
        return joints

    def _joints_to_motors(self, joint_vals: np.ndarray) -> np.ndarray:
        """Convert joint-level values to motor-level using JOINT_MAP.

        Dual-motor joints get the same value for both motors.
        """
        motors = []
        for j, motor_ids in enumerate(JOINT_MAP):
            val = joint_vals[j] if j < len(joint_vals) else 0.0
            for _ in motor_ids:
                motors.append(val)
        return np.array(motors)
