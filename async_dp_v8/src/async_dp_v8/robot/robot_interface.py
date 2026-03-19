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

    def __init__(self, dxl_client: DxlClient, home_qpos: Optional[np.ndarray] = None):
        self.dxl = dxl_client
        self.home_qpos = home_qpos if home_qpos is not None else np.zeros(NUM_JOINTS)
        self._last_state: Dict[str, np.ndarray] = {}

    def get_observation(self) -> Dict[str, np.ndarray]:
        raw = self.dxl.read_state()
        voltage = self.dxl.read_voltage()

        # Convert raw ticks to radians for joints
        pos_rad = (raw["present_position"] - self.CENTER_TICK) / self.TICKS_PER_RAD
        vel_rad = raw["present_velocity"] * 0.229 * (2 * np.pi / 60)  # rev/min to rad/s

        # Split arm and gripper
        qpos = self._motors_to_joints(pos_rad[:NUM_MOTORS])
        qvel = self._motors_to_joints(vel_rad[:NUM_MOTORS])

        # Gripper
        gripper_pos = pos_rad[NUM_MOTORS - 1] if len(pos_rad) >= NUM_MOTORS else 0.0
        gripper_vel = vel_rad[NUM_MOTORS - 1] if len(vel_rad) >= NUM_MOTORS else 0.0

        # EE pose from FK
        ee_pose = qpos_to_ee_pose(qpos[:NUM_JOINTS])

        obs = {
            "qpos": qpos[:NUM_JOINTS],
            "qvel": qvel[:NUM_JOINTS],
            "ee_pose": ee_pose,
            "gripper_pos": np.array([gripper_pos]),
            "gripper_vel": np.array([gripper_vel]),
            "current": raw["present_current"][:NUM_MOTORS],
            "pwm": raw["present_pwm"][:NUM_MOTORS],
            "voltage": np.array([voltage]),
        }
        self._last_state = obs
        return obs

    def send_command(self, cmd: dict):
        if cmd.get("arm_actions") is not None:
            actions = cmd["arm_actions"]
            for t in range(len(actions)):
                joints = actions[t]
                motor_positions = self._joints_to_motors(joints)
                for i, motor_id in enumerate(self.dxl.ids[:len(motor_positions)]):
                    self.dxl.goal_position_rad(motor_id, motor_positions[i])

        gripper = cmd.get("gripper", "hold")
        if gripper == "close":
            self.close_gripper()
        elif gripper == "open":
            self.open_gripper()

    def close_gripper(self):
        gripper_id = self.dxl.ids[-1]  # Last motor is gripper
        self.dxl.goal_position(gripper_id, FOLLOWER_GRIP_MIN)

    def open_gripper(self):
        gripper_id = self.dxl.ids[-1]
        self.dxl.goal_position(gripper_id, FOLLOWER_GRIP_MAX)

    def gripper_contact_confirmed(self) -> bool:
        if not self._last_state:
            return False
        current = self._last_state.get("current", np.zeros(NUM_MOTORS))
        gripper_current = abs(current[-1]) if len(current) >= NUM_MOTORS else 0
        return gripper_current > 80  # mA threshold

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

    def _motors_to_joints(self, motor_vals: np.ndarray) -> np.ndarray:
        """Convert motor-level values to joint-level using JOINT_MAP."""
        joints = np.zeros(len(JOINT_MAP))
        idx = 0
        for j, motor_ids in enumerate(JOINT_MAP):
            if idx < len(motor_vals):
                joints[j] = motor_vals[idx]
            idx += len(motor_ids)
        return joints

    def _joints_to_motors(self, joint_vals: np.ndarray) -> np.ndarray:
        """Convert joint-level values to motor-level using JOINT_MAP."""
        motors = []
        for j, motor_ids in enumerate(JOINT_MAP):
            val = joint_vals[j] if j < len(joint_vals) else 0.0
            for _ in motor_ids:
                motors.append(val)
        return np.array(motors)
