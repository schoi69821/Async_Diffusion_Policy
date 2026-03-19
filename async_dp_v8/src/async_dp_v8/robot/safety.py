"""Safety guards for robot operation."""
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SafetyGuard:
    def __init__(
        self,
        dxl_client=None,
        max_joint_step_rad: float = 0.12,
        max_joint_vel_rad_s: float = 2.0,
        max_current_ma: float = 500.0,
        voltage_min: float = 10.0,
        voltage_max: float = 14.0,
        timeout_steps: int = 300,
    ):
        self.dxl = dxl_client
        self.max_joint_step_rad = max_joint_step_rad
        self.max_joint_vel_rad_s = max_joint_vel_rad_s
        self.max_current_ma = max_current_ma
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.timeout_steps = timeout_steps
        self._e_stop = False

    def check_command(
        self,
        target: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[bool, str]:
        """Validate command before sending.

        Returns: (is_safe, reason)
        """
        if self._e_stop:
            return False, "E-STOP active"

        delta = np.abs(target - current)
        if np.max(delta) > self.max_joint_step_rad:
            return False, f"Joint step {np.max(delta):.3f} exceeds limit {self.max_joint_step_rad}"

        return True, "OK"

    def clamp_command(
        self,
        target: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        """Clamp command to safe joint step limits.

        Returns clamped target positions.
        """
        delta = target - current
        delta = np.clip(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
        return current + delta

    def check_and_clamp_command(
        self,
        target: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[np.ndarray, bool, str]:
        """Check command safety, clamp if needed.

        Returns: (clamped_target, was_clamped, reason)
        """
        if self._e_stop:
            return current, True, "E-STOP active - holding position"

        delta = target - current
        max_delta = np.max(np.abs(delta))

        if max_delta > self.max_joint_step_rad:
            clamped = self.clamp_command(target, current)
            logger.warning(f"Command clamped: max_delta={max_delta:.3f} > limit={self.max_joint_step_rad}")
            return clamped, True, f"Clamped from {max_delta:.3f}"

        return target, False, "OK"

    def check_state(
        self,
        velocity: np.ndarray,
        current: np.ndarray,
        voltage: float,
    ) -> Tuple[bool, str]:
        """Check robot state for safety violations.

        Returns: (is_safe, reason)
        """
        if self._e_stop:
            return False, "E-STOP active"

        if np.max(np.abs(velocity)) > self.max_joint_vel_rad_s:
            return False, f"Velocity {np.max(np.abs(velocity)):.2f} exceeds limit"

        if np.max(np.abs(current)) > self.max_current_ma:
            return False, f"Current {np.max(np.abs(current)):.0f}mA exceeds limit"

        if voltage < self.voltage_min or voltage > self.voltage_max:
            return False, f"Voltage {voltage:.1f}V out of range [{self.voltage_min}, {self.voltage_max}]"

        return True, "OK"

    def trigger_estop(self, reason: str = ""):
        """Emergency stop: disable all motor torques immediately."""
        logger.critical(f"E-STOP triggered: {reason}")
        self._e_stop = True
        if self.dxl is not None:
            try:
                self.dxl.torque_off_all()
                logger.critical("All motor torques disabled")
            except Exception as e:
                logger.critical(f"Failed to disable torques during E-STOP: {e}")

    def reset_estop(self):
        logger.info("E-STOP reset")
        self._e_stop = False

    @property
    def is_estopped(self) -> bool:
        return self._e_stop
