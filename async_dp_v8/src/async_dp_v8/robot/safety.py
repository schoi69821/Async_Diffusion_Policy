"""Safety guards for robot operation."""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SafetyGuard:
    def __init__(
        self,
        max_joint_step_rad: float = 0.12,
        max_joint_vel_rad_s: float = 2.0,
        max_current_ma: float = 500.0,
        voltage_min: float = 10.0,
        voltage_max: float = 14.0,
        watchdog_timeout_ms: int = 1000,
    ):
        self.max_joint_step_rad = max_joint_step_rad
        self.max_joint_vel_rad_s = max_joint_vel_rad_s
        self.max_current_ma = max_current_ma
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.watchdog_timeout_ms = watchdog_timeout_ms
        self._e_stop = False

    def check_command(
        self,
        target: np.ndarray,
        current: np.ndarray,
    ) -> tuple:
        """Validate command before sending.

        Returns: (is_safe, reason)
        """
        if self._e_stop:
            return False, "E-STOP active"

        delta = np.abs(target - current)
        if np.max(delta) > self.max_joint_step_rad:
            return False, f"Joint step {np.max(delta):.3f} exceeds limit {self.max_joint_step_rad}"

        return True, "OK"

    def check_state(
        self,
        velocity: np.ndarray,
        current: np.ndarray,
        voltage: float,
    ) -> tuple:
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
        logger.critical(f"E-STOP triggered: {reason}")
        self._e_stop = True

    def reset_estop(self):
        logger.info("E-STOP reset")
        self._e_stop = False

    @property
    def is_estopped(self) -> bool:
        return self._e_stop
