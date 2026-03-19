"""Safety guards for inference-time decisions."""
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class InferenceGuard:
    def __init__(
        self,
        max_joint_step: float = 0.12,
        max_ee_vel: float = 0.5,
        timeout_steps: int = 300,
    ):
        self.max_joint_step = max_joint_step
        self.max_ee_vel = max_ee_vel
        self.timeout_steps = timeout_steps

    def check_action(
        self,
        action: np.ndarray,
        current_qpos: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Validate and clip action. Returns (clipped_action, was_clipped)."""
        delta = action - current_qpos
        max_delta = np.max(np.abs(delta))

        if max_delta > self.max_joint_step:
            scale = self.max_joint_step / max_delta
            clipped = current_qpos + delta * scale
            logger.warning(f"Action clipped: max_delta={max_delta:.3f}")
            return clipped, True

        return action, False

    def check_timeout(self, steps_in_state: int, state_name: str) -> bool:
        if steps_in_state > self.timeout_steps:
            logger.warning(f"Timeout in state {state_name} after {steps_in_state} steps")
            return True
        return False
