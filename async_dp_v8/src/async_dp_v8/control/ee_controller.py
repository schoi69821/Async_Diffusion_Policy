"""End-effector level controller for waypoint tracking."""
import numpy as np
from typing import Optional


class EEController:
    def __init__(self, max_vel: float = 0.5, dt: float = 1.0 / 15.0):
        self.max_vel = max_vel
        self.dt = dt

    def compute_joint_targets(
        self,
        ee_actions: np.ndarray,
        current_qpos: np.ndarray,
    ) -> np.ndarray:
        """Convert EE-space actions to joint targets (simple pass-through for now).

        In production, this would use IK. For now, we assume actions are in joint space.
        """
        return ee_actions
