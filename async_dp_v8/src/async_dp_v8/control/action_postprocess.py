"""Post-processing for raw action outputs."""
import numpy as np
from typing import Optional


def clip_joint_step(
    actions: np.ndarray,
    current_pos: np.ndarray,
    max_step_rad: float = 0.12,
) -> np.ndarray:
    """Clip actions so no single joint moves more than max_step_rad per step.

    actions: [H, A]
    current_pos: [A]
    """
    clipped = actions.copy()
    pos = current_pos.copy()
    for t in range(len(clipped)):
        delta = clipped[t] - pos
        delta = np.clip(delta, -max_step_rad, max_step_rad)
        clipped[t] = pos + delta
        pos = clipped[t]
    return clipped


def denormalize_actions(
    actions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Denormalize actions from training distribution."""
    return actions * std + mean


def ema_smooth(
    actions: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply EMA smoothing over action horizon."""
    smoothed = actions.copy()
    for t in range(1, len(smoothed)):
        smoothed[t] = alpha * smoothed[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed
