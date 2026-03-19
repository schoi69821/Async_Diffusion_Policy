"""Contact event labeling from gripper current/velocity signals."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ContactLabelConfig:
    current_thresh: float = 35.0
    velocity_thresh: float = -0.05
    position_closed_thresh: float = -1.05
    soft_blend_window: int = 5
    hard_thresh: float = 0.65


class ContactRelabeler:
    def __init__(self, cfg: ContactLabelConfig):
        self.cfg = cfg

    def label_episode(self, df: pd.DataFrame) -> pd.DataFrame:
        cur = df["gripper_current"].to_numpy()
        vel = df["gripper_vel"].to_numpy()
        pos = df["gripper_pos_rad"].to_numpy()
        dcur = np.diff(cur, prepend=cur[0])

        # Soft contact: blend of current rise and velocity stall
        current_score = np.clip((dcur - self.cfg.current_thresh) / 50.0, 0, 1)
        vel_score = np.clip((-vel - 0.05) / 0.20, 0, 1)
        contact_soft = 0.55 * current_score + 0.45 * vel_score
        contact_soft = np.clip(contact_soft, 0.0, 1.0)

        # Smooth with rolling window
        if self.cfg.soft_blend_window > 1:
            kernel = np.ones(self.cfg.soft_blend_window) / self.cfg.soft_blend_window
            contact_soft = np.convolve(contact_soft, kernel, mode="same")
            contact_soft = np.clip(contact_soft, 0.0, 1.0)

        contact_hard = (contact_soft > self.cfg.hard_thresh).astype(np.int64)

        df = df.copy()
        df["contact_soft"] = contact_soft
        df["contact_hard"] = contact_hard
        return df
