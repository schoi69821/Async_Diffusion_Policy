"""Phase relabeling from demonstration data."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from async_dp_v8.constants import (
    PHASE_REACH, PHASE_ALIGN, PHASE_CLOSE,
    PHASE_LIFT, PHASE_PLACE, PHASE_RETURN,
)


@dataclass
class PhaseLabelConfig:
    align_xy_thresh_mm: float = 12.0
    pregrasp_z_margin_mm: float = 25.0
    close_vel_thresh: float = -0.15
    current_rise_thresh: float = 35.0
    lift_dz_thresh_mm: float = 10.0
    home_dist_thresh_mm: float = 30.0
    min_phase_len: int = 3


class PhaseRelabeler:
    def __init__(self, cfg: PhaseLabelConfig):
        self.cfg = cfg

    def compute_contact_soft(self, df: pd.DataFrame) -> np.ndarray:
        cur = df["gripper_current"].to_numpy()
        vel = df["gripper_vel"].to_numpy()
        dcur = np.diff(cur, prepend=cur[0])

        score = (
            0.55 * np.clip((dcur - self.cfg.current_rise_thresh) / 50.0, 0, 1)
            + 0.45 * np.clip((-vel - 0.05) / 0.20, 0, 1)
        )
        return np.clip(score, 0.0, 1.0)

    def label_episode(self, df: pd.DataFrame) -> pd.DataFrame:
        contact_soft = self.compute_contact_soft(df)
        phase = np.full(len(df), PHASE_REACH, dtype=np.int64)

        for i in range(len(df)):
            xy_err = df.iloc[i]["target_xy_error_mm"]
            ee_z = df.iloc[i]["ee_z_mm"]
            ee_vz = df.iloc[i]["ee_vz_mm_s"]
            grip = df.iloc[i]["gripper_pos_rad"]
            dist_home = df.iloc[i]["home_dist_mm"]

            is_open = grip > -0.9
            is_closed = grip < -1.05
            has_contact = contact_soft[i] > 0.65

            if dist_home < self.cfg.home_dist_thresh_mm and i > len(df) * 0.7:
                phase[i] = PHASE_RETURN
            elif is_closed and has_contact and ee_vz > 5.0:
                phase[i] = PHASE_LIFT
            elif is_closed and has_contact and ee_vz < -5.0 and i > len(df) * 0.45:
                phase[i] = PHASE_PLACE
            elif (not is_closed) and (
                df.iloc[i]["gripper_vel_rad_s"] < self.cfg.close_vel_thresh
                or contact_soft[i] > 0.5
            ):
                phase[i] = PHASE_CLOSE
            elif xy_err <= self.cfg.align_xy_thresh_mm and is_open and abs(ee_vz) < 4.0:
                phase[i] = PHASE_ALIGN
            else:
                phase[i] = PHASE_REACH

        phase = self._smooth_short_segments(phase, self.cfg.min_phase_len)
        df = df.copy()
        df["contact_soft"] = contact_soft
        df["contact_hard"] = (contact_soft > 0.65).astype(np.int64)
        df["phase"] = phase
        return df

    def _smooth_short_segments(self, phase: np.ndarray, min_len: int) -> np.ndarray:
        out = phase.copy()
        start = 0
        while start < len(out):
            end = start + 1
            while end < len(out) and out[end] == out[start]:
                end += 1
            if end - start < min_len:
                left = out[start - 1] if start > 0 else out[end] if end < len(out) else out[start]
                out[start:end] = left
            start = end
        return out
