"""Test phase relabeling."""
import numpy as np
import pandas as pd
from async_dp_v8.data.relabel_phase import PhaseRelabeler, PhaseLabelConfig
from async_dp_v8.constants import PHASE_REACH, PHASE_ALIGN, PHASE_CLOSE, PHASE_LIFT


def _make_episode_df(n=50):
    df = pd.DataFrame({
        "target_xy_error_mm": np.linspace(50, 5, n),
        "ee_z_mm": np.concatenate([
            np.linspace(100, 50, n // 2),
            np.linspace(50, 80, n // 2),
        ]),
        "ee_vz_mm_s": np.zeros(n),
        "gripper_pos_rad": np.full(n, -0.5),
        "gripper_vel_rad_s": np.zeros(n),
        "gripper_current": np.zeros(n),
        "gripper_vel": np.zeros(n),
        "home_dist_mm": np.linspace(200, 20, n),
    })
    # Simulate closing at frame 30
    df.loc[30:, "gripper_pos_rad"] = -1.1
    df.loc[28:32, "gripper_vel_rad_s"] = -0.3
    df.loc[30:, "gripper_current"] = np.linspace(0, 100, n - 30)
    df.loc[30:, "gripper_vel"] = -0.2
    # Simulate lifting
    df.loc[35:, "ee_vz_mm_s"] = 10.0
    return df


def test_phase_relabeler_basic():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_episode_df()
    result = relabeler.label_episode(df)
    assert "phase" in result.columns
    assert "contact_soft" in result.columns
    assert "contact_hard" in result.columns
    assert len(result) == len(df)


def test_contact_soft_range():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_episode_df()
    contact = relabeler.compute_contact_soft(df)
    assert np.all(contact >= 0.0)
    assert np.all(contact <= 1.0)


def test_smooth_short_segments():
    cfg = PhaseLabelConfig(min_phase_len=3)
    relabeler = PhaseRelabeler(cfg)
    phase = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    smoothed = relabeler._smooth_short_segments(phase, min_len=3)
    assert smoothed[3] == 0  # short segment should be smoothed
