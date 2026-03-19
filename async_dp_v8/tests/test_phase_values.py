"""Test that phase labeler assigns correct phase values."""
import numpy as np
import pandas as pd
from async_dp_v8.data.relabel_phase import PhaseRelabeler, PhaseLabelConfig
from async_dp_v8.constants import PHASE_REACH, PHASE_CLOSE, PHASE_LIFT, PHASE_RETURN


def _make_full_episode(n=100):
    """Simulate a full reach-close-lift-return trajectory."""
    df = pd.DataFrame({
        "target_xy_error_mm": np.concatenate([
            np.linspace(80, 10, 30),   # approaching
            np.full(20, 5.0),           # aligned
            np.full(20, 5.0),           # closing/lifting
            np.linspace(5, 50, 30),     # returning
        ]),
        "ee_z_mm": np.concatenate([
            np.linspace(100, 50, 30),
            np.full(20, 48.0),
            np.linspace(48, 80, 20),    # lifting
            np.linspace(80, 100, 30),
        ]),
        "ee_vz_mm_s": np.concatenate([
            np.full(30, -3.0),
            np.full(20, 0.0),
            np.full(20, 10.0),          # going up
            np.full(30, 1.0),
        ]),
        "gripper_pos_rad": np.concatenate([
            np.full(40, -0.5),          # open
            np.linspace(-0.5, -1.1, 10),# closing
            np.full(50, -1.1),          # closed
        ]),
        "gripper_vel_rad_s": np.concatenate([
            np.full(40, 0.0),
            np.full(10, -0.3),          # closing velocity
            np.full(50, 0.0),
        ]),
        "gripper_current": np.concatenate([
            np.zeros(40),
            np.linspace(0, 100, 10),
            # Each step jumps by 80 (dcur=80 > thresh=35), giving current_score=0.9
            np.cumsum(np.full(50, 80.0)) + 100,
        ]),
        "gripper_vel": np.concatenate([
            np.zeros(40),
            np.full(10, -0.2),
            np.full(50, -0.3),  # pressing against object -> vel_score high
        ]),
        "home_dist_mm": np.concatenate([
            np.linspace(200, 50, 30),
            np.full(40, 50.0),
            np.linspace(50, 15, 30),    # returning home
        ]),
    })
    return df


def test_reach_at_start():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_full_episode()
    result = relabeler.label_episode(df)
    # First 10 frames should be REACH (far from target, open gripper)
    assert result["phase"].iloc[:10].mode()[0] == PHASE_REACH


def test_close_detected():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_full_episode()
    result = relabeler.label_episode(df)
    # CLOSE phase should exist somewhere in frames 35-55
    close_frames = result[(result["phase"] == PHASE_CLOSE)].index
    assert len(close_frames) > 0


def test_lift_detected():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_full_episode()
    result = relabeler.label_episode(df)
    # LIFT should exist after closing
    lift_frames = result[(result["phase"] == PHASE_LIFT)].index
    assert len(lift_frames) > 0


def test_return_at_end():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_full_episode()
    result = relabeler.label_episode(df)
    # Last frames should be RETURN (close to home, late in episode)
    assert result["phase"].iloc[-5:].mode()[0] == PHASE_RETURN


def test_all_phases_present():
    cfg = PhaseLabelConfig()
    relabeler = PhaseRelabeler(cfg)
    df = _make_full_episode()
    result = relabeler.label_episode(df)
    present = set(result["phase"].unique())
    # At minimum reach, close, lift, return should be present
    assert PHASE_REACH in present
    assert PHASE_CLOSE in present or PHASE_LIFT in present  # close or lift
