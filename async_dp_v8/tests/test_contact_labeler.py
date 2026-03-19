"""Test contact relabeling."""
import numpy as np
import pandas as pd
from async_dp_v8.data.relabel_contact import ContactRelabeler, ContactLabelConfig


def _make_df(n=100):
    """Create synthetic gripper data with a clear contact event at frame 50."""
    df = pd.DataFrame({
        "gripper_current": np.concatenate([
            np.zeros(50),
            np.linspace(0, 200, 50),  # current ramps up during contact
        ]),
        "gripper_vel": np.concatenate([
            np.full(50, -0.3),  # closing
            np.full(50, 0.0),   # stalled (contact)
        ]),
        "gripper_pos_rad": np.concatenate([
            np.linspace(-0.5, -1.0, 50),
            np.full(50, -1.05),  # closed
        ]),
    })
    return df


def test_contact_soft_range():
    cfg = ContactLabelConfig()
    relabeler = ContactRelabeler(cfg)
    df = _make_df()
    result = relabeler.label_episode(df)
    assert np.all(result["contact_soft"] >= 0.0)
    assert np.all(result["contact_soft"] <= 1.0)


def test_contact_hard_binary():
    cfg = ContactLabelConfig()
    relabeler = ContactRelabeler(cfg)
    df = _make_df()
    result = relabeler.label_episode(df)
    assert set(result["contact_hard"].unique()).issubset({0, 1})


def test_no_contact_before_event():
    cfg = ContactLabelConfig()
    relabeler = ContactRelabeler(cfg)
    df = _make_df()
    result = relabeler.label_episode(df)
    # First 40 frames should have low contact (before closing starts)
    assert result["contact_soft"].iloc[:40].mean() < 0.3


def test_contact_after_event():
    cfg = ContactLabelConfig()
    relabeler = ContactRelabeler(cfg)
    df = _make_df()
    result = relabeler.label_episode(df)
    # Last 20 frames should have high contact (current is high, velocity stalled)
    assert result["contact_soft"].iloc[-20:].mean() > 0.3


def test_smoothing_window():
    cfg = ContactLabelConfig(soft_blend_window=5)
    relabeler = ContactRelabeler(cfg)
    df = _make_df()
    result = relabeler.label_episode(df)
    # Should not have sharp transitions due to smoothing
    diffs = np.abs(np.diff(result["contact_soft"].to_numpy()))
    assert np.max(diffs) < 0.5  # no jump > 0.5 in one frame


def test_configurable_threshold():
    cfg_low = ContactLabelConfig(hard_thresh=0.3)
    cfg_high = ContactLabelConfig(hard_thresh=0.9)
    relabeler_low = ContactRelabeler(cfg_low)
    relabeler_high = ContactRelabeler(cfg_high)
    df = _make_df()
    result_low = relabeler_low.label_episode(df)
    result_high = relabeler_high.label_episode(df)
    # Lower threshold should detect more contact
    assert result_low["contact_hard"].sum() >= result_high["contact_hard"].sum()
