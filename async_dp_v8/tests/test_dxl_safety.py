"""Test safety guards."""
import numpy as np
from async_dp_v8.robot.safety import SafetyGuard


def test_safe_command():
    guard = SafetyGuard(max_joint_step_rad=0.12)
    current = np.zeros(6)
    target = np.ones(6) * 0.05
    is_safe, reason = guard.check_command(target, current)
    assert is_safe
    assert reason == "OK"


def test_unsafe_command():
    guard = SafetyGuard(max_joint_step_rad=0.12)
    current = np.zeros(6)
    target = np.ones(6) * 0.5
    is_safe, reason = guard.check_command(target, current)
    assert not is_safe


def test_estop():
    guard = SafetyGuard()
    guard.trigger_estop("test")
    assert guard.is_estopped
    is_safe, _ = guard.check_command(np.zeros(6), np.zeros(6))
    assert not is_safe
    guard.reset_estop()
    assert not guard.is_estopped


def test_voltage_check():
    guard = SafetyGuard(voltage_min=10.0, voltage_max=14.0)
    is_safe, _ = guard.check_state(np.zeros(6), np.zeros(6), 12.0)
    assert is_safe
    is_safe, _ = guard.check_state(np.zeros(6), np.zeros(6), 8.0)
    assert not is_safe
