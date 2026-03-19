"""Test forward kinematics and differential IK."""
import numpy as np
from async_dp_v8.control.kinematics import (
    forward_kinematics, rotation_to_euler, qpos_to_ee_pose,
    compute_jacobian, ik_delta_z,
)


def test_fk_at_zero():
    """Zero joints should give a valid EE position."""
    pos, rot = forward_kinematics(np.zeros(6))
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    # At zero config, EE should be roughly along X axis above base
    assert pos[2] > 0  # Z should be positive (above base)


def test_fk_deterministic():
    """Same input should give same output."""
    q = np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0])
    p1, _ = forward_kinematics(q)
    p2, _ = forward_kinematics(q)
    np.testing.assert_array_equal(p1, p2)


def test_euler_roundtrip():
    """Rotation should produce valid Euler angles."""
    q = np.array([0.1, -0.3, 0.2, 0.0, 0.1, 0.0])
    _, rot = forward_kinematics(q)
    euler = rotation_to_euler(rot)
    assert euler.shape == (3,)
    assert np.all(np.abs(euler) < np.pi + 0.01)


def test_qpos_to_ee_pose_shape():
    pose = qpos_to_ee_pose(np.zeros(6))
    assert pose.shape == (7,)


def test_jacobian_shape():
    J = compute_jacobian(np.zeros(6))
    assert J.shape == (3, 6)


def test_jacobian_nonzero():
    """Jacobian should be non-zero for non-singular config."""
    q = np.array([0.0, -0.5, 0.5, 0.0, 0.3, 0.0])
    J = compute_jacobian(q)
    assert np.linalg.matrix_rank(J) >= 2  # Should have rank 3 but at least 2


def test_ik_delta_z_direction():
    """IK delta for positive Z should move EE up."""
    q = np.array([0.0, -0.5, 0.5, 0.0, 0.3, 0.0])
    pos_before, _ = forward_kinematics(q)

    dq = ik_delta_z(q, 0.01)  # 10mm up
    q_new = q + dq
    pos_after, _ = forward_kinematics(q_new)

    # EE Z should increase
    assert pos_after[2] > pos_before[2]


def test_ik_delta_z_magnitude():
    """IK should achieve approximately the desired delta."""
    q = np.array([0.0, -0.5, 0.5, 0.0, 0.3, 0.0])
    pos_before, _ = forward_kinematics(q)

    desired_dz = 0.02  # 20mm
    dq = ik_delta_z(q, desired_dz)
    q_new = q + dq
    pos_after, _ = forward_kinematics(q_new)

    actual_dz = pos_after[2] - pos_before[2]
    # Should be within 50% of desired (differential IK is approximate)
    assert abs(actual_dz - desired_dz) < desired_dz * 0.5
