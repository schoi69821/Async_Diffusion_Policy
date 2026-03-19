"""Forward kinematics for Interbotix VX300s 6DOF arm.

Uses URDF-derived transform chain instead of DH convention for accuracy.

VX300s 6DOF joint chain:
  1. waist:        Z rotation, origin [0, 0, 0.07285] from base
  2. shoulder:     Y rotation, origin [0.04825, 0, 0.04805] from waist
  3. elbow:        Y rotation, origin [0.3, 0, 0] from shoulder (upper arm)
  4. forearm_roll: X rotation, origin [0, 0, 0] from elbow
  5. wrist_angle:  Y rotation, origin [0.3, 0, 0] from forearm (forearm length)
  6. wrist_rotate: X rotation, origin [0.065, 0, 0] from wrist (to EE)
"""
import numpy as np
from typing import Tuple


# Link offsets from Interbotix VX300s URDF (meters)
LINK_OFFSETS = [
    np.array([0.0, 0.0, 0.07285]),     # base to waist
    np.array([0.04825, 0.0, 0.04805]),  # waist to shoulder
    np.array([0.300, 0.0, 0.0]),        # shoulder to elbow (upper arm)
    np.array([0.0, 0.0, 0.0]),          # elbow to forearm_roll (coincident)
    np.array([0.300, 0.0, 0.0]),        # forearm_roll to wrist (forearm)
    np.array([0.065, 0.0, 0.0]),        # wrist to EE (gripper offset)
]

# Joint axes: 'z', 'y', 'y', 'x', 'y', 'x'
JOINT_AXES = ['z', 'y', 'y', 'x', 'y', 'x']


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


_ROT_FN = {'x': _rot_x, 'y': _rot_y, 'z': _rot_z}


def _make_tf(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from 3x3 rotation and 3D translation."""
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def forward_kinematics(qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and rotation matrix from joint angles.

    Uses the Interbotix VX300s URDF-derived transform chain.

    qpos: [6] joint angles in radians
    Returns: (pos [3], rot [3, 3])
    """
    T = np.eye(4)
    for i in range(6):
        # Translate to joint origin
        T = T @ _make_tf(np.eye(3), LINK_OFFSETS[i])
        # Rotate around joint axis
        rot_fn = _ROT_FN[JOINT_AXES[i]]
        T = T @ _make_tf(rot_fn(qpos[i]), np.zeros(3))

    return T[:3, 3], T[:3, :3]


def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to ZYX Euler angles."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def compute_jacobian(qpos: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute 3x6 position Jacobian via finite differences.

    Returns: [3, 6] matrix mapping joint velocities to EE linear velocity.
    """
    J = np.zeros((3, 6))
    pos0, _ = forward_kinematics(qpos)
    for i in range(6):
        q_perturbed = qpos.copy()
        q_perturbed[i] += eps
        pos_perturbed, _ = forward_kinematics(q_perturbed)
        J[:, i] = (pos_perturbed - pos0) / eps
    return J


def ik_delta_z(qpos: np.ndarray, delta_z_m: float, damping: float = 0.01) -> np.ndarray:
    """Compute joint delta to achieve a desired EE Z displacement.

    Uses damped least-squares (Jacobian pseudoinverse) for robustness.

    qpos: [6] current joint angles
    delta_z_m: desired Z displacement in meters
    damping: regularization factor
    Returns: [6] joint angle deltas
    """
    J = compute_jacobian(qpos)
    # We only care about Z (row 2 of Jacobian)
    Jz = J[2:3, :]  # [1, 6]

    # Damped pseudoinverse: J^T (J J^T + lambda^2 I)^-1
    JJT = Jz @ Jz.T + damping ** 2 * np.eye(1)
    dq = Jz.T @ np.linalg.solve(JJT, np.array([[delta_z_m]]))
    return dq.flatten()


def qpos_to_ee_pose(qpos: np.ndarray) -> np.ndarray:
    """Convert joint positions to 7D ee pose [x, y, z, roll, pitch, yaw, grip_dummy].

    qpos: [6] arm joints
    Returns: [7] (position + euler angles + 0)
    """
    pos, rot = forward_kinematics(qpos)
    euler = rotation_to_euler(rot)
    return np.concatenate([pos, euler, [0.0]])
