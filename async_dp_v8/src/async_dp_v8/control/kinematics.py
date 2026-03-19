"""Forward kinematics for VX300s arm."""
import numpy as np
from typing import Tuple


# VX300s DH parameters (modified DH convention, approximate).
# Derived from Interbotix VX300s URDF — verify against actual URDF if precision matters.
#   d[0] = 0.0727  base-to-shoulder height
#   a[1] = 0.300   upper arm length (shoulder to elbow)
#   a[2] = 0.300   forearm length (elbow to wrist)
#   d[5] = 0.065   gripper offset (wrist_rotate to ee)
VX300S_DH = {
    "a": [0, 0.300, 0.300, 0, 0, 0],
    "alpha": [-np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2, 0],
    "d": [0.0727, 0, 0, 0, 0, 0.065],
}


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Compute 4x4 DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


def forward_kinematics(qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and rotation matrix from joint angles.

    qpos: [6] joint angles in radians
    Returns: (pos [3], rot [3, 3])
    """
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_matrix(
            VX300S_DH["a"][i],
            VX300S_DH["alpha"][i],
            VX300S_DH["d"][i],
            qpos[i],
        )
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
