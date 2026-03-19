"""Geometry utilities for pose conversions."""
import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """ZYX Euler angles to quaternion [x, y, z, w]."""
    return Rotation.from_euler("ZYX", euler).as_quat()


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] to ZYX Euler angles."""
    return Rotation.from_quat(quat).as_euler("ZYX")


def pose_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 3D positions."""
    return float(np.linalg.norm(p1[:3] - p2[:3]))
