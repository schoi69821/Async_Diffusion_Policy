"""
Abstract Robot Interface
Defines the contract for all robot communication implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class InterfaceConfig:
    """Configuration for robot interfaces"""
    # Common
    action_dim: int = 14
    obs_dim: int = 14
    pred_horizon: int = 16

    # SharedMemory specific
    shm_name: str = "asyncdp_robot_interface"

    # gRPC specific
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    grpc_timeout: float = 1.0  # seconds

    # Connection
    max_retries: int = 5
    retry_delay: float = 0.1


@dataclass
class RobotState:
    """Robot state data structure"""
    qpos: np.ndarray  # Joint positions (obs_dim,)
    qvel: Optional[np.ndarray] = None  # Joint velocities (obs_dim,)
    timestamp: float = 0.0
    is_valid: bool = True


@dataclass
class ActionCommand:
    """Action command data structure"""
    action: np.ndarray  # Single action (action_dim,) or trajectory (horizon, action_dim)
    timestamp: float = 0.0
    sequence_id: int = 0


class RobotInterface(ABC):
    """
    Abstract base class for robot communication interfaces.

    This interface defines the contract between the Async DP controller
    and external robot control software.

    Implementations:
        - SharedMemoryInterface: Ultra-low latency, same PC only
        - GrpcInterface: Network-capable, type-safe, supports remote

    Usage:
        interface = create_interface('shm', config)
        interface.connect()

        # In control loop:
        state = interface.get_state()
        interface.send_action(action)

        interface.disconnect()
    """

    def __init__(self, config: InterfaceConfig):
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to robot controller.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """
        Get current robot state.

        Returns:
            RobotState with current joint positions/velocities
        """
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> bool:
        """
        Send action command to robot.

        Args:
            action: Joint position command (action_dim,)

        Returns:
            True if command sent successfully
        """
        pass

    @abstractmethod
    def send_trajectory(self, trajectory: np.ndarray) -> bool:
        """
        Send full action trajectory to robot.

        Args:
            trajectory: Action trajectory (pred_horizon, action_dim)

        Returns:
            True if trajectory sent successfully
        """
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """
        Get interface status information.

        Returns:
            Dict with connection status, latency, etc.
        """
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
