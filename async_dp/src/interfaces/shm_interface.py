"""
Shared Memory Interface for Robot Communication
Ultra-low latency (~1μs), same PC only
"""
from multiprocessing import shared_memory
import numpy as np
import time
import logging
from typing import Optional

from src.interfaces.base import RobotInterface, InterfaceConfig, RobotState, ActionCommand

logger = logging.getLogger(__name__)


class SharedMemoryInterface(RobotInterface):
    """
    Shared Memory based robot interface.

    Memory Layout:
        [0:action_size]     - action_command: (action_dim,) float32
        [action_size:+traj] - action_trajectory: (pred_horizon, action_dim) float32
        [+traj:+obs]        - robot_state: (obs_dim,) float32
        [+obs:+vel]         - robot_velocity: (obs_dim,) float32
        [+vel:+meta]        - metadata: [timestamp, sequence_id, state_valid] float64

    Pros:
        - Ultra-low latency (~1μs)
        - Zero-copy data sharing
        - Simple implementation

    Cons:
        - Same PC only
        - No built-in synchronization (use external lock)
        - Platform-specific behavior

    Usage:
        # Robot controller side (creates shared memory)
        interface = SharedMemoryInterface(config, is_server=True)
        interface.connect()

        # Async DP side (attaches to shared memory)
        interface = SharedMemoryInterface(config, is_server=False)
        interface.connect()
    """

    def __init__(self, config: InterfaceConfig, is_server: bool = False):
        """
        Initialize SharedMemory interface.

        Args:
            config: Interface configuration
            is_server: If True, creates shared memory. If False, attaches to existing.
        """
        super().__init__(config)
        self.is_server = is_server

        # Calculate sizes
        self.action_size = 4 * config.action_dim  # float32
        self.traj_size = 4 * config.pred_horizon * config.action_dim  # float32
        self.state_size = 4 * config.obs_dim  # float32
        self.vel_size = 4 * config.obs_dim  # float32
        self.meta_size = 8 * 4  # 4 x float64 (timestamp, seq_id, valid, reserved)

        self.total_size = (
            self.action_size +
            self.traj_size +
            self.state_size +
            self.vel_size +
            self.meta_size
        )

        # Offsets
        self.offset_action = 0
        self.offset_traj = self.action_size
        self.offset_state = self.offset_traj + self.traj_size
        self.offset_vel = self.offset_state + self.state_size
        self.offset_meta = self.offset_vel + self.vel_size

        # Shared memory handle
        self.shm: Optional[shared_memory.SharedMemory] = None

        # Numpy arrays mapped to shared memory
        self.action_cmd: Optional[np.ndarray] = None
        self.action_traj: Optional[np.ndarray] = None
        self.robot_state: Optional[np.ndarray] = None
        self.robot_vel: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None

        # Statistics
        self._last_read_time = 0.0
        self._read_count = 0
        self._write_count = 0

    def connect(self) -> bool:
        """Connect to or create shared memory."""
        try:
            if self.is_server:
                self._create_shm()
            else:
                self._attach_shm()

            self._map_arrays()
            self._connected = True

            mode = "server (created)" if self.is_server else "client (attached)"
            logger.info(f"[SHM Interface] Connected as {mode}: {self.config.shm_name}")
            return True

        except Exception as e:
            logger.error(f"[SHM Interface] Connection failed: {e}")
            return False

    def _create_shm(self):
        """Create new shared memory segment."""
        # Clean up existing if present
        try:
            existing = shared_memory.SharedMemory(name=self.config.shm_name)
            existing.close()
            existing.unlink()
            time.sleep(0.05)
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(
            name=self.config.shm_name,
            create=True,
            size=self.total_size
        )

    def _attach_shm(self):
        """Attach to existing shared memory with retry."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=self.config.shm_name)
                return
            except FileNotFoundError:
                last_error = FileNotFoundError(
                    f"Shared memory '{self.config.shm_name}' not found"
                )
                logger.warning(
                    f"[SHM Interface] Waiting for server... "
                    f"({attempt + 1}/{self.config.max_retries})"
                )
                time.sleep(self.config.retry_delay)

        raise last_error

    def _map_arrays(self):
        """Map numpy arrays to shared memory buffer."""
        buf = self.shm.buf

        self.action_cmd = np.ndarray(
            (self.config.action_dim,),
            dtype=np.float32,
            buffer=buf,
            offset=self.offset_action
        )

        self.action_traj = np.ndarray(
            (self.config.pred_horizon, self.config.action_dim),
            dtype=np.float32,
            buffer=buf,
            offset=self.offset_traj
        )

        self.robot_state = np.ndarray(
            (self.config.obs_dim,),
            dtype=np.float32,
            buffer=buf,
            offset=self.offset_state
        )

        self.robot_vel = np.ndarray(
            (self.config.obs_dim,),
            dtype=np.float32,
            buffer=buf,
            offset=self.offset_vel
        )

        self.metadata = np.ndarray(
            (4,),
            dtype=np.float64,
            buffer=buf,
            offset=self.offset_meta
        )

        # Initialize if server
        if self.is_server:
            self.action_cmd.fill(0)
            self.action_traj.fill(0)
            self.robot_state.fill(0)
            self.robot_vel.fill(0)
            self.metadata.fill(0)
            self.metadata[2] = 1.0  # state_valid = True

    def disconnect(self) -> None:
        """Close shared memory connection."""
        if self.shm is None:
            return

        try:
            self.shm.close()
            if self.is_server:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception as e:
            logger.warning(f"[SHM Interface] Error closing: {e}")
        finally:
            self.shm = None
            self._connected = False

        logger.info("[SHM Interface] Disconnected")

    def get_state(self) -> RobotState:
        """Get current robot state from shared memory."""
        if not self._connected:
            return RobotState(qpos=np.zeros(self.config.obs_dim), is_valid=False)

        self._last_read_time = time.time()
        self._read_count += 1

        return RobotState(
            qpos=self.robot_state.copy(),
            qvel=self.robot_vel.copy(),
            timestamp=float(self.metadata[0]),
            is_valid=bool(self.metadata[2] > 0.5)
        )

    def send_action(self, action: np.ndarray) -> bool:
        """Send single action command."""
        if not self._connected:
            return False

        try:
            np.copyto(self.action_cmd, action.astype(np.float32))
            self.metadata[1] += 1  # Increment sequence_id
            self._write_count += 1
            return True
        except Exception as e:
            logger.error(f"[SHM Interface] Send action failed: {e}")
            return False

    def send_trajectory(self, trajectory: np.ndarray) -> bool:
        """Send full action trajectory."""
        if not self._connected:
            return False

        try:
            np.copyto(self.action_traj, trajectory.astype(np.float32))
            self.metadata[1] += 1  # Increment sequence_id
            self._write_count += 1
            return True
        except Exception as e:
            logger.error(f"[SHM Interface] Send trajectory failed: {e}")
            return False

    def update_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> bool:
        """
        Update robot state (called by robot controller side).

        Args:
            qpos: Current joint positions
            qvel: Current joint velocities (optional)
        """
        if not self._connected:
            return False

        try:
            np.copyto(self.robot_state, qpos.astype(np.float32))
            if qvel is not None:
                np.copyto(self.robot_vel, qvel.astype(np.float32))
            self.metadata[0] = time.time()  # Update timestamp
            self.metadata[2] = 1.0  # state_valid = True
            return True
        except Exception as e:
            logger.error(f"[SHM Interface] Update state failed: {e}")
            return False

    def get_action(self) -> np.ndarray:
        """
        Get current action command (called by robot controller side).

        Returns:
            Action command array
        """
        if not self._connected:
            return np.zeros(self.config.action_dim)

        return self.action_cmd.copy()

    def get_trajectory(self) -> np.ndarray:
        """
        Get current action trajectory (called by robot controller side).

        Returns:
            Action trajectory array
        """
        if not self._connected:
            return np.zeros((self.config.pred_horizon, self.config.action_dim))

        return self.action_traj.copy()

    def get_status(self) -> dict:
        """Get interface status."""
        return {
            'type': 'shared_memory',
            'name': self.config.shm_name,
            'connected': self._connected,
            'is_server': self.is_server,
            'size_bytes': self.total_size,
            'read_count': self._read_count,
            'write_count': self._write_count,
            'last_read_time': self._last_read_time,
        }
