"""
Base Robot Controller
Abstract base class with built-in synchronization and error recovery
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum
import numpy as np
import threading
import time
import logging

from src.interfaces import create_interface, InterfaceConfig
from src.interfaces.shm_interface import SharedMemoryInterface

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """Controller state machine"""
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class ControllerConfig:
    """Controller configuration"""
    # Timing
    control_freq: float = 500.0  # Hz
    state_update_freq: float = 500.0  # Hz

    # Interface
    interface_type: str = "shm"  # "shm" or "grpc"
    interface_config: InterfaceConfig = field(default_factory=InterfaceConfig)

    # Safety
    max_joint_velocity: float = 2.0  # rad/s
    emergency_stop_threshold: float = 3.14  # rad
    watchdog_timeout: float = 1.0  # seconds

    # Error recovery
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0  # seconds
    max_reconnect_attempts: int = 10

    # Logging
    log_interval: float = 1.0  # seconds


class BaseRobotController(ABC):
    """
    Base class for robot controllers with Async DP interface.

    Features:
        - Thread-safe state management
        - Automatic reconnection on failure
        - Watchdog timer for safety
        - Configurable control frequency
        - Built-in logging and monitoring

    Usage:
        class MyRobotController(BaseRobotController):
            def _read_state(self):
                return my_robot.get_joint_positions()

            def _write_command(self, action):
                my_robot.set_joint_positions(action)

        controller = MyRobotController(config)
        controller.start()
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._state = ControllerState.IDLE

        # Interface
        self._interface: Optional[SharedMemoryInterface] = None

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None

        # State
        self._current_qpos = np.zeros(config.interface_config.obs_dim, dtype=np.float32)
        self._current_qvel = np.zeros(config.interface_config.obs_dim, dtype=np.float32)
        self._target_action = np.zeros(config.interface_config.action_dim, dtype=np.float32)
        self._last_action_time = 0.0

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        self._on_action_received: Optional[Callable] = None

        # Statistics
        self._loop_count = 0
        self._error_count = 0
        self._last_log_time = 0.0
        self._loop_times: List[float] = []

    # =========================================================================
    # Abstract methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _read_state(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Read current robot state from hardware.

        Returns:
            tuple: (qpos, qvel) - joint positions and velocities
                   qvel can be None if not available
        """
        pass

    @abstractmethod
    def _write_command(self, action: np.ndarray) -> bool:
        """
        Write command to robot hardware.

        Args:
            action: Joint position command

        Returns:
            True if command was sent successfully
        """
        pass

    @abstractmethod
    def _connect_hardware(self) -> bool:
        """
        Connect to robot hardware.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def _disconnect_hardware(self) -> None:
        """Disconnect from robot hardware."""
        pass

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self) -> bool:
        """
        Start the controller.

        Returns:
            True if started successfully
        """
        if self._state == ControllerState.RUNNING:
            logger.warning("[Controller] Already running")
            return True

        logger.info("[Controller] Starting...")
        self._set_state(ControllerState.CONNECTING)

        # Connect to hardware
        if not self._connect_hardware():
            logger.error("[Controller] Hardware connection failed")
            self._set_state(ControllerState.ERROR)
            return False

        # Create and connect interface
        if not self._connect_interface():
            logger.error("[Controller] Interface connection failed")
            self._disconnect_hardware()
            self._set_state(ControllerState.ERROR)
            return False

        # Start control thread
        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="ControlLoop",
            daemon=True
        )
        self._control_thread.start()

        self._set_state(ControllerState.RUNNING)
        logger.info("[Controller] Started successfully")
        return True

    def stop(self) -> None:
        """Stop the controller gracefully."""
        if self._state == ControllerState.IDLE:
            return

        logger.info("[Controller] Stopping...")
        self._set_state(ControllerState.STOPPING)
        self._stop_event.set()

        # Wait for control thread
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)

        # Disconnect
        self._disconnect_interface()
        self._disconnect_hardware()

        self._set_state(ControllerState.IDLE)
        logger.info("[Controller] Stopped")

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return self._state

    def get_statistics(self) -> dict:
        """Get controller statistics."""
        with self._lock:
            avg_loop_time = np.mean(self._loop_times) if self._loop_times else 0
            return {
                'state': self._state.value,
                'loop_count': self._loop_count,
                'error_count': self._error_count,
                'avg_loop_time_ms': avg_loop_time * 1000,
                'current_qpos': self._current_qpos.tolist(),
                'target_action': self._target_action.tolist(),
            }

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_action_received: Optional[Callable] = None
    ) -> None:
        """Set event callbacks."""
        self._on_state_change = on_state_change
        self._on_error = on_error
        self._on_action_received = on_action_received

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _connect_interface(self) -> bool:
        """Connect to Async DP interface."""
        try:
            self._interface = create_interface(
                self.config.interface_type,
                self.config.interface_config,
                is_server=True
            )
            return self._interface.connect()
        except Exception as e:
            logger.error(f"[Controller] Interface error: {e}")
            return False

    def _disconnect_interface(self) -> None:
        """Disconnect from interface."""
        if self._interface:
            self._interface.disconnect()
            self._interface = None

    def _set_state(self, new_state: ControllerState) -> None:
        """Set controller state with callback."""
        old_state = self._state
        self._state = new_state
        if self._on_state_change and old_state != new_state:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"[Controller] State callback error: {e}")

    def _control_loop(self) -> None:
        """Main control loop running at control_freq Hz."""
        dt = 1.0 / self.config.control_freq

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                self._control_step()
            except Exception as e:
                self._handle_error(e)

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Track timing
            with self._lock:
                self._loop_times.append(elapsed)
                if len(self._loop_times) > 1000:
                    self._loop_times = self._loop_times[-500:]
                self._loop_count += 1

            # Periodic logging
            self._periodic_log()

    def _control_step(self) -> None:
        """Single control step."""
        # 1. Read hardware state
        qpos, qvel = self._read_state()

        with self._lock:
            self._current_qpos = qpos.astype(np.float32)
            if qvel is not None:
                self._current_qvel = qvel.astype(np.float32)

        # 2. Update interface with current state
        if self._interface and self._interface.is_connected:
            self._interface.update_state(self._current_qpos, self._current_qvel)

            # 3. Read action from Async DP
            new_action = self._interface.get_action()

            # Check if action changed (watchdog)
            action_changed = not np.allclose(new_action, self._target_action, atol=1e-6)

            if action_changed:
                self._last_action_time = time.time()
                with self._lock:
                    self._target_action = new_action.copy()

                if self._on_action_received:
                    try:
                        self._on_action_received(new_action)
                    except Exception as e:
                        logger.error(f"[Controller] Action callback error: {e}")

            # 4. Safety checks
            if self._check_safety():
                # 5. Apply action to hardware
                action = self._apply_velocity_limit(self._target_action)
                self._write_command(action)

    def _check_safety(self) -> bool:
        """Check safety conditions."""
        # Watchdog: check if receiving actions
        if self.config.watchdog_timeout > 0:
            time_since_action = time.time() - self._last_action_time
            if time_since_action > self.config.watchdog_timeout:
                if self._last_action_time > 0:  # Only warn if we were receiving
                    logger.warning(f"[Controller] Watchdog: No action for {time_since_action:.1f}s")
                return False

        # Emergency stop: check position limits
        if np.any(np.abs(self._current_qpos) > self.config.emergency_stop_threshold):
            logger.error("[Controller] Emergency stop: Position limit exceeded!")
            return False

        return True

    def _apply_velocity_limit(self, action: np.ndarray) -> np.ndarray:
        """Apply velocity limiting to action."""
        dt = 1.0 / self.config.control_freq
        max_delta = self.config.max_joint_velocity * dt

        delta = action - self._current_qpos
        delta = np.clip(delta, -max_delta, max_delta)

        return self._current_qpos + delta

    def _handle_error(self, error: Exception) -> None:
        """Handle control loop error."""
        self._error_count += 1
        logger.error(f"[Controller] Error: {error}")

        if self._on_error:
            try:
                self._on_error(error)
            except Exception as e:
                logger.error(f"[Controller] Error callback failed: {e}")

        # Attempt reconnection if configured
        if self.config.auto_reconnect and self._error_count < self.config.max_reconnect_attempts:
            logger.info(f"[Controller] Attempting reconnect ({self._error_count}/{self.config.max_reconnect_attempts})")
            time.sleep(self.config.reconnect_delay)
            self._reconnect()

    def _reconnect(self) -> None:
        """Attempt to reconnect hardware and interface."""
        try:
            self._disconnect_hardware()
            self._disconnect_interface()
            time.sleep(0.5)

            if self._connect_hardware() and self._connect_interface():
                logger.info("[Controller] Reconnected successfully")
                self._error_count = 0
            else:
                logger.error("[Controller] Reconnection failed")
        except Exception as e:
            logger.error(f"[Controller] Reconnection error: {e}")

    def _periodic_log(self) -> None:
        """Periodic status logging."""
        now = time.time()
        if now - self._last_log_time >= self.config.log_interval:
            self._last_log_time = now
            stats = self.get_statistics()
            logger.info(
                f"[Controller] Loops: {stats['loop_count']}, "
                f"Avg: {stats['avg_loop_time_ms']:.2f}ms, "
                f"Errors: {stats['error_count']}"
            )
