"""
gRPC Server for Robot Controller
Complete implementation with streaming support
"""
import numpy as np
import time
import threading
import logging
from typing import Optional, Callable, Generator
from concurrent import futures
from dataclasses import dataclass

from src.interfaces.base import InterfaceConfig, RobotState

logger = logging.getLogger(__name__)

# Try to import gRPC
try:
    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


# Try to import generated proto files
try:
    from src.interfaces.proto import robot_interface_pb2 as pb2
    from src.interfaces.proto import robot_interface_pb2_grpc as pb2_grpc
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False
    logger.info("[gRPC Server] Proto files not compiled. Using fallback implementation.")


@dataclass
class GrpcServerConfig:
    """gRPC server configuration"""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size: int = 50 * 1024 * 1024  # 50MB

    # Robot dimensions
    action_dim: int = 14
    obs_dim: int = 14
    pred_horizon: int = 16


class RobotControllerServicer:
    """
    gRPC service implementation for robot control.

    This servicer handles requests from Async DP clients.
    """

    def __init__(self, config: GrpcServerConfig):
        self.config = config
        self._lock = threading.RLock()

        # Current state
        self._qpos = np.zeros(config.obs_dim, dtype=np.float32)
        self._qvel = np.zeros(config.obs_dim, dtype=np.float32)
        self._state_timestamp = 0.0
        self._state_valid = True
        self._state_seq = 0

        # Current action/trajectory from client
        self._action = np.zeros(config.action_dim, dtype=np.float32)
        self._trajectory = np.zeros((config.pred_horizon, config.action_dim), dtype=np.float32)
        self._action_seq = 0

        # Callbacks
        self._on_action_callback: Optional[Callable] = None
        self._on_trajectory_callback: Optional[Callable] = None

    def set_action_callback(self, callback: Callable):
        """Set callback for when action is received."""
        self._on_action_callback = callback

    def set_trajectory_callback(self, callback: Callable):
        """Set callback for when trajectory is received."""
        self._on_trajectory_callback = callback

    def update_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None):
        """Update robot state (called by robot controller)."""
        with self._lock:
            np.copyto(self._qpos, qpos.astype(np.float32))
            if qvel is not None:
                np.copyto(self._qvel, qvel.astype(np.float32))
            self._state_timestamp = time.time()
            self._state_seq += 1

    def get_latest_action(self) -> np.ndarray:
        """Get latest action from client."""
        with self._lock:
            return self._action.copy()

    def get_latest_trajectory(self) -> np.ndarray:
        """Get latest trajectory from client."""
        with self._lock:
            return self._trajectory.copy()

    # =========================================================================
    # gRPC Service Methods (called by generated servicer)
    # =========================================================================

    def GetState(self, request, context):
        """Handle GetState RPC."""
        with self._lock:
            if PROTO_AVAILABLE:
                response = pb2.RobotState(
                    qpos=self._qpos.tolist(),
                    qvel=self._qvel.tolist() if request.include_velocity else [],
                    timestamp=self._state_timestamp,
                    is_valid=self._state_valid,
                    sequence_id=self._state_seq
                )
            else:
                response = {
                    'qpos': self._qpos.tolist(),
                    'qvel': self._qvel.tolist(),
                    'timestamp': self._state_timestamp,
                    'is_valid': self._state_valid,
                    'sequence_id': self._state_seq
                }
            return response

    def SendAction(self, request, context):
        """Handle SendAction RPC."""
        with self._lock:
            if PROTO_AVAILABLE:
                action = np.array(request.action, dtype=np.float32)
            else:
                action = np.array(request.get('action', []), dtype=np.float32)

            np.copyto(self._action, action)
            self._action_seq += 1

        # Call callback if registered
        if self._on_action_callback:
            try:
                self._on_action_callback(action)
            except Exception as e:
                logger.error(f"[gRPC] Action callback error: {e}")

        if PROTO_AVAILABLE:
            return pb2.CommandResponse(
                success=True,
                message="OK",
                sequence_id=self._action_seq
            )
        else:
            return {'success': True, 'message': 'OK', 'sequence_id': self._action_seq}

    def SendTrajectory(self, request, context):
        """Handle SendTrajectory RPC."""
        with self._lock:
            if PROTO_AVAILABLE:
                flat_traj = np.array(request.trajectory, dtype=np.float32)
                horizon = request.horizon
                action_dim = request.action_dim
            else:
                flat_traj = np.array(request.get('trajectory', []), dtype=np.float32)
                horizon = request.get('horizon', self.config.pred_horizon)
                action_dim = request.get('action_dim', self.config.action_dim)

            trajectory = flat_traj.reshape(horizon, action_dim)
            np.copyto(self._trajectory, trajectory)
            self._action_seq += 1

        # Call callback if registered
        if self._on_trajectory_callback:
            try:
                self._on_trajectory_callback(trajectory)
            except Exception as e:
                logger.error(f"[gRPC] Trajectory callback error: {e}")

        if PROTO_AVAILABLE:
            return pb2.CommandResponse(
                success=True,
                message="OK",
                sequence_id=self._action_seq
            )
        else:
            return {'success': True, 'message': 'OK', 'sequence_id': self._action_seq}

    def StreamControl(self, request_iterator, context):
        """Handle bidirectional streaming for real-time control."""
        logger.info("[gRPC] Stream control started")

        try:
            for request in request_iterator:
                # Process incoming action
                with self._lock:
                    if PROTO_AVAILABLE:
                        action = np.array(request.action, dtype=np.float32)
                    else:
                        action = np.array(request.get('action', []), dtype=np.float32)
                    np.copyto(self._action, action)

                # Call callback
                if self._on_action_callback:
                    self._on_action_callback(action)

                # Yield current state
                with self._lock:
                    if PROTO_AVAILABLE:
                        yield pb2.RobotState(
                            qpos=self._qpos.tolist(),
                            qvel=self._qvel.tolist(),
                            timestamp=self._state_timestamp,
                            is_valid=self._state_valid,
                            sequence_id=self._state_seq
                        )
                    else:
                        yield {
                            'qpos': self._qpos.tolist(),
                            'qvel': self._qvel.tolist(),
                            'timestamp': self._state_timestamp,
                            'is_valid': self._state_valid,
                            'sequence_id': self._state_seq
                        }

        except Exception as e:
            logger.error(f"[gRPC] Stream error: {e}")

        logger.info("[gRPC] Stream control ended")

    def GetStatus(self, request, context):
        """Handle GetStatus RPC."""
        if PROTO_AVAILABLE:
            return pb2.ControllerStatus(
                controller_name="AsyncDP_RobotController",
                is_ready=True,
                is_enabled=True,
                mode="running",
                control_frequency=500.0,
                action_dim=self.config.action_dim,
                obs_dim=self.config.obs_dim
            )
        else:
            return {
                'controller_name': 'AsyncDP_RobotController',
                'is_ready': True,
                'is_enabled': True,
                'mode': 'running',
                'control_frequency': 500.0,
                'action_dim': self.config.action_dim,
                'obs_dim': self.config.obs_dim
            }


class GrpcRobotServer:
    """
    gRPC Server for robot controller.

    Usage:
        config = GrpcServerConfig(port=50051)
        server = GrpcRobotServer(config)

        # Set callbacks
        server.set_action_callback(handle_action)

        # Start server
        server.start()

        # Update state in control loop
        while running:
            qpos = read_motors()
            server.update_state(qpos)
            action = server.get_latest_action()

        # Stop server
        server.stop()
    """

    def __init__(self, config: GrpcServerConfig):
        if not GRPC_AVAILABLE:
            raise ImportError("grpcio not installed. Install with: pip install grpcio")

        self.config = config
        self._server: Optional[grpc.Server] = None
        self._servicer = RobotControllerServicer(config)

    def set_action_callback(self, callback: Callable):
        """Set callback for action received."""
        self._servicer.set_action_callback(callback)

    def set_trajectory_callback(self, callback: Callable):
        """Set callback for trajectory received."""
        self._servicer.set_trajectory_callback(callback)

    def update_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None):
        """Update robot state."""
        self._servicer.update_state(qpos, qvel)

    def get_latest_action(self) -> np.ndarray:
        """Get latest action from client."""
        return self._servicer.get_latest_action()

    def get_latest_trajectory(self) -> np.ndarray:
        """Get latest trajectory from client."""
        return self._servicer.get_latest_trajectory()

    def start(self, blocking: bool = False):
        """Start gRPC server."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ('grpc.max_send_message_length', self.config.max_message_size),
                ('grpc.max_receive_message_length', self.config.max_message_size),
            ]
        )

        # Add servicer
        if PROTO_AVAILABLE:
            pb2_grpc.add_RobotControllerServicer_to_server(
                self._servicer, self._server
            )
        else:
            # Fallback: use generic service
            logger.warning("[gRPC] Proto not compiled. Server functionality limited.")

        # Start server
        address = f"{self.config.host}:{self.config.port}"
        self._server.add_insecure_port(address)
        self._server.start()

        logger.info(f"[gRPC Server] Started on {address}")

        if blocking:
            self._server.wait_for_termination()

    def stop(self, grace: float = 1.0):
        """Stop gRPC server."""
        if self._server:
            self._server.stop(grace)
            logger.info("[gRPC Server] Stopped")


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = GrpcServerConfig(port=50051, action_dim=6, obs_dim=6)
    server = GrpcRobotServer(config)

    def on_action(action):
        print(f"Action received: {action[:3]}...")

    server.set_action_callback(on_action)
    server.start(blocking=False)

    print("Server running. Press Ctrl+C to stop...")

    try:
        # Simulate robot state updates
        t = 0
        while True:
            qpos = np.sin(np.arange(6) * 0.1 + t * 0.1).astype(np.float32)
            server.update_state(qpos)
            t += 1
            time.sleep(0.002)

    except KeyboardInterrupt:
        server.stop()
