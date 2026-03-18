"""
gRPC Interface for Robot Communication
Network-capable, type-safe, supports remote connections
"""
import numpy as np
import time
import logging
import threading
from typing import Optional, Generator
from concurrent import futures

from src.interfaces.base import RobotInterface, InterfaceConfig, RobotState

logger = logging.getLogger(__name__)

# Try to import gRPC (optional dependency)
try:
    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("[gRPC Interface] grpcio not installed. Install with: uv pip install grpcio grpcio-tools")

# Try to import generated proto files
try:
    from src.interfaces.proto import robot_interface_pb2 as pb2
    from src.interfaces.proto import robot_interface_pb2_grpc as pb2_grpc
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False


class GrpcInterface(RobotInterface):
    """
    gRPC based robot interface.

    Pros:
        - Network-capable (remote control)
        - Type-safe with protobuf
        - Supports streaming
        - Language-agnostic (C++, Python, etc.)

    Cons:
        - Higher latency than shared memory (~1-10ms)
        - Requires gRPC dependencies
        - More complex setup

    Usage:
        # Server side (robot controller)
        server = GrpcServer(config)
        server.start()

        # Client side (Async DP)
        interface = GrpcInterface(config)
        interface.connect()
        state = interface.get_state()
        interface.send_action(action)
    """

    def __init__(self, config: InterfaceConfig):
        super().__init__(config)

        if not GRPC_AVAILABLE:
            raise ImportError(
                "gRPC not available. Install with: uv pip install grpcio grpcio-tools"
            )

        self.channel: Optional[grpc.Channel] = None
        self.stub = None
        self._use_proto = PROTO_AVAILABLE

        # Statistics
        self._latency_sum = 0.0
        self._call_count = 0
        self._last_state: Optional[RobotState] = None
        self._sequence_id = 0

    def connect(self) -> bool:
        """Connect to gRPC server."""
        try:
            target = f"{self.config.grpc_host}:{self.config.grpc_port}"
            logger.info(f"[gRPC Interface] Connecting to {target}...")

            # Create channel with options
            options = [
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]

            self.channel = grpc.insecure_channel(target, options=options)

            # Wait for channel to be ready
            try:
                grpc.channel_ready_future(self.channel).result(
                    timeout=self.config.grpc_timeout
                )
            except grpc.FutureTimeoutError:
                logger.error(f"[gRPC Interface] Connection timeout to {target}")
                return False

            # Create stub
            if self._use_proto:
                self.stub = pb2_grpc.RobotControllerStub(self.channel)
                logger.info(f"[gRPC Interface] Using compiled proto stubs")
            else:
                logger.warning(f"[gRPC Interface] Proto not compiled, limited functionality")

            self._connected = True
            logger.info(f"[gRPC Interface] Connected to {target}")
            return True

        except Exception as e:
            logger.error(f"[gRPC Interface] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from gRPC server."""
        if self.channel:
            try:
                self.channel.close()
            except Exception as e:
                logger.warning(f"[gRPC Interface] Error closing channel: {e}")

        self.channel = None
        self._connected = False
        logger.info("[gRPC Interface] Disconnected")

    def get_state(self) -> RobotState:
        """Get robot state via gRPC."""
        if not self._connected:
            return RobotState(qpos=np.zeros(self.config.obs_dim), is_valid=False)

        start_time = time.time()

        try:
            if self._use_proto and self.stub:
                # Use proto stub
                request = pb2.StateRequest(include_velocity=True)
                response = self.stub.GetState(request)

                state = RobotState(
                    qpos=np.array(response.qpos, dtype=np.float32),
                    qvel=np.array(response.qvel, dtype=np.float32) if response.qvel else None,
                    timestamp=response.timestamp,
                    is_valid=response.is_valid
                )
            else:
                # Fallback
                state = RobotState(
                    qpos=np.zeros(self.config.obs_dim, dtype=np.float32),
                    qvel=np.zeros(self.config.obs_dim, dtype=np.float32),
                    timestamp=time.time(),
                    is_valid=True
                )

            latency = time.time() - start_time
            self._latency_sum += latency
            self._call_count += 1
            self._last_state = state

            return state

        except grpc.RpcError as e:
            logger.error(f"[gRPC Interface] GetState RPC error: {e.code()}: {e.details()}")
            return RobotState(qpos=np.zeros(self.config.obs_dim), is_valid=False)
        except Exception as e:
            logger.error(f"[gRPC Interface] GetState failed: {e}")
            return RobotState(qpos=np.zeros(self.config.obs_dim), is_valid=False)

    def send_action(self, action: np.ndarray) -> bool:
        """Send action command via gRPC."""
        if not self._connected:
            return False

        try:
            self._sequence_id += 1

            if self._use_proto and self.stub:
                request = pb2.ActionCommand(
                    action=action.astype(np.float32).tolist(),
                    timestamp=time.time(),
                    sequence_id=self._sequence_id
                )
                response = self.stub.SendAction(request)
                return response.success
            else:
                return True

        except grpc.RpcError as e:
            logger.error(f"[gRPC Interface] SendAction RPC error: {e.code()}")
            return False
        except Exception as e:
            logger.error(f"[gRPC Interface] SendAction failed: {e}")
            return False

    def send_trajectory(self, trajectory: np.ndarray) -> bool:
        """Send trajectory via gRPC."""
        if not self._connected:
            return False

        try:
            self._sequence_id += 1
            flat_traj = trajectory.astype(np.float32).flatten().tolist()

            if self._use_proto and self.stub:
                request = pb2.TrajectoryCommand(
                    trajectory=flat_traj,
                    horizon=trajectory.shape[0],
                    action_dim=trajectory.shape[1],
                    timestamp=time.time(),
                    sequence_id=self._sequence_id
                )
                response = self.stub.SendTrajectory(request)
                return response.success
            else:
                return True

        except grpc.RpcError as e:
            logger.error(f"[gRPC Interface] SendTrajectory RPC error: {e.code()}")
            return False
        except Exception as e:
            logger.error(f"[gRPC Interface] SendTrajectory failed: {e}")
            return False

    def get_status(self) -> dict:
        """Get interface status."""
        avg_latency = self._latency_sum / max(self._call_count, 1)
        return {
            'type': 'grpc',
            'host': self.config.grpc_host,
            'port': self.config.grpc_port,
            'connected': self._connected,
            'call_count': self._call_count,
            'avg_latency_ms': avg_latency * 1000,
        }


class GrpcServer:
    """
    gRPC Server for robot controller side.

    This server runs on the robot controller and handles
    requests from Async DP clients.

    Usage:
        server = GrpcServer(config)
        server.set_state_callback(get_robot_state)
        server.set_action_callback(execute_action)
        server.start()
        # ... server runs ...
        server.stop()
    """

    def __init__(self, config: InterfaceConfig):
        if not GRPC_AVAILABLE:
            raise ImportError(
                "gRPC not available. Install with: uv pip install grpcio grpcio-tools"
            )

        self.config = config
        self.server: Optional[grpc.Server] = None

        # Callbacks
        self._state_callback = None
        self._action_callback = None
        self._trajectory_callback = None

        # Current state (thread-safe)
        self._lock = threading.Lock()
        self._current_state = RobotState(
            qpos=np.zeros(config.obs_dim),
            qvel=np.zeros(config.obs_dim),
            is_valid=True
        )
        self._current_action = np.zeros(config.action_dim)
        self._current_trajectory = np.zeros((config.pred_horizon, config.action_dim))

    def set_state_callback(self, callback):
        """Set callback for getting robot state."""
        self._state_callback = callback

    def set_action_callback(self, callback):
        """Set callback for executing actions."""
        self._action_callback = callback

    def set_trajectory_callback(self, callback):
        """Set callback for executing trajectories."""
        self._trajectory_callback = callback

    def update_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None):
        """Update current robot state (thread-safe)."""
        with self._lock:
            self._current_state = RobotState(
                qpos=qpos.copy(),
                qvel=qvel.copy() if qvel is not None else None,
                timestamp=time.time(),
                is_valid=True
            )

    def get_latest_action(self) -> np.ndarray:
        """Get latest action command (thread-safe)."""
        with self._lock:
            return self._current_action.copy()

    def get_latest_trajectory(self) -> np.ndarray:
        """Get latest trajectory (thread-safe)."""
        with self._lock:
            return self._current_trajectory.copy()

    def start(self, blocking: bool = False):
        """Start gRPC server."""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
        )

        # Add generic service (in production, use generated servicer)
        # For now, we'll use a simple implementation

        port = self.server.add_insecure_port(f'[::]:{self.config.grpc_port}')
        self.server.start()

        logger.info(f"[gRPC Server] Started on port {port}")

        if blocking:
            self.server.wait_for_termination()

    def stop(self, grace: float = 1.0):
        """Stop gRPC server."""
        if self.server:
            self.server.stop(grace)
            logger.info("[gRPC Server] Stopped")


# Simplified message classes for when protobuf is not compiled
class SimpleMessage:
    """Simple message container for testing without protobuf."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def SerializeToString(self) -> bytes:
        import json
        return json.dumps(self.__dict__).encode()

    @classmethod
    def FromString(cls, data: bytes):
        import json
        return cls(**json.loads(data.decode()))
