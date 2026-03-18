"""
Interface Factory
Creates robot communication interfaces based on configuration
"""
from enum import Enum
from typing import Union
import logging

from src.interfaces.base import RobotInterface, InterfaceConfig

logger = logging.getLogger(__name__)


class InterfaceType(Enum):
    """Available interface types"""
    SHARED_MEMORY = "shm"
    GRPC = "grpc"
    DUMMY = "dummy"


def create_interface(
    interface_type: Union[InterfaceType, str],
    config: InterfaceConfig = None,
    is_server: bool = False,
    **kwargs
) -> RobotInterface:
    """
    Factory function to create robot interfaces.

    Args:
        interface_type: Type of interface ('shm', 'grpc', or 'dummy')
        config: Interface configuration (uses defaults if None)
        is_server: For SharedMemory, whether to create (True) or attach (False)
        **kwargs: Additional arguments passed to interface constructor

    Returns:
        RobotInterface instance

    Example:
        # Create SharedMemory interface (client mode)
        interface = create_interface('shm')

        # Create gRPC interface
        interface = create_interface('grpc', config=InterfaceConfig(grpc_host='192.168.1.100'))

        # Create SharedMemory interface (server mode)
        interface = create_interface('shm', is_server=True)
    """
    if config is None:
        config = InterfaceConfig()

    # Normalize interface type
    if isinstance(interface_type, str):
        interface_type = interface_type.lower()
        type_map = {
            'shm': InterfaceType.SHARED_MEMORY,
            'shared_memory': InterfaceType.SHARED_MEMORY,
            'grpc': InterfaceType.GRPC,
            'dummy': InterfaceType.DUMMY,
        }
        interface_type = type_map.get(interface_type, InterfaceType.DUMMY)

    # Create interface
    if interface_type == InterfaceType.SHARED_MEMORY:
        from src.interfaces.shm_interface import SharedMemoryInterface
        logger.info(f"[Factory] Creating SharedMemory interface (server={is_server})")
        return SharedMemoryInterface(config, is_server=is_server, **kwargs)

    elif interface_type == InterfaceType.GRPC:
        from src.interfaces.grpc_interface import GrpcInterface
        logger.info(f"[Factory] Creating gRPC interface ({config.grpc_host}:{config.grpc_port})")
        return GrpcInterface(config, **kwargs)

    elif interface_type == InterfaceType.DUMMY:
        logger.info("[Factory] Creating Dummy interface")
        return DummyInterface(config, **kwargs)

    else:
        raise ValueError(f"Unknown interface type: {interface_type}")


class DummyInterface(RobotInterface):
    """
    Dummy interface for testing without actual robot connection.

    Simulates robot state and accepts commands without execution.
    """

    def __init__(self, config: InterfaceConfig, **kwargs):
        super().__init__(config)
        import numpy as np
        self._state = np.zeros(config.obs_dim, dtype=np.float32)
        self._velocity = np.zeros(config.obs_dim, dtype=np.float32)
        self._action = np.zeros(config.action_dim, dtype=np.float32)
        self._trajectory = np.zeros(
            (config.pred_horizon, config.action_dim), dtype=np.float32
        )
        self._command_count = 0

    def connect(self) -> bool:
        self._connected = True
        logger.info("[Dummy Interface] Connected (simulation mode)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[Dummy Interface] Disconnected")

    def get_state(self):
        import numpy as np
        import time
        from src.interfaces.base import RobotState

        if not self._connected:
            return RobotState(qpos=np.zeros(self.config.obs_dim), is_valid=False)

        # Simulate slight state changes
        self._state += np.random.randn(self.config.obs_dim).astype(np.float32) * 0.001

        return RobotState(
            qpos=self._state.copy(),
            qvel=self._velocity.copy(),
            timestamp=time.time(),
            is_valid=True
        )

    def send_action(self, action) -> bool:
        import numpy as np
        if not self._connected:
            return False

        self._action = np.array(action, dtype=np.float32)
        self._command_count += 1

        # Simulate state moving towards action
        self._state = 0.9 * self._state + 0.1 * self._action[:self.config.obs_dim]
        return True

    def send_trajectory(self, trajectory) -> bool:
        import numpy as np
        if not self._connected:
            return False

        self._trajectory = np.array(trajectory, dtype=np.float32)
        self._command_count += 1
        return True

    def get_status(self) -> dict:
        return {
            'type': 'dummy',
            'connected': self._connected,
            'command_count': self._command_count,
        }


def get_available_interfaces() -> list:
    """Get list of available interface types."""
    available = [InterfaceType.DUMMY, InterfaceType.SHARED_MEMORY]

    try:
        import grpc
        available.append(InterfaceType.GRPC)
    except ImportError:
        pass

    return available


def print_interface_info():
    """Print information about available interfaces."""
    print("=" * 60)
    print("  Available Robot Interfaces")
    print("=" * 60)

    interfaces = [
        {
            'name': 'SharedMemory',
            'type': 'shm',
            'latency': '~1μs',
            'network': 'No (same PC)',
            'deps': 'None',
        },
        {
            'name': 'gRPC',
            'type': 'grpc',
            'latency': '~1-10ms',
            'network': 'Yes',
            'deps': 'grpcio, grpcio-tools',
        },
        {
            'name': 'Dummy',
            'type': 'dummy',
            'latency': 'N/A',
            'network': 'N/A',
            'deps': 'None',
        },
    ]

    for iface in interfaces:
        available = iface['type'] in [i.value for i in get_available_interfaces()]
        status = "✓ Available" if available else "✗ Not installed"
        print(f"\n{iface['name']} ({iface['type']})")
        print(f"  Latency: {iface['latency']}")
        print(f"  Network: {iface['network']}")
        print(f"  Dependencies: {iface['deps']}")
        print(f"  Status: {status}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_interface_info()
