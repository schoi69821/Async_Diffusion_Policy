"""
Tests for Robot Communication Interfaces
"""
import numpy as np
import sys
import os
import time
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interfaces.base import InterfaceConfig, RobotState
from src.interfaces.factory import create_interface, InterfaceType, get_available_interfaces


def test_interface_config():
    """Test interface configuration defaults"""
    config = InterfaceConfig()

    assert config.action_dim == 14
    assert config.obs_dim == 14
    assert config.pred_horizon == 16
    assert config.grpc_port == 50051

    print("[Test] Interface config: PASSED")


def test_dummy_interface():
    """Test dummy interface for basic operations"""
    config = InterfaceConfig()
    interface = create_interface('dummy', config)

    # Test connection
    assert interface.connect() == True
    assert interface.is_connected == True

    # Test get state
    state = interface.get_state()
    assert isinstance(state, RobotState)
    assert state.qpos.shape == (config.obs_dim,)
    assert state.is_valid == True

    # Test send action
    action = np.random.randn(config.action_dim).astype(np.float32)
    assert interface.send_action(action) == True

    # Test send trajectory
    trajectory = np.random.randn(config.pred_horizon, config.action_dim).astype(np.float32)
    assert interface.send_trajectory(trajectory) == True

    # Test status
    status = interface.get_status()
    assert status['type'] == 'dummy'
    assert status['connected'] == True
    assert status['command_count'] == 2

    # Test disconnect
    interface.disconnect()
    assert interface.is_connected == False

    print("[Test] Dummy interface: PASSED")


def test_shm_interface_server_client():
    """Test SharedMemory interface server-client communication"""
    config = InterfaceConfig(shm_name="test_robot_interface")

    # Create server (robot controller side)
    from src.interfaces.shm_interface import SharedMemoryInterface
    server = SharedMemoryInterface(config, is_server=True)
    assert server.connect() == True

    # Create client (Async DP side)
    client = SharedMemoryInterface(config, is_server=False)
    assert client.connect() == True

    # Server updates state
    test_qpos = np.array([0.1, 0.2, 0.3] + [0.0] * 11, dtype=np.float32)
    server.update_state(test_qpos)

    # Client reads state
    state = client.get_state()
    assert state.is_valid == True
    np.testing.assert_array_almost_equal(state.qpos[:3], test_qpos[:3], decimal=5)

    # Client sends action
    test_action = np.array([1.0, 2.0, 3.0] + [0.0] * 11, dtype=np.float32)
    assert client.send_action(test_action) == True

    # Server reads action
    received_action = server.get_action()
    np.testing.assert_array_almost_equal(received_action[:3], test_action[:3], decimal=5)

    # Client sends trajectory
    test_traj = np.random.randn(config.pred_horizon, config.action_dim).astype(np.float32)
    assert client.send_trajectory(test_traj) == True

    # Server reads trajectory
    received_traj = server.get_trajectory()
    np.testing.assert_array_almost_equal(received_traj, test_traj, decimal=5)

    # Cleanup
    client.disconnect()
    server.disconnect()

    print("[Test] SharedMemory interface server-client: PASSED")


def test_shm_interface_context_manager():
    """Test SharedMemory interface with context manager"""
    config = InterfaceConfig(shm_name="test_robot_interface_ctx")

    from src.interfaces.shm_interface import SharedMemoryInterface

    with SharedMemoryInterface(config, is_server=True) as server:
        assert server.is_connected == True

        with SharedMemoryInterface(config, is_server=False) as client:
            assert client.is_connected == True

            # Basic communication test
            server.update_state(np.ones(config.obs_dim, dtype=np.float32))
            state = client.get_state()
            assert state.qpos[0] == 1.0

    print("[Test] SharedMemory context manager: PASSED")


def test_shm_interface_concurrent_access():
    """Test SharedMemory interface with concurrent access"""
    config = InterfaceConfig(shm_name="test_robot_interface_concurrent")

    from src.interfaces.shm_interface import SharedMemoryInterface

    server = SharedMemoryInterface(config, is_server=True)
    server.connect()

    client = SharedMemoryInterface(config, is_server=False)
    client.connect()

    errors = []
    iterations = 100

    def writer_thread():
        try:
            for i in range(iterations):
                action = np.full(config.action_dim, float(i), dtype=np.float32)
                client.send_action(action)
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Writer error: {e}")

    def reader_thread():
        try:
            for _ in range(iterations):
                action = server.get_action()
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Reader error: {e}")

    writer = threading.Thread(target=writer_thread)
    reader = threading.Thread(target=reader_thread)

    writer.start()
    reader.start()

    writer.join()
    reader.join()

    client.disconnect()
    server.disconnect()

    assert len(errors) == 0, f"Concurrent access errors: {errors}"

    print("[Test] SharedMemory concurrent access: PASSED")


def test_interface_factory():
    """Test interface factory function"""
    config = InterfaceConfig()

    # Test dummy interface creation
    dummy = create_interface('dummy', config)
    assert dummy is not None

    # Test SharedMemory interface creation
    shm = create_interface('shm', config, is_server=True)
    assert shm is not None
    shm.connect()
    shm.disconnect()

    # Test available interfaces
    available = get_available_interfaces()
    assert InterfaceType.DUMMY in available
    assert InterfaceType.SHARED_MEMORY in available

    print("[Test] Interface factory: PASSED")


def test_robot_state_dataclass():
    """Test RobotState dataclass"""
    qpos = np.array([1.0, 2.0, 3.0])
    qvel = np.array([0.1, 0.2, 0.3])

    state = RobotState(
        qpos=qpos,
        qvel=qvel,
        timestamp=time.time(),
        is_valid=True
    )

    np.testing.assert_array_equal(state.qpos, qpos)
    np.testing.assert_array_equal(state.qvel, qvel)
    assert state.is_valid == True

    print("[Test] RobotState dataclass: PASSED")


if __name__ == "__main__":
    test_interface_config()
    test_robot_state_dataclass()
    test_dummy_interface()
    test_interface_factory()
    test_shm_interface_server_client()
    test_shm_interface_context_manager()
    test_shm_interface_concurrent_access()
    print("\nAll interface tests passed!")
