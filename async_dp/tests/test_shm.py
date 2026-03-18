"""
Tests for SharedMemoryManager
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.shared_mem import SharedMemoryManager, SharedMemoryError
from config.settings import Config


def test_shm_create_and_close():
    """Test creating and closing shared memory"""
    sm = SharedMemoryManager(create=True, name="test_shm_1")
    assert sm.shm is not None
    assert sm.action_traj.shape == (Config.PRED_HORIZON, Config.ACTION_DIM)
    assert sm.obs_state.shape == (Config.OBS_DIM,)
    sm.close()
    assert sm.shm is None
    print("[Test] SHM create and close: PASSED")


def test_shm_data_persistence():
    """Test that data persists in shared memory"""
    # Create and write data
    sm1 = SharedMemoryManager(create=True, name="test_shm_2")
    sm1.action_traj[0, 0] = 1.234
    sm1.obs_state[0] = 5.678
    sm1.update_counter[0] = 42

    # Attach and read data
    sm2 = SharedMemoryManager(create=False, name="test_shm_2")
    assert np.isclose(sm2.action_traj[0, 0], 1.234)
    assert np.isclose(sm2.obs_state[0], 5.678)
    assert sm2.update_counter[0] == 42

    sm2.close()
    sm1.close()
    print("[Test] SHM data persistence: PASSED")


def test_shm_context_manager():
    """Test using SharedMemoryManager as context manager"""
    with SharedMemoryManager(create=True, name="test_shm_3") as sm:
        sm.action_traj.fill(1.0)
        assert np.all(sm.action_traj == 1.0)
    # Should be closed after exiting context
    print("[Test] SHM context manager: PASSED")


def test_shm_status():
    """Test get_status method"""
    sm = SharedMemoryManager(create=True, name="test_shm_4")
    status = sm.get_status()

    assert status['name'] == "test_shm_4"
    assert status['connected'] is True
    assert status['is_creator'] is True
    assert status['update_counter'] == 0

    sm.close()
    print("[Test] SHM status: PASSED")


def test_shm_cleanup_existing():
    """Test that creating with existing name cleans up properly"""
    # Create first instance
    sm1 = SharedMemoryManager(create=True, name="test_shm_5")
    sm1.action_traj[0, 0] = 999.0
    sm1.close()

    # Create second instance with same name (should cleanup and recreate)
    sm2 = SharedMemoryManager(create=True, name="test_shm_5")
    # Data should be zeroed out after recreation
    assert sm2.action_traj[0, 0] == 0.0
    sm2.close()
    print("[Test] SHM cleanup existing: PASSED")


if __name__ == "__main__":
    test_shm_create_and_close()
    test_shm_data_persistence()
    test_shm_context_manager()
    test_shm_status()
    test_shm_cleanup_existing()
    print("\nAll SHM tests passed!")
