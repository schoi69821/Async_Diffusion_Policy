"""
Shared Memory Manager for Inter-Process Communication
Provides thread-safe access to action trajectories and observations
"""
from multiprocessing import shared_memory
import numpy as np
import time
import logging
from config.settings import Config

# Setup logging
logger = logging.getLogger(__name__)


class SharedMemoryError(Exception):
    """Custom exception for shared memory errors"""
    pass


class SharedMemoryManager:
    """
    Manages shared memory for real-time communication between
    Inference (Brain) and Control (Muscle) processes.

    Memory Layout:
        [0:action_size] - action_traj: (PRED_HORIZON, ACTION_DIM) float32
        [action_size:action_size+obs_size] - obs_state: (OBS_DIM,) float32
        [action_size+obs_size:] - update_counter: (1,) int64

    Thread Safety:
        All access should be protected by an external lock.
        This class does NOT provide internal locking.
    """

    # Retry configuration
    MAX_RETRIES = 5
    RETRY_DELAY = 0.1  # seconds

    def __init__(self, create=False, name=None):
        """
        Initialize SharedMemoryManager

        Args:
            create: If True, create new shared memory. If False, attach to existing.
            name: Optional custom name. Defaults to Config.SHM_NAME.
        """
        self.name = name or Config.SHM_NAME
        self.create = create
        self.shm = None
        self.buffer = None

        # Calculate sizes
        self.action_size = 4 * Config.ACTION_DIM * Config.PRED_HORIZON  # float32
        self.obs_size = 4 * Config.OBS_DIM  # float32
        self.counter_size = 8  # int64
        self.total_size = self.action_size + self.obs_size + self.counter_size

        # Initialize
        self._connect()

    def _connect(self):
        """Connect to or create shared memory with retry logic"""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.create:
                    self._create_shm()
                else:
                    self._attach_shm()

                self._map_arrays()
                logger.info(f"[SHM] {'Created' if self.create else 'Attached to'} "
                           f"shared memory '{self.name}' (size: {self.total_size} bytes)")
                return

            except FileExistsError:
                # Shared memory already exists, try to clean up and recreate
                if self.create:
                    logger.warning(f"[SHM] Shared memory '{self.name}' exists. Cleaning up...")
                    self._cleanup_existing()
                    continue
                else:
                    raise

            except FileNotFoundError:
                # Shared memory doesn't exist yet
                if not self.create:
                    logger.warning(f"[SHM] Shared memory '{self.name}' not found. "
                                  f"Retry {attempt + 1}/{self.MAX_RETRIES}")
                    time.sleep(self.RETRY_DELAY)
                    last_error = FileNotFoundError(f"Shared memory '{self.name}' not found")
                    continue
                else:
                    raise

            except Exception as e:
                last_error = e
                logger.error(f"[SHM] Error on attempt {attempt + 1}: {e}")
                time.sleep(self.RETRY_DELAY)
                continue

        # All retries exhausted
        raise SharedMemoryError(
            f"Failed to {'create' if self.create else 'attach to'} "
            f"shared memory '{self.name}' after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _create_shm(self):
        """Create new shared memory segment"""
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.total_size
            )
        except FileExistsError:
            # Try to unlink and recreate
            self._cleanup_existing()
            self.shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.total_size
            )

    def _attach_shm(self):
        """Attach to existing shared memory segment"""
        self.shm = shared_memory.SharedMemory(name=self.name)

        # Validate size
        if self.shm.size < self.total_size:
            raise SharedMemoryError(
                f"Shared memory size mismatch. Expected >= {self.total_size}, "
                f"got {self.shm.size}"
            )

    def _cleanup_existing(self):
        """Clean up existing shared memory with same name"""
        try:
            existing = shared_memory.SharedMemory(name=self.name)
            existing.close()
            existing.unlink()
            logger.info(f"[SHM] Cleaned up existing shared memory '{self.name}'")
            time.sleep(0.05)  # Brief delay after cleanup
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"[SHM] Cleanup warning: {e}")

    def _map_arrays(self):
        """Map numpy arrays to shared memory buffer"""
        self.buffer = self.shm.buf

        # action_traj: (PRED_HORIZON, ACTION_DIM) at offset 0
        self.action_traj = np.ndarray(
            (Config.PRED_HORIZON, Config.ACTION_DIM),
            dtype=np.float32,
            buffer=self.buffer,
            offset=0
        )

        # obs_state: (OBS_DIM,) after action_traj
        self.obs_state = np.ndarray(
            (Config.OBS_DIM,),
            dtype=np.float32,
            buffer=self.buffer,
            offset=self.action_size
        )

        # update_counter: (1,) at the end
        self.update_counter = np.ndarray(
            (1,),
            dtype=np.int64,
            buffer=self.buffer,
            offset=self.action_size + self.obs_size
        )

        # Initialize to zeros if creating
        if self.create:
            self.action_traj.fill(0)
            self.obs_state.fill(0)
            self.update_counter.fill(0)

    def close(self):
        """Close and optionally unlink shared memory"""
        if self.shm is None:
            return

        try:
            self.shm.close()
            if self.create:
                try:
                    self.shm.unlink()
                    logger.info(f"[SHM] Unlinked shared memory '{self.name}'")
                except FileNotFoundError:
                    pass
        except Exception as e:
            logger.warning(f"[SHM] Error closing shared memory: {e}")
        finally:
            self.shm = None
            self.buffer = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def get_status(self):
        """Get current shared memory status"""
        return {
            'name': self.name,
            'size': self.total_size,
            'connected': self.shm is not None,
            'is_creator': self.create,
            'update_counter': int(self.update_counter[0]) if self.shm else None
        }

    def __repr__(self):
        status = "connected" if self.shm else "disconnected"
        mode = "creator" if self.create else "client"
        return f"SharedMemoryManager(name='{self.name}', {status}, {mode})"
