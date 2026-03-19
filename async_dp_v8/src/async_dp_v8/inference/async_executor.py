"""Async executor for separating inference and control loops."""
import threading
import time
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """Runs inference in a separate thread while control loop executes actions."""

    def __init__(self, policy_runner, control_freq: int = 500, inference_freq: int = 15):
        self.policy_runner = policy_runner
        self.control_freq = control_freq
        self.inference_freq = inference_freq

        self._current_actions: Optional[np.ndarray] = None
        self._action_index: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._last_info: Dict = {}

    def start(self):
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        logger.info("AsyncExecutor started")

    def stop(self):
        self._running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
        logger.info("AsyncExecutor stopped")

    def get_action(self) -> Optional[np.ndarray]:
        """Get next action for control loop. Called at control_freq."""
        with self._lock:
            if self._current_actions is None:
                return None
            if self._action_index >= len(self._current_actions):
                return self._current_actions[-1]  # Hold last action
            action = self._current_actions[self._action_index]
            self._action_index += 1
            return action

    def _inference_loop(self):
        dt = 1.0 / self.inference_freq
        while self._running:
            t0 = time.monotonic()
            try:
                obs = self.policy_runner.robot.get_observation()
                # Convert to batch format (would need proper tensor conversion)
                cmd, info = self.policy_runner.step(obs)
                self._last_info = info

                with self._lock:
                    if cmd.get("arm_actions") is not None:
                        self._current_actions = cmd["arm_actions"]
                        self._action_index = 0
            except Exception as e:
                logger.error(f"Inference error: {e}")

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def info(self) -> Dict:
        return self._last_info
