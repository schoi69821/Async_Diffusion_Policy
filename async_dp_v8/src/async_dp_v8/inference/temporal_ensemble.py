"""Temporal ensemble for averaging overlapping chunk predictions."""
import numpy as np
from collections import deque
from typing import Optional


class TemporalEnsemble:
    def __init__(self, pred_horizon: int = 12, execute_horizon: int = 4):
        self.pred_horizon = pred_horizon
        self.execute_horizon = execute_horizon
        self.buffer: deque = deque()
        self.weights: deque = deque()

    def add_prediction(self, chunk: np.ndarray):
        """Add a new predicted chunk. chunk: [H, A]"""
        self.buffer.append(chunk.copy())
        # Newer predictions weighted more
        weight = np.exp(-0.3 * np.arange(len(chunk)))
        self.weights.append(weight)

        # Remove old predictions that no longer overlap
        max_history = self.pred_horizon // self.execute_horizon + 1
        while len(self.buffer) > max_history:
            self.buffer.popleft()
            self.weights.popleft()

    def get_actions(self, num_actions: int = None) -> np.ndarray:
        """Get averaged actions for next execution window."""
        if num_actions is None:
            num_actions = self.execute_horizon

        if len(self.buffer) == 0:
            raise ValueError("No predictions in buffer")

        if len(self.buffer) == 1:
            return self.buffer[-1][:num_actions]

        # Average overlapping predictions
        latest = self.buffer[-1]
        A = latest.shape[-1]
        result = np.zeros((num_actions, A))
        total_weight = np.zeros(num_actions)

        for i, (chunk, w) in enumerate(zip(self.buffer, self.weights)):
            offset = (len(self.buffer) - 1 - i) * self.execute_horizon
            for t in range(num_actions):
                src_t = t + offset
                if 0 <= src_t < len(chunk):
                    result[t] += chunk[src_t] * w[src_t]
                    total_weight[t] += w[src_t]

        total_weight = np.maximum(total_weight, 1e-8)
        result /= total_weight[:, None]
        return result

    def reset(self):
        self.buffer.clear()
        self.weights.clear()
