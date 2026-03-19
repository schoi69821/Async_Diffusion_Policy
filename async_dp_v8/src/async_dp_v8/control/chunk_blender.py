"""Chunk blender for smooth action transitions between inference steps."""
import numpy as np
from typing import Optional


class ChunkBlender:
    def __init__(self, execute_horizon: int = 4, alpha: float = 0.6):
        self.execute_horizon = execute_horizon
        self.alpha = alpha
        self.prev_tail: Optional[np.ndarray] = None

    def blend(self, new_chunk: np.ndarray) -> np.ndarray:
        """Blend new chunk with tail of previous chunk.

        new_chunk: [H, A] new predicted actions
        Returns: [H, A] blended actions
        """
        if self.prev_tail is None:
            blended = new_chunk.copy()
        else:
            k = min(len(self.prev_tail), len(new_chunk))
            blended = new_chunk.copy()
            blended[:k] = self.alpha * self.prev_tail[:k] + (1 - self.alpha) * blended[:k]

        self.prev_tail = blended[self.execute_horizon:].copy()
        return blended

    def reset(self):
        self.prev_tail = None
