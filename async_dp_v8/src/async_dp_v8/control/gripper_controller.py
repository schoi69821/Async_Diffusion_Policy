"""Gripper controller with current-based position control."""
import numpy as np
from typing import Optional

from async_dp_v8.constants import (
    FOLLOWER_GRIP_MIN, FOLLOWER_GRIP_MAX,
    GRIP_OPEN, GRIP_HOLD, GRIP_CLOSE,
)


class GripperController:
    def __init__(
        self,
        grip_min: int = FOLLOWER_GRIP_MIN,
        grip_max: int = FOLLOWER_GRIP_MAX,
        close_current_limit: int = 150,
    ):
        self.grip_min = grip_min
        self.grip_max = grip_max
        self.close_current_limit = close_current_limit
        self._state = "open"

    def command(self, token: int) -> dict:
        """Convert grip token to motor command.

        token: 0=open, 1=hold, 2=close
        Returns: dict with position and current limit
        """
        if token == GRIP_OPEN:
            self._state = "open"
            return {
                "position": self.grip_max,
                "current_limit": None,
            }
        elif token == GRIP_CLOSE:
            self._state = "closed"
            return {
                "position": self.grip_min,
                "current_limit": self.close_current_limit,
            }
        else:  # GRIP_HOLD
            if self._state == "closed":
                return {
                    "position": self.grip_min,
                    "current_limit": self.close_current_limit,
                }
            return {
                "position": self.grip_max,
                "current_limit": None,
            }

    @property
    def is_closed(self) -> bool:
        return self._state == "closed"

    def reset(self):
        self._state = "open"
