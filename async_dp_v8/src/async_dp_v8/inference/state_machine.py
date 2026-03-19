"""Inference state machine for v8 policy deployment."""
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class State(Enum):
    BOOT = auto()
    SAFE_HOME = auto()
    REACH = auto()
    ALIGN = auto()
    CLOSE_ARMED = auto()
    CLOSE_COMMIT = auto()
    LIFT = auto()
    PLACE = auto()
    RELEASE = auto()
    RETURN = auto()
    DONE = auto()
    RECOVERY = auto()
    E_STOP = auto()


@dataclass
class RuntimeContext:
    state: State = State.BOOT
    close_count: int = 0
    open_count: int = 0
    has_object: bool = False
    last_phase: int = 0
    timeout_steps: int = 0
    total_steps: int = 0
    state_entry_step: int = 0

    def transition(self, new_state: State):
        logger.info(f"State transition: {self.state.name} -> {new_state.name} (step={self.total_steps})")
        self.state = new_state
        self.state_entry_step = self.total_steps
        self.timeout_steps = 0

    def steps_in_state(self) -> int:
        return self.total_steps - self.state_entry_step
