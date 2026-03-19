"""Test inference state machine."""
from async_dp_v8.inference.state_machine import State, RuntimeContext


def test_initial_state():
    ctx = RuntimeContext()
    assert ctx.state == State.BOOT


def test_transition():
    ctx = RuntimeContext()
    ctx.total_steps = 5
    ctx.transition(State.REACH)
    assert ctx.state == State.REACH
    assert ctx.state_entry_step == 5


def test_steps_in_state():
    ctx = RuntimeContext()
    ctx.total_steps = 10
    ctx.transition(State.ALIGN)
    ctx.total_steps = 15
    assert ctx.steps_in_state() == 5


def test_state_flow():
    ctx = RuntimeContext()
    ctx.transition(State.SAFE_HOME)
    ctx.transition(State.REACH)
    ctx.transition(State.ALIGN)
    ctx.transition(State.CLOSE_ARMED)
    ctx.transition(State.CLOSE_COMMIT)
    ctx.transition(State.LIFT)
    ctx.transition(State.PLACE)
    ctx.transition(State.RELEASE)
    ctx.transition(State.RETURN)
    ctx.transition(State.DONE)
    assert ctx.state == State.DONE
