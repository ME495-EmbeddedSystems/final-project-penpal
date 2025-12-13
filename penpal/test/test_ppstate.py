"""Tests for ppstate.py."""

from threading import Lock

from penpal import ppstate


def test_fsm_basic():
    """
    Test basic funcionality for fsm class.

    (without testing internal logic cuz that makes a britle test.)
    """
    lock = Lock()
    fsm = ppstate.ConvoFSM(lock)

    assert fsm.get_state() == ppstate.S.STARTUP
    assert not fsm.is_awake() == ppstate.S.ASLEEP

    # try a transition to make sure it doesn't crash
    s = fsm.transition(ppstate.E.BOARD_IN_WORKSPACE)
    assert type(s) is ppstate.S
