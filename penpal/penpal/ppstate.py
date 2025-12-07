"""State machine for PenPal node."""

from enum import Enum, auto
from threading import Lock

from rclpy.logging import RcutilsLogger


class S(Enum):
    """State for PenPal node."""

    ASLEEP = auto()
    """Not in conversational mode."""
    ASLEEP_IN_USE = auto()
    """In use, cannot be awoken (i.e. due to WriteMessage)."""
    READY_TO_READ = auto()
    """Ready to read new user text on the board."""
    READING = auto()
    """Reading text + generating a response using the VLM."""
    READY_TO_WRITE = auto()
    """Ready to write text to the board once it's reachable."""
    WRITING = auto()
    """Writing to the board."""
    WRITE_COMPLETE = auto()
    """Finished writing to the board."""


class E(Enum):
    """Events driving the penpal node state machine."""

    WAKE = auto()
    """Wake command received."""
    WRITEMESSAGE_CALLED = auto()
    """WriteMessage command used while asleep."""
    SLEEP = auto()
    """Sleep command received."""
    BOARD_VISIBLE = auto()
    """The board has been fully visible (i.e. its pose has been
    published by the BoardDetector) for longer than the threshold
    duration."""
    BOARD_NOT_VISIBLE = auto()
    """The board hasn't been seen (i.e. its pose has not been
    published by the BoardDetector) for longer than the threshold
    duration."""
    BOARD_IN_WORKSPACE = auto()
    """The board has been fully inside the arm's workspace for 
    longer than the threshold duration."""
    WRITE_FAILED = auto()
    """An attempt to write the board failed for some reason."""
    WRITE_INCOMPLETE = auto()
    """Text was left unwritten after a write action."""
    OCR_VLM_TRIGGERED = auto()
    """The OCR node has been told to read the board text & generate
    a response."""
    OCR_VLM_TEXT_RECEIVED = auto()
    """We received text to write from the OCR node."""
    WRITE_STARTED = auto()
    """Started writing to the board."""
    WRITE_SUCCEEDED = auto()
    """Successfully wrote all characters."""


class ConvoFSM:
    """Concurrency-safe state machine for PenPal conversation mode."""

    def __init__(
        self, transition_lock: Lock, logger: RcutilsLogger | None = None
    ) -> None:
        """Initialize the object."""
        self._s = S.ASLEEP
        self._lock = transition_lock
        self._logger = logger

    def is_awake(self) -> bool:
        """Return true if we are in the live conversation loop."""
        return self._s not in [S.ASLEEP, S.ASLEEP_IN_USE]

    def get_state(self) -> S:
        """Get current state of the FSM."""
        return self._s

    def transition(self, e: E) -> S:
        """Given an event, update the state + return the new state."""
        sestr = f'{self._s} ({e})'
        if self._logger is not None:
            self._logger.debug(
                f'Awaiting lock for state transition {sestr}...'
            )
        with self._lock:
            new_s = self._s

            match self._s:
                case S.ASLEEP:
                    if e == E.WAKE:
                        new_s = S.READY_TO_READ
                    if e == E.WRITEMESSAGE_CALLED:
                        new_s = S.ASLEEP_IN_USE
                case S.ASLEEP_IN_USE:
                    if e in [
                        E.WRITE_FAILED,
                        E.WRITE_INCOMPLETE,
                        E.WRITE_SUCCEEDED,
                    ]:
                        new_s = S.ASLEEP
                case S.READY_TO_READ:
                    if e == E.BOARD_VISIBLE:
                        new_s = S.READING
                case S.READING:
                    if e == E.OCR_VLM_TEXT_RECEIVED:
                        new_s = S.READY_TO_WRITE
                case S.READY_TO_WRITE:
                    if e == E.BOARD_IN_WORKSPACE:
                        new_s = S.WRITING
                case S.WRITING:
                    if e == E.WRITE_INCOMPLETE:
                        # for now, just discard the rest of the text
                        # and go to write complete.
                        # but later we can add some states here to
                        # wait for the user to erase the board so we can write
                        # the rest.
                        new_s = S.WRITE_COMPLETE
                    elif e == E.WRITE_FAILED:
                        # try again
                        new_s = S.READY_TO_WRITE
                    elif e == E.WRITE_SUCCEEDED:
                        new_s = S.WRITE_COMPLETE
                case S.WRITE_COMPLETE:
                    # wait for the user to take away the board
                    # before looking for new text
                    if e == E.BOARD_NOT_VISIBLE:
                        new_s = S.READY_TO_READ
                case _:
                    raise NotImplementedError(f'Unrecognized state {self._s}')

            # we are currently allowed to sleep from any state.
            # the robot always stops what it's doing when sleep is called.
            if e == E.SLEEP:
                new_s = S.ASLEEP

            if self._logger is not None and new_s != self._s:
                self._logger.info(f'STATE TRANSITION: {sestr} -> {new_s}')

            self._s = new_s
            return self._s
