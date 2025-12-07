"""Grab bag of useful utilities."""

from threading import Lock


class LockedString:
    """String that can only be written with a lock."""

    def __init__(self) -> None:
        """Initialize the object."""
        self._lock = Lock()
        self._text: str | None = None

    @property
    def text(self) -> str | None:
        """Read the string."""
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        with self._lock:
            self._text = text
