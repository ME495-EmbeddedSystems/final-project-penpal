"""Locate & orient the whiteboard(s)."""

from rclpy import Node


class BoardDetector:
    """Locates & orients the whiteboards in space."""

    def __init__(self, node: Node) -> None:
        """Initialize the object."""
        self._node = node
