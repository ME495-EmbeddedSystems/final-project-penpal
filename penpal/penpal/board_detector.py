"""Detect pose+dimensions of the whiteboard(s)."""

from rclpy import Node


class BoardDetector:
    """Detects pose+dimensions of the whiteboard(s) using the camera."""

    def __init__(self, node: Node) -> None:
        """Initialize the object."""
        self._node = node
