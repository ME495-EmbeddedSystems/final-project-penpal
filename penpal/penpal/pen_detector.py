"""Detect pose+dimensions of the pen."""

from rclpy.node import Node


class PenDetector:
    """Detects pose+dimensions of the pen."""

    def __init__(self, node: Node) -> None:
        """Initialize the object."""
        self._node = node
