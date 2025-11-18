"""Grabs the pen."""

from dataclasses import dataclass

import numpy as np
from penpal.penpal.pen_detector import PenDetector
from penpal.penpal.control.pp_control import PPControlBase
from rclpy import Node


@dataclass
class Character:
    """Represents a set of strokes for a single character."""

    char: str
    """Actual UTF character represented by this trajectory"""

    trajectory: np.ndarray
    """
    Trajectory for this character.

    N points, each point in R3
    Nx3 array
    each point is [x, y, z]
    where x is position in virtual board
    z in [0, 1] where: 
    - 0 = off the board (no pressure)
    - (0, 1] = pressure, with 1 being hardest and epsilon being softest.
    """


class GrabPlanner:
    """Compute trajectories to write on the real board."""

    def __init__(self, node: Node, controller: PPControlBase, pen: PenDetector) -> None:
        """Initialize the object."""
        self.control = controller
        self.pen = pen

    def grab_pen(self) -> None:
        """Grab the pen (must be visible to camera)."""
        pass
