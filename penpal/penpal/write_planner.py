"""Plans trajectories to write characters."""

from dataclasses import dataclass

import numpy as np
from penpal.board_detector import BoardDetector
from penpal.control.pp_control import PPControlBase
from rclpy.node import Node


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


class WritePlanner:
    """Compute trajectories to write on the real board."""

    def __init__(
        self, node: Node, controller: PPControlBase, board: BoardDetector
    ) -> None:
        """Initialize the object."""
        self.control = controller
        self.board = board

    def write_characters(self, characters: list[Character]) -> None:
        """
        Write a list of characters to the board.

        Creates newlines when necessary.

        Args:
            characters (list[Character]): list of characters to write.

        """
        pass
