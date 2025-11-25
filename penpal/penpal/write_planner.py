"""Plans trajectories to write characters."""

from dataclasses import dataclass

import numpy as np

from penpal.board_detector import BoardDetector
from penpal.control.pp_control import PPControlBase, Trajectory

from rclpy.node import Node

from scipy.spatial.transform import Rotation as R


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

    def __init__(self, char, trajectory):
        """Initalize property values."""
        self.char = char
        self.trajectory = trajectory

    @property
    def width_m(self) -> float:
        """Return width of the character."""
        return np.max(self.trajectory[:, 0]) - np.min(self.trajectory[:, 0])

    @property
    def height_m(self) -> float:
        """Return height of the character."""
        return np.max(self.trajectory[:, 1]) - np.min(self.trajectory[:, 1])


class WritePlanner:
    """Compute trajectories to write on the real board."""

    def __init__(
        self,
        node: Node,
        controller: PPControlBase,
        board: BoardDetector
    ) -> None:
        """
        Initialize the object.

        Args:
            board_center: [x,y,z]
            board_orientation: [qx,qy,qz,qw]
        """
        self.node = node
        self.control = controller
        self.board = board

        self.board_width = 0.7  # in meters
        self.board_height = 0.5  # in meters
        self.x = 0
        self.y = 0
        self.line_height = 0.1
        self.line_space = 0.01
        self.scale = 1.0

    def write_characters(self, characters:list[Character]) -> None:
        """
        Write a list of characters to the board.

        Creates newlines when necessary.

        Args:
            characters (list[Character]): list of characters to write.

        """
        board_center = self.board.center
        board_quat = self.board.quat
        board_rot = R.from_quat(board_quat)
        final_traj = []
        for i, char in enumerate(characters):
            points = char.trajectory
            char_width = char.width_m
            x_offset = - np.min(points[:, 0])
            if char_width > self.board_width - self.x:
                self.y += self.line_height
                self.x = 0

            new_point = []
            for point in points:
                local_x = point[0] + self.x + x_offset
                local_y = point[1] + self.y
                local_z = point[2]

                local_vector = np.array([local_x, local_y, local_z])
                offset = board_rot.apply(local_vector)

                new_x = board_center[0] + offset[0]
                new_y = board_center[1] + offset[1]
                new_z = board_center[2] + offset[2]

                new_point.append([new_x, new_y, new_z,
                                  board_quat[0], board_quat[1],
                                  board_quat[2], board_quat[3]
                                  ])

            label = f'{char.char}_{i}'
            new_traj = Trajectory(label, np.array(new_point))
            final_traj.append(new_traj)
            self.x += char_width + self.line_space
        return final_traj
