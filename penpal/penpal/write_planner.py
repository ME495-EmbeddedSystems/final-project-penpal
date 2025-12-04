"""Plans trajectories to write characters."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node

from penpal.control.pp_control import PPControlBase, Trajectory


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
    each point is [x_mm, y_mm, z_mm]
    where x_mm, y_mm is position in virtual board in millimeters
    z in [0, 1] where:
        - 0 = off the board (no pressure)
        - (0, 1] = pressure, with 1 being hardest and epsilon being softest.
    """


@dataclass
class BoardInfo:
    """Important info about the whiteboard for writing on it."""

    pos: np.ndarray
    """Board origin (top left corner) position [x,y,z] in world frame"""

    ori: R
    """Board orientation in world frame."""

    # these shouldn't change
    width_m: float
    height_m: float

    writeable_area: np.ndarray
    """
    Rectangular region available for writing, relative to the board origin.
    [[x_tl, y_tl], [x_br, y_br]].
    Note that in board coordinates, +x is to the right and +y is down.
    """


class WritePlanner:
    """Compute trajectories to write on the real board."""

    @dataclass
    class Config:
        """Configuration for this class."""

        traj_len: int = 10
        """Max length of trajectory to write at a time."""

        ee_velocity_m_s: float = 0.02
        max_force_N: float = 1.0

    def __init__(
        self, node: Node, controller: PPControlBase, cfg: Config | None = None
    ) -> None:
        """Initialize the object."""
        self.control = controller
        self._world_frame_name = 'base'  # todo correct this if needed
        self.c = cfg if cfg is not None else WritePlanner.Config()
        self._node = node

        # TODO - subscribe to BoardDetector topics

    async def write_characters(self, characters: list[Character]) -> None:
        """
        Write a list of characters to the board.

        Creates newlines when necessary.

        Args:
            characters (list[Character]): list of characters to write.

        """
        # create a 3D plan for writing the characters in board frame.
        trajs = self._plan_path_in_board_frame(characters)

        # in order to ensure responsiveness to board pose updates,
        # each character's trajectory is split into several to be
        # passed into the controller.
        short_trajs: list[Trajectory] = []
        for traj in trajs:
            short_trajs.extend(traj.split_with_len(self.c.traj_len))

        # write the trajectories to the board,
        # transforming each into world frame as its time comes.
        for traj in short_trajs:
            board = self.get_latest_board_info()
            world_traj = traj.transform(-board.pos, board.ori.inv())
            await self.control.execute_trajectory(
                world_traj, self.c.ee_velocity_m_s
            )

    def _plan_path_in_board_frame(
        self, cs: list[Character]
    ) -> list[Trajectory]:
        """
        Plan a complete trajectory for the pen tip expressed in board frame.

        This means:
        - placing the text correctly on the empty space in the board
        - inserting connecting points in the spaces between characters
        - inserting newlines where appropriate

        Args:
            cs: list of characters to write.

        Returns:
            list[Trajectory]: ordered list of trajectories, one for
                              each character.

        """
        # todo actually implement this
        # for now, to get the integration test running, just return
        # the character trajectories unmodified

        trajs = []
        upright = R.from_euler('xyz', [0, np.pi / 2, 0])
        up_q = upright.as_quat(True)

        for char in cs:
            data = np.zeros(shape=(char.trajectory.shape[0], 8))
            data[:, 0:2] = char.trajectory[:, 0:2] / 1000.0
            data[:, 3:7] = up_q[np.newaxis, :]
            data[:, 7] = char.trajectory[:, 2] * self.c.max_force_N
            traj = Trajectory(char.char, data)
            trajs.append(traj)

        return trajs

    def get_latest_board_info(self) -> BoardInfo:
        """Return the most recently update board location + dimensions."""
        # todo - grab this from the BoardDetector topics.
        raise NotImplementedError
