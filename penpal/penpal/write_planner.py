"""Plans trajectories to write characters."""

from dataclasses import dataclass, field

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

    font_size_mm: float
    """Font size (aka max character height) in mm"""

    def __post_init__(self) -> None:
        """Post-initialization for dataclass."""
        # input checking
        if self.trajectory.shape[1] != 3:
            raise ValueError(
                f'Invalid trajectory shape {self.trajectory.shape}'
            )

    def get_bounding_box_mm(self) -> np.ndarray:
        """
        Return the bounding rectangle around the character in mm.

        Note: somewhat expensive to calculate. get from font_trajectory if this is
        too slow for you.

        Returns:
            np.ndarray: [TL, BR] where each is [x, y]

        """
        xmax, ymax = self.trajectory[:, :2].max(axis=0)
        xmin, ymin = self.trajectory[:, :2].min(axis=0)
        TL = [xmin, ymax]
        BR = [xmax, ymin]
        return np.array([TL, BR])


@dataclass()
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
    Note that board coordinates are in R2 with +x = right, -y = down
    """

    T_sb: np.ndarray = field(init=False)
    """Transformation matrix from world to board frame."""

    T_bs: np.ndarray = field(init=False)
    """Transformation matrix from board to world frame."""

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        if self.writeable_area.shape != (2, 2):
            raise ValueError(
                'Incorrect shape for writeable area: '
                f'{self.writeable_area.shape}'
            )
        writeable_dims = self.writeable_area[1, :] - self.writeable_area[0, :]
        if writeable_dims[0] <= 0 or writeable_dims[1] >= 0:
            raise ValueError('Writeable area is negative.')

        T_sb = np.empty((4, 4))
        T_sb[0:3, 0:3] = self.ori.as_matrix()
        T_sb[0:3, 3] = self.pos
        T_sb[3, :] = [0, 0, 0, 1]
        self.T_sb = T_sb
        self.T_bs = np.linalg.inv(self.T_sb)

    def get_board_corners_world_frame(self) -> np.ndarray:
        """
        Get the 4 corners of the board in world frame.

        Returns:
            np.ndarray: [[TL], [TR], [BR], [BL]] with each being [x,y,z]

        """
        # corners as homogenous coordinate column vectors
        corners = np.array(
            [
                [0, 0, 0, 1],
                [self.width_m, 0, 0, 1],
                [self.width_m, -self.height_m, 0, 1],
                [0, -self.height_m, 0, 1],
            ]
        ).T

        corners_s = self.T_sb @ corners
        # remove extra 1's and transpose to return R3 row vectors.
        return corners_s[0:3, :].T

    def get_writeable_area_corners_world_frame(self) -> np.ndarray:
        """
        Get the 4 corners of the writeable area in world frame.

        Returns:
            np.ndarray: [[TL], [TR], [BR], [BL]] with each being [x,y,z]

        """
        # corners as homogenous coordinate column vectors
        TL = [*self.writeable_area[0, :], 0, 1]
        TR = [self.writeable_area[1, 0], self.writeable_area[0, 1], 0, 1]
        BR = [*self.writeable_area[1, :], 0, 1]
        BL = [self.writeable_area[0, 0], self.writeable_area[1, 1], 0, 1]
        corners = np.array([TL, TR, BR, BL]).T

        corners_s = self.T_sb @ corners
        # remove extra 1's and transpose to return R3 row vectors.
        return corners_s[0:3, :].T


class WritePlanner:
    """Compute trajectories to write on the real board."""

    DOWN_Q = R.from_euler('xyz', [0, np.pi / 2, 0]).as_quat(True)
    """Quaternion orientation pointing straight down."""

    @dataclass
    class Config:
        """Configuration for this class."""

        traj_len: int = 10
        """Max length of trajectory to write at a time."""
        ee_velocity_m_s: float = 0.02
        """End-effector forward velocity while writing."""
        max_force_N: float = 1.0
        """Maximum pressure to apply into the board."""
        off_board_height_m: float = 0.03
        """Distance to lift the pen off the board when moving between chars."""
        pen_lift_thresh_N: float = 1e-4
        """Force threshold below which the pen is lifted off the board."""

    def __init__(
        self, node: Node, controller: PPControlBase, cfg: Config | None = None
    ) -> None:
        """Initialize the object."""
        self.control = controller
        self._world_frame_name = 'base'  # todo correct this if needed
        self.c = cfg if cfg is not None else WritePlanner.Config()
        self._node = node
        self._logger = node.get_logger().get_child('WritePlanner')

        # TODO - subscribe to BoardDetector topics

    async def write_characters(
        self,
        characters: list[Character],
        line_spacing_factor: float,
    ) -> list[Character]:
        """
        Write a list of characters to the board.

        Creates newlines when necessary.

        Args:
            characters (list[Character]): list of characters to write.
            line_spacing_factor (float): space between lines
                represented as a fraction of line height

        Returns:
            list[Character]: characters which did not get written due
                to running out of room, if any.

        """
        # create a 3D plan for writing the characters in board frame.
        trajs, leftovers = self._plan_path_in_board_frame(
            characters, line_spacing_factor
        )

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
            world_traj = traj.transform(board.pos, board.ori)
            await self.control.execute_trajectory(
                world_traj, self.c.ee_velocity_m_s
            )

        return leftovers

    def _plan_path_in_board_frame(
        self,
        cs: list[Character],
        line_spacing_factor: float,
        padding_mm: float = 2.0,
    ) -> tuple[list[Trajectory], list[Character]]:
        """
        Plan a complete trajectory for the pen tip expressed in board frame.

        This means:
        - placing the text correctly on the empty space in the board
        - inserting connecting paths in the spaces between characters
        - inserting newlines where appropriate

        Args:
            cs: list of characters to write.
            line_spacing_factor: space between lines as a fraction of line height
            padding_mm: padding within the bounding box, css style

        Returns:
            list[Trajectory]: ordered list of trajectories, one for
                each character.
            list[Character]: list of characters that weren't written
                due to running out of vertical room (if any).

        """
        trajs = []

        if len(cs) == 0:
            return trajs, []

        # Note: Character object trajectories are expressed in mm,
        # so we must keep that unit conversion in mind here.

        # we assume that the dimensions + writeable area of the
        # board remain unchanged while we write.
        board = self.get_latest_board_info()
        font_height = max([c.font_size_mm for c in cs]) / 1000.0
        line_spacing = line_spacing_factor * font_height
        print(line_spacing)
        padding = padding_mm / 1000.0

        # maintain an offset for factoring in the writing area +
        # if/when we run out of space so we can use it to create a newline
        c_bounds = cs[0].get_bounding_box_mm() / 1000.0
        offset = (
            np.array([padding, -(padding + font_height)])
            + board.writeable_area[0, :]
        )
        newline_start_x = offset[0]

        for i, char in enumerate(cs):
            # for now, until this is proven to be too slow,
            # let's just calculate the bounding box for every character.
            # there's less expensive ways to do this but this is an
            # easy first pass.
            c_bounds = (char.get_bounding_box_mm() / 1000.0) + offset[
                np.newaxis, :
            ]
            if c_bounds[1, 0] >= board.writeable_area[1, 0]:
                # insert a newline before writing this character
                added_offset = np.array(
                    [
                        -(c_bounds[0, 0] - newline_start_x) + padding,
                        -(line_spacing),
                    ]
                )
                offset += added_offset
                c_bounds += added_offset

            if c_bounds[0, 0] < board.writeable_area[0, 0]:
                # this is a strange case that occurs when a newline character
                # is given to font_trajectory. it inserts its own newline,
                # which we have to handle here by fixing our own offset.

                added_offset = np.array(
                    [
                        newline_start_x - c_bounds[0, 0] + padding,
                        0,
                    ]
                )
                offset += added_offset
                c_bounds += added_offset

            if c_bounds[1, 1] <= board.writeable_area[1, 1]:
                # we can't fit this line vertically. return
                missing_chars = cs[i:]
                charstr = ''.join([c.char for c in missing_chars])
                self._logger.warning(
                    "Some characters can't fit in the writeable area. Could"
                    f'not write: {charstr}'
                )
                return trajs, missing_chars

            data = np.zeros(shape=(char.trajectory.shape[0], 8))
            data[:, 0:2] = (char.trajectory[:, 0:2] / 1000.0) + offset
            data[:, 3:7] = self.DOWN_Q[np.newaxis, :]
            data[:, 7] = char.trajectory[:, 2] * self.c.max_force_N
            traj = Trajectory(char.char, data)
            trajs.append(traj)

        return self._insert_pen_lifts(trajs), []

    def _insert_pen_lifts(self, trajs: list[Trajectory]) -> list[Trajectory]:
        """
        Handle lifting the pen off the board.

        We do this by inserting filler trajectories between letters,
        AND lifting any points inside of characters where force=0
        off the board. (to handle unconnected characters like "i")
        """
        if len(trajs) == 0:
            return []

        # doing 2 separate loops for simplicity
        # lift pen for any zero-force regions
        for c_traj in trajs:
            lift_start_i = 0
            lifted = False
            for i in range(c_traj.data.shape[0]):
                if c_traj.data[i, 7] <= self.c.pen_lift_thresh_N:
                    if not lifted:
                        lift_start_i = i
                        lifted = True
                else:
                    if lifted:
                        # lift up the pen for this region
                        c_traj.data[lift_start_i:i, 2] = (
                            self.c.off_board_height_m
                        )
                        c_traj.data[lift_start_i:i, 7] = 0.0
                        lifted = False

        # add connecting trajectories
        out = []
        for i in range(len(trajs) - 1):
            start = trajs[i].data[-1, :].copy()
            end = trajs[i + 1].data[0, :].copy()
            start[2] = self.c.off_board_height_m
            end[2] = self.c.off_board_height_m

            connector = Trajectory(label='--', data=np.array([start, end]))
            out.append(trajs[i])
            out.append(connector)

        # add the last trajectory
        out.append(trajs[-1])
        return out

    def get_latest_board_info(self) -> BoardInfo:
        """Return the most recently update board location + dimensions."""
        # todo - grab this from the BoardDetector topics.
        raise NotImplementedError
