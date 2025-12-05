"""Integration tests for bringing up the write planner."""

import asyncio
from pathlib import Path
from matplotlib import pyplot as plt
from mock import Mock, MagicMock, patch

from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np

from penpal.integration_tests import plot
from penpal.write_planner import BoardInfo, WritePlanner, Character
from penpal.font_trajectory import FontTrajectory
from penpal.control.pp_control import PPControlBase, Trajectory


class MockController(PPControlBase):
    """Controller mock for testing purposes."""

    def __init__(
        self, node: Node, cfg: PPControlBase.Config | None = None
    ) -> None:
        """Initialize the object."""
        super().__init__(node, cfg)

        self.trajs = []

    async def _execute_trajectory(
        self,
        traj: plot.Trajectory,
        target_ee_velocity_m_s: float,
        publish_markers: bool = False,
    ) -> None:
        self._logger.debug(f'Received trajectory: {traj.label}')
        self.trajs.append(traj)

    async def grip(
        self, offset_m: float, grip_force_N: float | None = None
    ) -> None:
        """Grip mock."""
        pass


async def test_static_board() -> tuple[list[Trajectory], BoardInfo]:
    """Write some characters on a non-moving board."""
    board = BoardInfo(
        # pos=np.array([0, 0, 0]),
        # ori=R.identity(),
        pos=np.array([0.1, 0.2, 0.3], dtype=float),
        ori=R.from_euler('x', (30), degrees=True),
        width_m=0.2,
        height_m=0.2,
        writeable_area=np.array([[0.01, -0.1], [0.19, -0.2]]),
    )

    dr = R.from_euler('x', (0.1), True)

    def mock_board_info(o: WritePlanner) -> BoardInfo:
        # move the board a little bit every time
        # this is called to simulate the real motion
        board.pos[2] -= 0.001
        board.ori *= dr
        return board

    node = MagicMock(Node)
    control = MockController(node)
    writer = WritePlanner(node, control)
    font = FontTrajectory()

    fonts_dir = Path(__file__).parents[3] / 'fonts/'
    roboto_path = fonts_dir / 'Roboto-Regular.ttf'

    font.add_font(roboto_path)
    font_size = 20.0
    chars = font.write_text(
        'Hello World! My name is PenPal. :)\nI can write many '
        'things but eventually, I run out of room. ',
        # 'Hello World! My name is PenPal. :) I am unwriteable',
        # 'Hello World! My name is PenPal. :)',
        'Roboto-Regular',
        font_size,
        1.0,
    )
    with patch(
        'penpal.write_planner.WritePlanner.get_latest_board_info',
        mock_board_info,
    ):
        leftovers = await writer.write_characters(
            chars, font.c.line_spacing_factor
        )

        if leftovers:
            print(
                'Leftover characters: ' + ''.join([c.char for c in leftovers])
            )

    # now return the actual trajectories as written to the board in space.
    return control.trajs, board


if __name__ == '__main__':
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        trajs, board = asyncio.run(test_static_board())
        plot.plot_trajectories_and_board(trajs, board, show_ori=True)
        plt.show()
    finally:
        print('Test complete.')
