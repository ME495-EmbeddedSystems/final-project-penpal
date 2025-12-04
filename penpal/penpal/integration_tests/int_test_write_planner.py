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
        super().__init__(node, cfg)

        self.trajs = []

    async def _execute_trajectory(
        self,
        traj: plot.Trajectory,
        target_ee_velocity_m_s: float,
        publish_markers: bool = False,
    ) -> None:
        print(f'Received trajectory: {traj.label}')
        self.trajs.append(traj)

    async def grip(
        self, offset_m: float, grip_force_N: float | None = None
    ) -> None:
        """Grip mock."""
        pass


async def test_static_board() -> tuple[list[Trajectory], BoardInfo]:
    """Write some characters on a non-moving board."""
    board = BoardInfo(
        pos=np.array([0, 0, 0]),
        ori=R.identity(),
        # pos=np.array([1, 2, 3], dtype=float),
        # ori=R.from_euler('xy', (30, 60), degrees=True),
        width_m=0.2,
        height_m=0.2,
        writeable_area=np.array([[0.01, -0.1], [0.19, -0.2]]),
    )

    def mock_board_info(o: WritePlanner) -> BoardInfo:
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
        'hello world! My name is PenPal.', 'Roboto-Regular', font_size, 1.0
    )
    print(chars)

    with patch(
        'penpal.write_planner.WritePlanner.get_latest_board_info',
        mock_board_info,
    ):
        await writer.write_characters(
            chars, font.c.line_spacing_factor * font_size
        )

    # now return the actual trajectories as written to the board in space.
    return control.trajs, board


if __name__ == '__main__':
    try:
        trajs, board = asyncio.run(test_static_board())
        plot.plot_trajectories_and_board(trajs, board)
        plt.show()
    finally:
        print('Test complete.')
