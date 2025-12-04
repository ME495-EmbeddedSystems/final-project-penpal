"""Unit tests for write_planner.py."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from penpal.write_planner import BoardInfo


def test_board_info_simple() -> None:
    """Quick unit test for essential BoardInfo functionality."""
    board = BoardInfo(
        pos=np.array([0, 0, 0], dtype=float),
        ori=R.identity(),
        width_m=0.2,
        height_m=0.2,
        writeable_area=np.array([[0, 0], [0.2, -0.2]]),
    )

    np.testing.assert_almost_equal(board.T_sb, np.eye(4))

    bcorn = board.get_board_corners_world_frame()
    np.testing.assert_almost_equal(
        bcorn,
        np.array(
            [
                [0, 0, 0],
                [0.2, 0, 0],
                [0.2, -0.2, 0],
                [0, -0.2, 0],
            ]
        ),
    )

    wcorn = board.get_writeable_area_corners_world_frame()
    np.testing.assert_almost_equal(wcorn, bcorn)
