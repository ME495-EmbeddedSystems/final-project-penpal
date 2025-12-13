"""Unit tests for write_planner.py."""

import numpy as np
from penpal.write_planner import BoardInfo, Character
from scipy.spatial.transform import Rotation as R


def test_board_info_basic() -> None:
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


def test_character_basic():
    """Test basic character logic."""
    radius_mm = 1.0
    n_points = 20
    xvals = radius_mm * np.sin(np.linspace(0, 2 * np.pi, n_points))
    yvals = radius_mm * np.cos(np.linspace(0, 2 * np.pi, n_points))
    circle_flat = np.vstack([xvals, yvals, np.zeros_like(yvals)]).T

    c = Character('c', trajectory=circle_flat, font_size_mm=2.0)

    bbox = c.get_bounding_box_mm()
    expected = np.array([[-1, 1], [1, -1]])
    np.testing.assert_almost_equal(bbox, expected, decimal=1)
