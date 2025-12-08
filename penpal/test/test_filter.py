import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from penpal.utils import MovingAveragePoseFilter
from penpal.write_planner import BoardInfo


def make_board(
    pos_xyz,
    yaw_deg: float = 0.0,
    width_m: float = 0.8,
    height_m: float = 0.6,
    writeable_area: np.ndarray | None = None,
) -> BoardInfo:
    """Create a BoardInfo instance with a simple yaw-only orientation."""
    pos = np.array(pos_xyz, dtype=float)
    ori = R.from_euler("z", yaw_deg, degrees=True)

    if writeable_area is None:
        x_tl = 0.0
        y_tl = -height_m / 2.0
        x_br = width_m
        y_br = -height_m
        writeable_area = np.array([[x_tl, y_tl], [x_br, y_br]], dtype=float)

    return BoardInfo(
        pos=pos,
        ori=ori,
        width_m=width_m,
        height_m=height_m,
        writeable_area=writeable_area,
    )


class TestMovingAveragePoseFilter(unittest.TestCase):
    """Unit tests for MovingAveragePoseFilter."""

    def test_queue_size_must_be_positive(self):
        """Queue size 0 or negative should raise ValueError."""
        with self.assertRaises(ValueError):
            MovingAveragePoseFilter(queue_size=0)
        with self.assertRaises(ValueError):
            MovingAveragePoseFilter(queue_size=-3)


if __name__ == "__main__":
    unittest.main()
