"""Unit tests for pp_control.py."""

import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest

from penpal.control.pp_control import Trajectory


@pytest.fixture
def sample_trajectory() -> Trajectory:
    """Arbitrary sample trajectory."""
    n_points = 11
    ori = R.from_euler('xyz', [0, np.pi, np.pi / 2], False)
    f = 3.0

    data = np.empty(shape=(n_points, 8))
    for i in range(data.shape[0]):
        data[i, :] = [i, i + 1, i + 2, *ori.as_quat(True), f]

    return Trajectory('test', data)


def test_traj_basic(sample_trajectory):
    """Perform basic sanity checks for Trajectory class."""
    traj = sample_trajectory

    out = traj.transform(np.array([0, 0, 0]), R.identity())
    np.testing.assert_allclose(out.data, traj.data)

    segs = traj.split_with_len(3)
    print(segs)
    assert len(segs) == 4


def test_traj_transform(sample_trajectory):
    """Test."""
    traj = sample_trajectory

    out = traj.transform(np.array([1, 2, 3]), R.identity())

    expected = traj.data + np.array([1, 2, 3, 0, 0, 0, 0, 0])
    np.testing.assert_almost_equal(out.data, expected)

    out2 = traj.transform(np.zeros(3), R.from_euler('xyz', (1, 2, 3)))
    assert not np.allclose(out2.data, traj.data)
