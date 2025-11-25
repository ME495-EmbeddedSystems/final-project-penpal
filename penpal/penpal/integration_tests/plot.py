"""Helpful plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from penpal.control.pp_control import Trajectory


def plot_trajectory_sequence(seq: list[Trajectory]) -> None:
    """Plot a sequence of trajectories on the same 3d plot."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    last_point = None
    for traj in seq:
        if last_point is not None:
            points = np.array([last_point[:3], traj.data[0][:3]])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c='black')
        ax.plot(
            traj.data[:, 0],
            traj.data[:, 1],
            traj.data[:, 2],
            label=traj.label,
        )
        last_point = traj.data[-1]

        # plot green & red dots for start & end, respectively
        ax.scatter(
            [traj.data[0, 0]],
            [traj.data[0, 1]],
            [traj.data[0, 2]],
            c='green',
            marker='o',
            s=20,
        )
        ax.scatter(
            [traj.data[-1, 0]],
            [traj.data[-1, 1]],
            [traj.data[-1, 2]],
            c='red',
            marker='o',
            s=20,
        )

    ax.legend()
