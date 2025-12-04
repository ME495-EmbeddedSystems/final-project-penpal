"""Helpful plotting functions."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

from penpal.control.pp_control import Trajectory
from penpal.write_planner import BoardInfo


def plot_trajectory_sequence(
    seq: list[Trajectory],
    plot_ori_arrows: bool = False,
    ax: Axes3D | None = None,  # type: ignore
) -> None:
    """Plot a sequence of trajectories on the same 3d plot."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    ax: Axes3D  # type: ignore

    last_point = None
    for traj in seq:
        print(f"Plotting '{traj.label}'")
        if last_point is not None:
            points = np.array([last_point[:3], traj.data[0][:3]])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c='black')
        X = traj.data[:, 0]
        Y = traj.data[:, 1]
        Z = traj.data[:, 2]
        ax.plot(X, Y, Z, label=traj.label)
        last_point = traj.data[-1]

        if plot_ori_arrows:
            rot = R.from_quat(traj.data[:, 3:7])
            euler = rot.as_euler('xyz')
            U = euler[:, 0]
            V = euler[:, 1]
            W = euler[:, 2]
            ax.quiver(X, Y, Z, U, V, W, length=0.01)

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

    set_axes_equal(ax)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()


def plot_trajectories_and_board(seq: list[Trajectory], b: BoardInfo) -> None:
    """Plot a set of trajectories + the whiteboard area."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_trajectory_sequence(seq, False, ax)

    b_corners = b.get_board_corners_world_frame()
    ax.plot(
        b_corners[:, 0],
        b_corners[:, 1],
        b_corners[:, 2],
        label='Board Outline',
    )
    w_corners = b.get_writeable_area_corners_world_frame()
    ax.plot(
        w_corners[:, 0],
        w_corners[:, 1],
        w_corners[:, 2],
        label='Writeable Area',
    )


############### Begin_Citation[1] ####################


def set_axes_equal(ax: Axes3D) -> None:
    """
    Make axes of 3D plot have equal scale.

    so that spheres appear as spheres, cubes as cubes, etc.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().

    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


############### End_Citation[1] ####################
