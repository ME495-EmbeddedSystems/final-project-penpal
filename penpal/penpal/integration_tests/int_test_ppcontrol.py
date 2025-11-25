"""
Integration test node for developing PP control.
"""

import asyncio
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from penpal.control.position_control import PositionPPControl
from penpal.control.pp_control import Trajectory


def get_circle_trajectory(
    radius_m: float,
    center: np.ndarray,
    n_points: int,
    force: np.ndarray | None = None,
) -> Trajectory:
    """
    Return a trajectory of a circle.

    Args:
        radius_m (float): radius of the circle
        center (np.ndarray): center & orientation [x, y, z, qx, qy, qz, qw]
        if orientation is 0, circle is on the xy plane.
        n_points (float): number of points with which to construct the circle
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory for a circle in space

    """
    if force is None:
        force = np.zeros(3)
    xvals = radius_m * np.sin(np.linspace(0, 2 * np.pi, n_points))
    yvals = radius_m * np.cos(np.linspace(0, 2 * np.pi, n_points))
    circle_flat = np.vstack([xvals, yvals, np.zeros_like(yvals)]).T
    circle_flat += center[:3]

    rot = R.from_quat(center[3:])
    points = rot.apply(circle_flat)

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('circle', points_with_force)

    return traj


def get_arrow_trajectory(
    scale: float, center: np.ndarray, force: np.ndarray | None = None
) -> Trajectory:
    """
    Return a trajectory of an arrow.

    Args:
        scale (float): length of the long line
        center (np.ndarray): arrow point location & orientation [x, y, z, qx, qy, qz, qw]
        if orientation is 0, arrow is on the xy plane.
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory through 3d space

    """
    if force is None:
        force = np.zeros(3)

    rot = R.from_quat(center[3:])
    center = center[:3]
    tail = center + np.array([-scale, 0, 0])
    lpoint = center + np.array([-scale * 0.2, -scale * 0.2, 0])
    rpoint = center + np.array([-scale * 0.2, scale * 0.2, 0])
    points = np.array([tail, center, lpoint, rpoint, center])

    points = rot.apply(points)

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('arrow', points_with_force)

    return traj


def get_square_trajectory(
    side: float, center: np.ndarray, force: np.ndarray | None = None
) -> Trajectory:
    """
    Return a trajectory of a square.

    Args:
        side (float): length of the long line
        center (np.ndarray): square center location & orientation [x, y, z, qx, qy, qz, qw]
        if orientation is 0, square is on the xy plane.
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory through 3d space

    """
    if force is None:
        force = np.zeros(3)

    rot = R.from_quat(center[3:])
    center = center[:3]

    bl = center - np.array([0.5 * side, 0.5 * side, 0])
    tl = bl + np.array([0, side, 0])
    tr = tl + np.array([side, 0, 0])
    br = tr + np.array([0, -side, 0])
    points = np.array([bl, tl, tr, br, bl])

    points = rot.apply(points)

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('arrow', points_with_force)

    return traj


async def integration_test(node: Node, ctl: PositionPPControl) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    try:
        logger.info('helloworld from integration test')

    finally:
        node.get_logger().info('Integration test finished.')


def main():
    """Run main."""
    rclpy.init()
    node = rclpy.create_node('test_position_controller_node')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    ctl = PositionPPControl(node)

    try:
        asyncio.run(integration_test(node, ctl))
    finally:
        executor.shutdown()
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()


def demo_shapes() -> None:
    """Quick plotting demos of the trajectory functions."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ori = R.from_euler('xyz', [np.pi / 2, 0, 0], False)

    c = np.array([1, 2, 3, *ori.as_quat(True)])
    traj1 = get_circle_trajectory(2.0, c, 30)
    traj2 = get_arrow_trajectory(2.0, c)
    trajsq = get_square_trajectory(2.0, c)

    for traj in [traj1, traj2, trajsq]:
        ax.plot(traj.data[:, 0], traj.data[:, 1], traj.data[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


if __name__ == '__main__':
    demo_shapes()
    # main()
