"""
Integration test node for developing PP control.
"""

import asyncio
import threading
import signal

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from penpal.control import position_control, moveit_control, pp_control
from penpal.control.pp_control import Trajectory
from penpal.integration_tests import plot


def traj_from_points(
    label: str,
    points: np.ndarray,
    center: np.ndarray,
    rot: R,
    force: float | None,
) -> Trajectory:
    """Convert set of R3 points into full 8D array."""
    if force is None:
        force = 0.0

    points = rot.apply(points) + center
    down = R.from_euler('xyz', [0, 0, -np.pi / 2])
    ori = R.from_matrix(rot.as_matrix() @ down.as_matrix())
    ori_quat = ori.as_quat(True)
    # ori_quat = down.as_quat(True)

    force_per_point = np.full((points.shape[0], 1), force)
    ori_per_point = np.broadcast_to(ori_quat, (points.shape[0], 4))
    points_full = np.hstack([points, ori_per_point, force_per_point])
    return Trajectory(label, points_full)


def get_circle_trajectory(
    radius_m: float,
    center: np.ndarray,
    rot: R,
    n_points: int,
    force: float | None = None,
) -> Trajectory:
    """
    Return a trajectory of a circle.

    Args:
        radius_m (float): radius of the circle
        center (np.ndarray): square center location [x, y, z]
        rot: rotation relative to world frame, where shape is on xy plane
        n_points (float): number of points with which to construct the circle
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory for a circle in space

    """
    xvals = radius_m * np.sin(np.linspace(0, 2 * np.pi, n_points))
    yvals = radius_m * np.cos(np.linspace(0, 2 * np.pi, n_points))
    circle_flat = np.vstack([xvals, yvals, np.zeros_like(yvals)]).T

    return traj_from_points('circle', circle_flat, center, rot, force)


def get_arrow_trajectory(
    scale: float, center: np.ndarray, rot: R, force: float | None = None
) -> Trajectory:
    """
    Return a trajectory of an arrow.

    Args:
        scale (float): length of the long line
        center (np.ndarray): square center location [x, y, z]
        rot: rotation relative to world frame, where shape is on xy plane
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory through 3d space

    """
    tip = np.array([0, 0, 0])
    tail = np.array([-scale, 0, 0])
    lpoint = np.array([-scale * 0.2, -scale * 0.2, 0])
    rpoint = np.array([-scale * 0.2, scale * 0.2, 0])
    points = np.array([tail, tip, lpoint, rpoint, tip])

    return traj_from_points('arrow', points, center, rot, force)


def get_square_trajectory(
    side: float, center: np.ndarray, rot: R, force: float | None = None
) -> Trajectory:
    """
    Return a trajectory of a square.

    Args:
        side (float): length of the long line
        center (np.ndarray): square center location [x, y, z]
        rot: rotation relative to world frame, where shape is on xy plane
        force (np.ndarray): 3dof force applied at EE

    Returns:
        Trajectory: trajectory through 3d space

    """
    bl = -np.array([0.5 * side, 0.5 * side, 0])
    tl = bl + np.array([0, side, 0])
    tr = tl + np.array([side, 0, 0])
    br = tr + np.array([0, -side, 0])
    points = np.array([bl, tl, tr, br, bl])

    return traj_from_points('square', points, center, rot, force)


def get_demo_traj_sequence(start_pose: np.ndarray) -> list[Trajectory]:
    """
    Put together a demo trajectory sequence of circle, arrow, sq.

    Args:
        start_pose: [x, y, z, qx, qy, qz, qw]

    """
    lineh = 0.02
    off_board_dist = 0.02
    rot = R.from_quat(start_pose[3:])
    circle_c = start_pose[:3]
    arrow_c = circle_c + rot.apply(np.array([lineh * 1.6, 0, 0]))
    square_c = arrow_c + rot.apply(np.array([lineh * 0.6, 0, 0]))

    # do them all in a line, then tilt em
    circle_traj = get_circle_trajectory(lineh / 2, circle_c, rot, 50)
    arrow_traj = get_arrow_trajectory(lineh, arrow_c, rot)
    square_traj = get_square_trajectory(lineh, square_c, rot)

    board_gap = rot.apply(np.array([0, 0, off_board_dist]))
    board_gap = np.array([*board_gap, 0, 0, 0, 0, 0])

    out = []
    last_point = None
    for traj in [circle_traj, arrow_traj, square_traj]:
        if last_point is not None:
            # insert extra points hovering off the board so we don't draw
            # between the shapes
            points = np.array(
                [
                    last_point + board_gap,
                    traj.data[0] + board_gap,
                ]
            )
            # these extra points should have no force
            points[:, 7] = 0
            space_traj = Trajectory('space', points)
            out.append(space_traj)
        out.append(traj)
        last_point = traj.data[-1]

    return out


async def integration_test(node: Node, ctl: pp_control.PPControlBase) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    try:
        wait_t = 10.0
        logger.info(f'Waiting {wait_t} seconds...')
        await asyncio.sleep(wait_t)
        logger.info('Starting integration test...')
        await ctl.configure()

        speed = 0.05
        rot = R.from_euler('xyz', [180, 0, 0], degrees=True)
        start_pose = np.array([0.4, 0, 0.4, *rot.as_quat(True)])
        node.get_logger().info("Moving to start position...")
        goal_handle = await ctl.move_to_ee_pose(goal_ee_position=start_pose[:3],
                                                goal_ee_orientation=start_pose[3:],
                                                execute_immediately=True)
        res = await goal_handle.get_result_async()
        if res.result.error_code.val != 1: # 1 is SUCCESS in MoveIt
            node.get_logger().error(f"CRITICAL: Failed to reach start position! Error: {res.result.error_code.val}")
            return

        seq = get_demo_traj_sequence(np.array([0, 0, 0, 0, 0, 0, 1]))

        # publish trajectories one by one
        # uncomment once no failure
        for traj in seq:
            logger.info(f'Executing trajectory {traj.label}...')
            await ctl.execute_trajectory(traj, speed, publish_markers=True)

        # logger.info('Publishing all trajectory markers...')
        # for traj in seq:
        #     await ctl.publish_marker(traj)

        # for traj in seq:
        #     logger.info(f'Executing trajectory {traj.label}...')
        #     # markers already published for now so no need to republish here
        #     await ctl.execute_trajectory(traj, speed, publish_markers=False)

    finally:
        node.get_logger().info('Integration test finished.')


def plot_shapes() -> None:
    """Quick plotting demos of the trajectory functions."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    rot = R.from_euler('xyz', [0, 0, 0], False)

    c = np.array([1, 2, 3])
    traj1 = get_circle_trajectory(2.0, c, rot, 30)
    traj2 = get_arrow_trajectory(2.0, c, rot)
    trajsq = get_square_trajectory(2.0, c, rot)

    for traj in [traj1, traj2, trajsq]:
        ax.plot(traj.data[:, 0], traj.data[:, 1], traj.data[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def plot_demo_seq() -> None:
    """Plot the demo sequence on a 3d plot."""
    rot = R.from_euler('xyz', [2, 3, -1])
    start_pose = np.array([0.4, 0, 0.2, *rot.as_quat(True)])
    seq = get_demo_traj_sequence(start_pose)
    plot.plot_trajectory_sequence(seq, True)
    plt.show()


def main_moveit():
    """Run main."""
    rclpy.init()
    node = rclpy.create_node('test_moveit_ctl')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    ctl = moveit_control.MoveItPPControl(node)

    try:
        asyncio.run(integration_test(node, ctl))
    finally:
        executor.shutdown()
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # make matplotlib handle ctrl+c
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # plot_shapes()
    plot_demo_seq()
    # main_moveit()
