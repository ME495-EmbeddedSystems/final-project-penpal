"""Integration test node for developing PP control."""

import asyncio
import threading

import numpy as np

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from penpal.control.moveit_control import MoveItPPControl
from penpal.control.position_control import PositionPPControl
from penpal.control.pp_control import Trajectory
from penpal.integration_tests import plot
from penpal.integration_tests.demo_write_planner import DemoWritePlanner


def get_circle_trajectory(
    radius_m: float,
    center: np.ndarray,
    rot: R,
    n_points: int,
    force: np.ndarray | None = None,
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
    if force is None:
        force = np.zeros(3)
    xvals = radius_m * np.sin(np.linspace(0, 2 * np.pi, n_points))
    yvals = radius_m * np.cos(np.linspace(0, 2 * np.pi, n_points))
    circle_flat = np.vstack([xvals, yvals, np.zeros_like(yvals)]).T

    points = rot.apply(circle_flat) + center

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('circle', points_with_force)

    return traj


def get_arrow_trajectory(
    scale: float, center: np.ndarray, rot: R, force: np.ndarray | None = None
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
    if force is None:
        force = np.zeros(3)

    tip = np.zeros(3)
    tail = np.array([-scale, 0, 0])
    lpoint = np.array([-scale * 0.2, -scale * 0.2, 0])
    rpoint = np.array([-scale * 0.2, scale * 0.2, 0])
    points = np.array([tail, tip, lpoint, rpoint, tip])

    points = rot.apply(points) + center

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('arrow', points_with_force)

    return traj


def get_square_trajectory(
    side: float, center: np.ndarray, rot: R, force: np.ndarray | None = None
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
    if force is None:
        force = np.zeros(3)

    bl = -np.array([0.5 * side, 0.5 * side, 0])
    tl = bl + np.array([0, side, 0])
    tr = tl + np.array([side, 0, 0])
    br = tr + np.array([0, -side, 0])
    points = np.array([bl, tl, tr, br, bl])

    points = rot.apply(points) + center

    force_per_point = np.broadcast_to(force, (points.shape[0], 3))
    points_with_force = np.hstack([points, force_per_point])
    traj = Trajectory('arrow', points_with_force)

    return traj


def get_demo_traj_sequence(start_pose: np.ndarray) -> list[Trajectory]:
    """
    Put together a demo trajectory sequence of circle, arrow, sq.

    Args:
        start_pose: [x, y, z, qx, qy, qz, qw]

    """
    lineh = 0.05
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
    board_gap = np.array([*board_gap, 0, 0, 0])

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
            points[:, 3:] = 0
            space_traj = Trajectory('space', points)
            out.append(space_traj)
        out.append(traj)
        last_point = traj.data[-1]

    return out


async def integration_test(node: Node, ctl: PositionPPControl) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    try:
        logger.info('Starting integration test...')
        speed = 0.05
        start_pose = np.array([0, 0, 0, 0, 0, 0, 1])
        seq = get_demo_traj_sequence(start_pose)
        for traj in seq:
            logger.info(f'Executing trajectory {traj.label}...')
            await ctl.execute_trajectory(traj, speed)

    finally:
        node.get_logger().info('Integration test finished.')


async def write_planner_test(node: Node, ctl: PositionPPControl) -> None:
    """Test the Write Planner logic."""
    logger = node.get_logger()
    try:
        logger.info('Starting Planner Integration Test...')
        speed = 0.05
        board_origin = np.array([0.0, 0.0, 0.0])
        planner = DemoWritePlanner(board_origin)
        start_pose = np.array([0, 0, 0, 0, 0, 0, 1])
        seq = get_demo_traj_sequence(start_pose) * 3
        global_sequence = planner.write_characters(seq)
        for traj in global_sequence:
            logger.info(f'Executing {traj.label}...')
            await ctl.execute_trajectory(traj, speed)
    finally:
        node.get_logger().info('Write_planner_test finished.')


async def integration_test_write_on_board(node: Node,
                                          planner: MoveItPPControl) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    viz = RvizVisualizer(node)
    try:
        logger.info('Service is ready. Waiting 3sec...')
        await asyncio.sleep(3.0)
        # cartesian pose of ee tip : [0.307, 0.000, 0.487]

        board_origin = np.array([0.4, -0.025, 0.191])

        r_measured = R.from_quat([-0.000, -0.087, 0.000, -0.996])
        r_fix = R.from_euler('x', 180, degrees=True)
        r_final = r_measured * r_fix
        board_orientation = r_final.as_quat()
        demo_planner = DemoWritePlanner(board_origin, board_orientation)
        seq = get_demo_traj_sequence(np.array([0, 0, 0, 0, 0, 0, 1]))
        global_sequence = demo_planner.write_characters(seq)
        logger.info('PUblihsing marker')
        viz.publish_trajectories(global_sequence)
        speed_m_s = 0.01
        if global_sequence:
            first_pt = global_sequence[0].data[0]
            hover_pos = first_pt[:3].copy()
            hover_pos[2] += 0.05
            logger.info('Move to start hover position')
            hover_pose_7d = np.concatenate([hover_pos, board_orientation])
            await planner.plan_cartesian_path(goal_ee_pose=hover_pose_7d,
                                              execute_immediately=True)

        for i, traj in enumerate(global_sequence):
            logger.info(f'Executing segment {i}: {traj.label}')
            # convert [x, y, z, fx, fy, fz]to[x, y, z, qx, qy, qz, qw]
            points_7d = []
            for pt in traj.data:
                p7 = np.concatenate([pt[:3], board_orientation])
                points_7d.append(p7)
            traj_7d = Trajectory(traj.label, np.array(points_7d))
            await planner.execute_trajectory(traj_7d, speed_m_s)
            await asyncio.sleep(0.5)

        # Start Citation[2] #
        logger.info('Test Complete. Keeping node alive to show markers...')
        logger.info('Press CTRL+C in the terminal to exit.')
        while rclpy.ok():
            await asyncio.sleep(1.0)
        # End Citation #

    finally:
        node.get_logger().info('Integration test finished.')


def main():
    """Run main."""
    rclpy.init()
    node = rclpy.create_node('test_position_controller_node')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # ctl = PositionPPControl(node)
    ctl = MoveItPPControl(node)
    try:
        asyncio.run(integration_test_write_on_board(node, ctl))
    finally:
        executor.shutdown()
        if executor_thread.join():
            executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()


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
    start_pose = np.array([0, 0, 0, *rot.as_quat(True)])
    seq = get_demo_traj_sequence(start_pose)
    plot.plot_trajectory_sequence(seq)
    plt.show()


def plot_wp_seq() -> None:
    """Plot the write planner sequcne on a 3d plot."""
    rot = R.from_euler('xyz', [0, 0, 0])
    start_pose = np.array([0, 0, 0, *rot.as_quat(True)])
    board_origin = np.array([0.0, 0.0, 0.0])
    board_orientation = np.array([0.0, -0.087, 0.0, -0.996])
    # Rotate 45 degrees in y axis
    planner = DemoWritePlanner(board_origin, board_orientation)
    seq = get_demo_traj_sequence(start_pose) * 5
    seq = planner.write_characters(seq)
    plot.plot_trajectory_sequence(seq)
    plt.show()


class RvizVisualizer:
    """Visualize Markers to see what the robot is writing."""

    def __init__(self, node: Node):
        """Initialize markers."""
        self.node = node
        self.pub = node.create_publisher(MarkerArray, '/writing_markers', 10)
        self.stored_markers = MarkerArray()
        self.timer = node.create_timer(1.0, self._timer_callback)

    # Begin Citation[2] #
    def publish_trajectories(self, trajectories: list[Trajectory]):
        """Generate the markers and stores them for continuous publishing."""
        ma = MarkerArray()
        id_counter = 0

        for traj in trajectories:
            marker = Marker()
            marker.header.frame_id = 'base'
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = 'writing_path'
            marker.id = id_counter
            id_counter += 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.002

            # Colors
            if 'space' in traj.label:
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5) 
            else:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) 

            for pt in traj.data:
                p = Point()
                p.x, p.y, p.z = pt[0], pt[1], pt[2]
                marker.points.append(p)

            ma.markers.append(marker)

        # Store and publish immediately
        self.stored_markers = ma
        self.pub.publish(ma)

    def _timer_callback(self):
        """Periodically republishes the markers so RViz doesn't miss them."""
        if self.stored_markers.markers:
            now = self.node.get_clock().now().to_msg()
            for m in self.stored_markers.markers:
                m.header.stamp = now
            self.pub.publish(self.stored_markers)
    # End Citation #


if __name__ == '__main__':
    # plot_shapes()
    # plot_demo_seq()
    plot_wp_seq()
    # main()
