"""Integration test node for developing PP control."""

import asyncio
import signal
import threading
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from franka_msgs.srv import SetFullCollisionBehavior
from franka_msgs.srv import SetTCPFrame

import matplotlib.pyplot as plt

import numpy as np

from penpal import font_trajectory, write_planner
from penpal.control import moveit_control, position_control, pp_control
from penpal.control.pp_control import Trajectory
from penpal.integration_tests import plot

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from scipy.spatial.transform import Rotation as R


def calculate_start_pose(buffer,
                         board_pose_position: np.array,
                         board_pose_rotation: R):
    """Calculate the start position for writing."""
    world_Z_vector = np.array([0, 0, 1])
    board_normal = np.array([1, 0, 0])
    desired_X_axis = board_pose_rotation.apply(board_normal)

    desired_vertical_axis = world_Z_vector
    desired_Y_axis = np.cross(desired_X_axis, desired_vertical_axis) 

    desired_Y_axis /= np.linalg.norm(desired_Y_axis)
    desired_Z_axis = np.cross(desired_X_axis, desired_Y_axis)
    R_matrix = np.vstack([desired_X_axis, desired_Y_axis, desired_Z_axis]).T
    target_rot = R.from_matrix(R_matrix)
    target_position = board_pose_position - (desired_X_axis * buffer)
    target_orientation_quat = target_rot.as_quat()
    start_pose = np.array([*target_position, *target_orientation_quat])
    return start_pose


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
    points_temp = np.zeros_like(points)
    points_temp[:, 1] = points[:, 0]
    points_temp[:, 2] = points[:, 1]
    points_rotated = rot.apply(points_temp) + center
    ori_quat = rot.as_quat()
    force_per_point = np.full((points.shape[0], 1), force)
    ori_per_point = np.broadcast_to(ori_quat, (points.shape[0], 4))
    points_full = np.hstack([points_rotated, ori_per_point, force_per_point])
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
    """Create a trajectory sequence for demo."""
    lineh = 0.05
    off_board_dist = 0.02
    rot = R.from_quat(start_pose[3:])
    circle_c = start_pose[:3]
    horizontal_unit_vec = rot.apply(np.array([0, 1, 0]))
    vertical_unit_vec = rot.apply(np.array([0, 0, 1]))
    circle_r = lineh / 2
    arrow_width = lineh
    half_square = lineh / 2
    gap = 0.02
    space = lineh * 1.5

    current_center = circle_c.copy()
    circle_traj = get_circle_trajectory(circle_r, current_center, rot, 50)
    out = []
    out.append(circle_traj)

    last_right_edge = current_center + horizontal_unit_vec * circle_r
    horizontal_offset_m = gap + (arrow_width / 2)

    arrow_c = last_right_edge + horizontal_unit_vec * horizontal_offset_m
    board_gap = rot.apply(np.array([-off_board_dist, 0, 0]))
    space_traj_ca = Trajectory('space', np.array([
        circle_traj.data[-1] + np.array([*board_gap, 0, 0, 0, 0, 0]),
        np.array([*arrow_c, *rot.as_quat(), 0.0])
        + np.array([*board_gap, 0, 0, 0, 0, 0]),
    ]))
    out.append(space_traj_ca)

    arrow_traj = get_arrow_trajectory(lineh, arrow_c, rot)
    out.append(arrow_traj)
    last_right_edge = arrow_c.copy()
    horizontal_offset_m = gap + half_square
    square_c = last_right_edge + horizontal_unit_vec * horizontal_offset_m
    space_traj_as = Trajectory('space', np.array([
        arrow_traj.data[-1] + np.array([*board_gap, 0, 0, 0, 0, 0]),
        np.array([*square_c, *rot.as_quat(), 0.0])
        + np.array([*board_gap, 0, 0, 0, 0, 0]),
    ]))
    out.append(space_traj_as)

    square_traj = get_square_trajectory(lineh, square_c, rot)
    out.append(square_traj)
    start_center_2 = circle_c + vertical_unit_vec * space
    current_center = start_center_2.copy()

    square2_traj = get_square_trajectory(lineh, current_center, rot)
    space_traj_1to2 = Trajectory('space', np.array([
        square_traj.data[-1] + np.array([*board_gap, 0, 0, 0, 0, 0]),
        np.array([*current_center, *rot.as_quat(), 0.0])
        + np.array([*board_gap, 0, 0, 0, 0, 0]),
    ]))
    out.append(space_traj_1to2)
    out.append(square2_traj)

    last_right_edge = current_center + horizontal_unit_vec * half_square
    horizontal_offset_m = gap + circle_r
    circle2_c = last_right_edge + horizontal_unit_vec * horizontal_offset_m

    space_traj_sc = Trajectory('space', np.array([
        square2_traj.data[-1] + np.array([*board_gap, 0, 0, 0, 0, 0]),
        np.array([*circle2_c, *rot.as_quat(), 0.0])
        + np.array([*board_gap, 0, 0, 0, 0, 0]),
    ]))
    out.append(space_traj_sc)

    circle2_traj = get_circle_trajectory(circle_r, circle2_c, rot, 50)
    out.append(circle2_traj)
    last_right_edge = circle2_c + horizontal_unit_vec * circle_r

    horizontal_offset_m = gap + (arrow_width / 2)
    arrow2_c = last_right_edge + horizontal_unit_vec * horizontal_offset_m
    space_traj_ca2 = Trajectory('space', np.array([
        circle2_traj.data[-1] + np.array([*board_gap, 0, 0, 0, 0, 0]),
        np.array([*arrow2_c, *rot.as_quat(), 0.0])
        + np.array([*board_gap, 0, 0, 0, 0, 0]),
    ]))
    out.append(space_traj_ca2)
    arrow2_traj = get_arrow_trajectory(lineh, arrow2_c, rot)
    out.append(arrow2_traj)
    return out


def get_demo_traj_sequence_real(start_pose: np.ndarray) -> list[Trajectory]:
    """
    Put together a demo trajectory sequence of circle, arrow, sq.

    Args:
        start_pose: [x, y, z, qx, qy, qz, qw]

    """
    lineh = 0.05
    off_board_dist = 0.02
    rot = R.from_quat(start_pose[3:])
    circle_c = start_pose[:3]
    arrow_c = circle_c + rot.apply(np.array([0, -lineh * 1.6, 0]))
    square_c = arrow_c + rot.apply(np.array([0, -lineh * 0.6, 0]))

    # do them all in a line, then tilt em
    circle_traj = get_circle_trajectory(lineh / 2, circle_c, rot, 50)
    arrow_traj = get_arrow_trajectory(lineh, arrow_c, rot)
    square_traj = get_square_trajectory(lineh, square_c, rot)

    board_gap = rot.apply(np.array([-off_board_dist, 0, 0]))
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


def ee_change_matrix():
    """Set matrix to move EE from tcp_hand to pen tip."""
    T_final = np.eye(4)
    T_final[2, 3] = 0.1
    return T_final.flatten(order='F').tolist()


async def generate_write_sequence(
    node: Node,
    ctl: pp_control.PPControlBase,
    text: str,
    demo_board_pose: np.ndarray,
    demo_board_rot: R,
    font_name: str = 'Roboto-Regular',
    font_size_mm: float = 30.0,
    pen_thickness_mm: float = 2.0,
    space_factor: float = 1.2
) -> list[pp_control.Trajectory]:
    """Set up the BoardInfo, character, and final trajectories."""
    fonts = font_trajectory.FontTrajectory()

    try:
        package_share = Path(get_package_share_directory('penpal'))
        font_path = package_share / 'fonts' / 'Roboto-Regular.ttf'
        fonts.add_font(font_path)
        node.get_logger().info(f'Loaded font from: {font_path}')
    except Exception as e:
        node.get_logger().error(f'Failed to load font path: {e}')

    wp = write_planner.WritePlanner(node, ctl)
    writeable_area_coords = np.array([[0.0, -0.305], [0.8, -0.61]])

    board_info = write_planner.BoardInfo(
        pos=demo_board_pose,
        ori=demo_board_rot,
        width_m=0.8,
        height_m=0.61,
        writeable_area=writeable_area_coords,
    )
    wp.set_board_info(board_info)
    chars = fonts.write_text(
        text,
        font_name=font_name,
        font_size_mm=font_size_mm,
        pen_thickness_mm=pen_thickness_mm,
    )
    seq = await wp.write_characters(chars, space_factor)
    return seq


async def integration_test(node: Node, ctl: pp_control.PPControlBase) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    collision_service = node.create_client(
        SetFullCollisionBehavior,
        '/service_server/set_full_collision_behavior')
    if not collision_service.wait_for_service(timeout_sec=5.0):
        logger.info('Service SetFullCollisionBehavior not there.')
        return

    # Spawn and grab pen
    await ctl.add_fixed_pen()
    # await ctl.add_demo_board()

    logger.info('Robot approaching the pen.')
    pen_pose = np.array([0.45, 0.2, 0.03])
    pen_rot = R.from_euler('xyz', [180, 0, 0], degrees=True)
    pen_ori = pen_rot.as_quat()
    pre_grasp_pos = pen_pose + np.array([0, 0, 0.10])

    try:
        wait_t = 5.0
        logger.info(f'Waiting {wait_t} seconds...')
        await asyncio.sleep(wait_t)
        logger.info('Starting pen grabbing...')
        await ctl.configure()

        logger.info('Robot moving to pre grasp position.')
        goal = await ctl.move_to_ee_pose(pre_grasp_pos,
                                         pen_ori,
                                         execute_immediately=True)
        res = await goal.get_result_async()
        if res.result.error_code.val != 1:
            return

        await ctl.gripper_move(0.025)
        point_data = np.hstack([pen_pose, pen_ori, np.array([0])])
        traj_approach = Trajectory('pen_grab', point_data.reshape(1, 8))
        await ctl._execute_trajectory(traj_approach, 0.01)
        await ctl.gripper_grasp(0.005)
        await ctl.attach_pen()
        await asyncio.sleep(5.0)

        # Set up SetTCPFrame
        tcp_matrix = ee_change_matrix()
        logger.info('Calling SetTCPFrame service')
        frame_service = node.create_client(SetTCPFrame,
                                           '/service_server/set_tcp_frame')
        if not frame_service.wait_for_service(timeout_sec=5.0):
            logger.info('Service SetTCPFrame not there.')
            return

        req = SetTCPFrame.Request()
        req.transformation = tcp_matrix

        await frame_service.call_async(req)

        lift_pos = pen_pose + np.array([0, 0, 0.05])
        await ctl.move_to_ee_pose(lift_pos, pen_ori)
        await ctl.plan_to_named_config(
            named_config='ready',
            execute_immediately=True,
        )

        wait_t = 2.0
        logger.info(f'Waiting {wait_t} seconds...')
        await asyncio.sleep(wait_t)
        logger.info('Starting integration test...')
        await ctl.configure()

        demo_board_pose = np.array([0.5, 0.0, 0.6])
        demo_board_rot = R.from_euler('xyz', [180, 0, 0], degrees=True)
        buffer = 0.05
        start_pose = calculate_start_pose(buffer,
                                          demo_board_pose,
                                          demo_board_rot)
        speed = 0.01
        await asyncio.sleep(2.0)
        node.get_logger().info('Moving to start position')
        goal_handle = await ctl.move_to_ee_pose(
            goal_ee_position=start_pose[:3],
            goal_ee_orientation=start_pose[3:],
            execute_immediately=True)
        res = await goal_handle.get_result_async()
        if res.result.error_code.val != 1:
            node.get_logger().error(f'Error: {res.result.error_code.val}')
            return

        seq = get_demo_traj_sequence(start_pose)
        # Define Treshold - Orange Zone
        high_req = SetFullCollisionBehavior.Request()
        high_req.lower_torque_thresholds_nominal = [20.0, 20.0, 20.0, 20.0,
                                                    20.0, 20.0, 20.0]
        high_req.upper_torque_thresholds_nominal = [80.0, 80.0, 80.0, 80.0,
                                                    80.0, 80.0, 80.0]
        high_req.lower_force_thresholds_nominal = [8.0, 5.0, 5.0, 5.0,
                                                   5.0, 5.0]
        high_req.upper_force_thresholds_nominal = [80.0, 80.0, 80.0,
                                                   80.0, 80.0, 80.0]
        logger.info('Setting Orange Zone Thresholds Higher for Writing')
        await collision_service.call_async(high_req)
        await asyncio.sleep(2.0)

        # publish trajectories one by one
        for traj in seq:
            logger.info(f'Executing trajectory {traj.label}...')
            await ctl.execute_trajectory(traj, speed, publish_markers=True)
        logger.info('Publishing all trajectory markers...')
        for traj in seq:
            await ctl.publish_marker(traj)

        # After Writing threshold
        low_req = SetFullCollisionBehavior.Request()
        low_req.lower_torque_thresholds_nominal = [20.0, 20.0, 20.0, 20.0,
                                                   15.0, 15.0, 15.0]
        low_req.upper_torque_thresholds_nominal = [20.0, 20.0, 20.0, 20.0,
                                                   15.0, 15.0, 15.0]
        low_req.lower_force_thresholds_nominal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        low_req.upper_force_thresholds_nominal = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        logger.info('Setting Orange Zone Threshold Lower.')
        await collision_service.call_async(low_req)
        for traj in seq:
            logger.info(f'Executing trajectory {traj.label}...')
            # markers already published for now so no need to republish here
            await ctl.execute_trajectory(traj, speed, publish_markers=False)

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
