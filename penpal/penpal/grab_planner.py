"""Grabs the pen."""

import asyncio

import numpy as np
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node

from penpal.control.moveit_control import MoveItPPControl
from penpal.control.pp_control import Trajectory
from franka_msgs.srv import SetFullCollisionBehavior


class GrabError(Exception):
    """Base exception class for the GrabPlanner module."""

    pass


class GrabPlanner:
    """Compute trajectories to write on the real board."""

    def __init__(self, node: Node, controller: MoveItPPControl) -> None:
        """Initialize the object."""
        self.ctl = controller
        self._node = node
        self._logger = node.get_logger().get_child('GrabPlanner')

    async def grab_pen(self) -> None:
        """Grab the pen (must be visible to camera)."""
        # add the pen to planning scene
        # TODO grab location from tf tree
        await self.ctl.add_fixed_pen()

        pen_pose = np.array([0.45, 0.2, 0.03])
        pen_rot = R.from_euler('xyz', [180, 0, 0], degrees=True)
        pen_ori = pen_rot.as_quat(True)
        pre_grasp_pos = pen_pose + np.array([0, 0, 0.10])

        self._logger.info('Starting pen grabbing...')
        await self.ctl.configure()

        self._logger.info('Robot moving to pre grasp position.')
        goal = await self.ctl.move_to_ee_pose(
            pre_grasp_pos, pen_ori, execute_immediately=True
        )
        res = await goal.get_result_async()  # type: ignore
        if res.result.error_code.val != 1:  # type: ignore
            raise GrabError(
                f'Pre-grasp position error code {res.result.error_code.val}'
            )

        await self.ctl.gripper_move(0.025)
        point_data = np.hstack([pen_pose, pen_ori, np.array([0])])
        traj_approach = Trajectory('pen_grab', point_data.reshape(1, 8))
        await self.ctl._execute_trajectory(traj_approach, 0.01)
        await self.ctl.gripper_grasp(0.005)
        await self.ctl.attach_pen()
        await asyncio.sleep(3.0)

        # Set up SetTCPFrame
        tcp_matrix = self._ee_change_matrix()
        await self.ctl.set_tcp_frame(tcp_matrix)

        lift_pos = pen_pose + np.array([0, 0, 0.05])
        await self.ctl.move_to_ee_pose(lift_pos, pen_ori)
        await self.ctl.plan_to_named_config(
            named_config='ready',
            execute_immediately=True,
        )
        wait_t = 5.0
        self._logger.info(f'Waiting {wait_t} seconds...')
        await asyncio.sleep(wait_t)

    def _ee_change_matrix(self) -> np.ndarray:
        """Set matrix to move EE from tcp_hand to pen tip."""
        T_final = np.eye(4)
        T_final[2, 3] = 0.1
        return T_final.flatten(order='F').tolist()

    async def home_arm(self) -> None:
        """Send the arm to the home position."""
        self._logger.info("Homing the arm to 'ready' position...")
        await self.ctl.gripper_move(0.025)
        await self.ctl.plan_to_named_config('ready', execute_immediately=True)

    def _calculate_start_pose(
        self, buffer, board_pose_position: np.ndarray, board_pose_rotation: R
    ):
        """Calculate the start pose to place the pen tip normal to the board."""
        board_normal = np.array([1, 0, 0])
        world_normal_vector = board_pose_rotation.apply(board_normal)
        current_pen_direction = np.array([1, 0, 0])
        desired_pen_direction = world_normal_vector
        R_align, _ = R.align_vectors(
            [desired_pen_direction], [current_pen_direction]
        )
        R_flip = R.from_euler('x', 180, degrees=True)
        target_rot = R_align * R_flip
        target_position = board_pose_position - (world_normal_vector * buffer)
        target_orientation_quat = target_rot.as_quat(True)
        start_pose = np.array([*target_position, *target_orientation_quat])
        return start_pose

    async def move_to_board(self) -> None:
        """Move the pen to hover slightly above the board."""
        demo_board_pose = np.array([0.5, 0.0, 0.6])
        demo_board_rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
        buffer = 0.05
        start_pose = self._calculate_start_pose(
            buffer, demo_board_pose, demo_board_rot
        )
        self._logger.info('Setting collision behavior for move to board...')
        free_space_req = SetFullCollisionBehavior.Request()
        free_space_req.upper_torque_thresholds_nominal = [
            60.0,
            60.0,
            60.0,
            60.0,
            50.0,
            50.0,
            50.0,
        ]
        free_space_req.upper_force_thresholds_nominal = [
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
        ]
        free_space_req.lower_torque_thresholds_nominal = [
            50.0,
            50.0,
            50.0,
            50.0,
            40.0,
            40.0,
            40.0,
        ]
        free_space_req.lower_force_thresholds_nominal = [
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
        await self.ctl.set_collision_thresholds(free_space_req)

        self._logger.info('Moving to start position')
        goal_handle = await self.ctl.move_to_ee_pose(
            goal_ee_position=start_pose[:3],
            goal_ee_orientation=start_pose[3:],
            execute_immediately=True,
        )
        res = await goal_handle.get_result_async()
        if res.result.error_code.val != 1:
            self._logger.error(f'Failed, Error: {res.result.error_code.val}')
            return
