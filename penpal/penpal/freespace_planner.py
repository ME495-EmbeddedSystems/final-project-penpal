"""Grabs the pen."""

import asyncio

from franka_msgs.srv import SetFullCollisionBehavior

import numpy as np

from penpal.constants import R_tcp_board
from penpal.control.moveit_control import MoveItPPControl
from penpal.control.pp_control import Trajectory
from penpal.write_planner import BoardInfo

from rclpy.node import Node

from scipy.spatial.transform import Rotation as R


class GrabError(Exception):
    """Base exception class for the GrabPlanner module."""

    pass


class FreespacePlanner:
    """Compute trajectories to grab the pen and move through free space."""

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
        await self.ctl.move_to_ee_pose(
            pre_grasp_pos, pen_ori, execute_immediately=True
        )

        await self.ctl.gripper_move(0.025)
        point_data = np.hstack([pen_pose, pen_ori, np.array([0])])
        traj_approach = Trajectory('pen_grab', point_data.reshape(1, 8))
        await self.ctl._execute_trajectory(traj_approach, 0.01)
        await self.ctl.gripper_grasp(0.0045)
        await self.ctl.attach_pen()
        await asyncio.sleep(1.0)

        # Set up SetTCPFrame
        tcp_matrix = self._get_ee_transform_matrix()
        await self.ctl.set_tcp_frame(tcp_matrix)

        lift_pos = pen_pose + np.array([0, 0, 0.05])
        await self.ctl.move_to_ee_pose(lift_pos, pen_ori)
        await self.ctl.plan_to_named_config(
            named_config='ready',
            execute_immediately=True,
        )
        wait_t = 1.0
        self._logger.info(f'Waiting {wait_t} seconds...')
        await asyncio.sleep(wait_t)

    def _get_ee_transform_matrix(self) -> np.ndarray:
        """
        Get matrix to move EE from tcp_hand to pen tip.

        Hardcoded to match pen dimensions.
        """
        T_final = np.eye(4)
        # T_final[0:3, 0:3] = R.from_euler('xy', (180, 90), True).as_matrix()
        T_final[0, 3] = 0.1  # The pen sticks out in the X direction.
        return T_final.flatten(order='F').tolist()

    async def home_arm(self) -> None:
        """Send the arm to the home position."""
        self._logger.info("Homing the arm to 'ready' position...")
        await self.ctl.plan_to_named_config('ready', execute_immediately=True)

    async def reset_gripper(self) -> None:
        """Open the gripper fully."""
        self._logger.info('Resetting gripper to open position...')
        await self.ctl.reset_gripper()

    def _calculate_start_pose(
        self,
        buffer: float,
        board_pose_position: np.ndarray,
        board_pose_rotation: R,
    ) -> np.ndarray:
        """
        Calculate the start pose for writing.

        Such that:
        - +X_tcp is aligned with the board normal (+Z_board)
        - TCP is 'buffer' meters in front of the board along -normal.
        """
        # Board normal in BOARD frame and in WORLD frame
        board_normal_board = np.array([0.0, 0.0, 1.0])
        world_normal = board_pose_rotation.apply(board_normal_board)

        # World -> TCP
        target_rot = board_pose_rotation * R_tcp_board

        # Put TCP 'buffer' meters in front of the board
        target_position = board_pose_position + world_normal * buffer

        q = target_rot.as_quat(True)
        start_pose = np.array([*target_position, *q])
        return start_pose

    async def move_to_board(
        self, board_info: BoardInfo, off_board_height_m: float
    ) -> None:
        """Move the pen to hover slightly above the board."""
        demo_board_pose = board_info.pos
        demo_board_rot = board_info.ori
        buffer = off_board_height_m
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
        # await self.ctl.set_collision_thresholds(free_space_req)

        self._logger.info('Moving to start position')
        await self.ctl.move_to_ee_pose(
            goal_ee_position=start_pose[:3],
            goal_ee_orientation=start_pose[3:],
            execute_immediately=True,
        )
