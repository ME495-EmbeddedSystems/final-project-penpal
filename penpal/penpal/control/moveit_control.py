"""Controller implementation using MoveIt. Lacks force control."""

import asyncio

from geometry_msgs.msg import Pose, Quaternion, PoseStamped

from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    RobotState,
    AttachedCollisionObject,
    CollisionObject,
    ObjectColor,
)
from moveit_msgs.msg import PlanningScene as PS
from moveit_msgs.srv import GetCartesianPath

from std_msgs.msg import ColorRGBA

import numpy as np

from penpal.control.pp_control import PPControlBase, PPControlError, Trajectory

from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from shape_msgs.msg import SolidPrimitive


class MoveItPPControl(PPControlBase):
    """Controller implementation using MoveIt. Lacks force control."""

    def __init__(
        self, node: Node, cfg: PPControlBase.Config | None = None
    ) -> None:
        """Initialize the object."""
        super().__init__(node, cfg)
        self._cbgroup = MutuallyExclusiveCallbackGroup()
        self._c_move_group = ActionClient(
            node, MoveGroup, '/move_action', callback_group=self._cbgroup
        )
        self._c_execute_trajectory = ActionClient(
            node,
            ExecuteTrajectory,
            '/execute_trajectory',
            callback_group=self._cbgroup,
        )
        self._logger = node.get_logger().get_child('MoveItPPControl')
        self._c_cartesian_path = self._node.create_client(
            GetCartesianPath,
            'compute_cartesian_path',
            callback_group=self._cbgroup,
        )
        self._scene_pub = self._node.create_publisher(
            PS, '/planning_scene', 10
        )
        self._board_sub = self._node.create_subscription(
            PoseStamped, 'whiteboard_pose', self.board_cb, 10
        )
        self._board_pose = None

    def board_cb(self, msg) -> None:
        """Execute board callback."""
        self._board_pose = msg

    async def _execute_trajectory(
        self,
        traj: Trajectory,
        target_ee_velocity_m_s: float,
    ) -> None:
        """
        Move the EE through a trajectory.

        Args:
            traj (Trajectory): path to send the EE through space
            target_ee_velocity_m_s (float): target average velocity
            for the trajectory execution.

        """
        pose_only = traj.data[:, :7]
        await self.plan_cartesian_path(pose_only, None, True)

    async def grip(
        self, offset_m: float, grip_force_N: float | None = None
    ) -> None:
        """
        Open or close the gripper to the desired offset, then applies a force.

        Args:
            offset_m: Offset (meters) of each finger from the EE frame.
            grip_force_N: Force to apply once gripped (i.e. to the marker when closed).
            If None, don't control the force.

        """
        goal_msg = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = 'hand'
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1

        constraints = Constraints()
        constraints.joint_constraints = []
        for joint in ['fer_finger_joint1', 'fer_finger_joint2']:
            jc = JointConstraint()
            jc.joint_name = joint
            jc.position = offset_m
            jc.tolerance_above = 0.005
            jc.tolerance_below = 0.005
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request.goal_constraints = [constraints]
        goal_msg.request = request
        self._logger.info(
            f'GRIPPING to {offset_m}: Sending goal to /move_action...'
        )
        response_goal_handle = await self._c_move_group.send_goal_async(
            goal_msg
        )
        self._logger.info(
            f'Received response goal handle: {response_goal_handle.accepted}'
        )
        self._logger.info('Awaiting the result')
        response = await response_goal_handle.get_result_async()
        self._logger.debug(f'Received the result: {response}')
        self._logger.info('Returning the result')

        return response.result

    async def move_to_ee_pose(
        self,
        goal_ee_position: np.ndarray | None,
        goal_ee_orientation: np.ndarray | None,
        start_joints: np.ndarray | None = None,
        execute_immediately: bool = False,
    ) -> ClientGoalHandle | None:
        """
        Move from a specified end-effector configuration to another.

        Args:
        ----
        goal_ee_position (np.ndarray): end EE position [x,y,z]. If not
              specified, any position is allowed such that the given
              orientation is achieved.
        goal_ee_orientation (np.ndarray): end EE orientation [x,y,z,w]
              quaternion. If not specified, any orientation is allowed
              such that the given position is achieved.
        start_joints (np.ndarray): array of joint angles for each joint.
              If not given, use current robot pose as start.
        execute_immediately (bool): immediately execute the path.

        """
        if goal_ee_orientation is None and goal_ee_position is None:
            raise ValueError(
                'One of orientation and position must be specified.'
            )
        goal_msg = MoveGroup.Goal()
        request = MotionPlanRequest()
        goal_constraint = Constraints()
        request.group_name = 'fer_manipulator'
        request.num_planning_attempts = 10
        request.allowed_planning_time = 20.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1

        if goal_ee_position is not None:
            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = 'base'
            pos_constraint.link_name = 'fer_hand_tcp'
            pos_constraint.target_point_offset.x = 0.0
            pos_constraint.target_point_offset.y = 0.0
            pos_constraint.target_point_offset.z = 0.0
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.01, 0.01, 0.01]
            pos_constraint.constraint_region = BoundingVolume()
            pos_constraint.constraint_region.primitives.append(box)  # type: ignore

            goal_box_pose = Pose()
            goal_box_pose.position.x = goal_ee_position[0]
            goal_box_pose.position.y = goal_ee_position[1]
            goal_box_pose.position.z = goal_ee_position[2]
            goal_box_pose.orientation.w = 1.0
            pos_constraint.weight = 1.0
            pos_constraint.constraint_region.primitive_poses.append(  # type: ignore
                goal_box_pose
            )
            goal_constraint.position_constraints.append(pos_constraint)  # type: ignore

        if goal_ee_orientation is not None:
            orient_constraint = OrientationConstraint()
            orient_constraint.header.frame_id = 'base'
            orient_constraint.link_name = 'fer_hand_tcp'
            q = Quaternion()
            q.x = goal_ee_orientation[0]
            q.y = goal_ee_orientation[1]
            q.z = goal_ee_orientation[2]
            q.w = goal_ee_orientation[3]
            orient_constraint.orientation = q
            orient_constraint.absolute_x_axis_tolerance = 0.2
            orient_constraint.absolute_y_axis_tolerance = 0.2
            orient_constraint.absolute_z_axis_tolerance = 0.2
            orient_constraint.weight = 1.0
            goal_constraint.orientation_constraints.append(orient_constraint)
            # type: ignore

        request.goal_constraints = [goal_constraint]
        goal_msg.request = request
        planning_options = PlanningOptions()
        planning_options.plan_only = not execute_immediately
        goal_msg.planning_options = planning_options

        if start_joints is not None:
            request.start_state = self.start_state(start_joints)

        self._logger.info(
            f'Sending goal to /move_action (pos:{goal_ee_position},'
            f'ori:{goal_ee_orientation})...'
        )
        goal_handle = await self._c_move_group.send_goal_async(goal_msg)
        if goal_handle is None:
            self._logger.error(
                'Received response goal of None from send_goal_async.'
            )
        else:
            self._logger.info(
                f'Received response goal handle: {goal_handle.accepted}'
            )
        res_msg = await goal_handle.get_result_async()
        result = res_msg.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._logger.error(
                f'Move failed with error code: {result.error_code.val}'
            )
        else:
            self._logger.info('Move execution succeeded.')
        return goal_handle

    async def plan_cartesian_path(
        self,
        waypoints: np.ndarray,
        start_ee_pose: np.ndarray | None = None,
        execute_immediately: bool = False,
    ) -> GetCartesianPath.Response:
        """
        Plan a Cartesian path from any valid starting pose to a goal pose.

        Uses moveit_msgs/GetCartesianPath Service.

        Args:
        ----
        waypoints (np.ndarray): destination poses [[x,y,z,qx,qy,qz,qw], ...]
        start_ee_pose (np.ndarray): start pose [x,y,z,qx,qy,qz,qw].
        If not provided, use current robot pose as start pose.
        execute_immediately (bool): immediately execute the path.

        Return:
        ------
            GetCartesianPath_Response: response of moveit GetCartesianPath srv.

        """
        request = GetCartesianPath.Request()
        request.group_name = 'fer_manipulator'
        request.link_name = 'fer_hand_tcp'
        request.waypoints = []
        request.max_step = 0.01

        for goal_ee_pose in waypoints:
            goal_pose = Pose()
            goal_pose.position.x = goal_ee_pose[0]
            goal_pose.position.y = goal_ee_pose[1]
            goal_pose.position.z = goal_ee_pose[2]
            goal_pose.orientation.x = goal_ee_pose[3]
            goal_pose.orientation.y = goal_ee_pose[4]
            goal_pose.orientation.z = goal_ee_pose[5]
            goal_pose.orientation.w = goal_ee_pose[6]
            request.waypoints.append(goal_pose)

        if start_ee_pose is not None:
            request.start_state = self.start_state(start_ee_pose)

        self._logger.info('Waiting for /compute_cartesian_path service')
        while not self._c_cartesian_path.wait_for_service(timeout_sec=5.0):
            self._logger.warn('Still waiting for service')

        self._logger.info('Request Cartesian path from service')
        response = await self._c_cartesian_path.call_async(request)  # type: ignore

        if execute_immediately:
            # check that we successfully planned
            if response is None:
                self._logger.error(
                    'Cannot execute trajectory--response was None'
                )
            response: GetCartesianPath.Response
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                self._logger.error(
                    f'Cannot execute trajectory ({response.error_code})'
                )

            # execute the cartesian path
            request = ExecuteTrajectory.Goal()
            request.trajectory = response.solution
            resp = await self._c_execute_trajectory.send_goal_async(request)
            if resp is not None and resp.accepted:
                res = await resp.get_result_async()
                if res.result.error_code.val != MoveItErrorCodes.SUCCESS:
                    self._logger.error(
                        f'Cartesian path execution failed: {res}'
                    )
                    raise PPControlError('Cartesian Path execution failed.')
                self._logger.info('Cartesian path execution success!')
            else:
                self._logger.error('Failed to execute cartesian path.')
        return response

    def joint_constraints(self, joints):
        """Set goal state."""
        constraints = Constraints()
        joint_names = [
            'fer_joint1',
            'fer_joint2',
            'fer_joint3',
            'fer_joint4',
            'fer_joint5',
            'fer_joint6',
            'fer_joint7',
        ]
        for i, joint_value in enumerate(joints):
            jointconstraint = JointConstraint()
            jointconstraint.joint_name = joint_names[i]
            jointconstraint.position = float(joint_value)
            jointconstraint.tolerance_above = 0.01
            jointconstraint.tolerance_below = 0.01
            constraints.joint_constraints.append(jointconstraint)  # type: ignore
        return [constraints]

    async def plan_to_named_config(
        self,
        named_config: str,
        execute_immediately: bool = False,
    ) -> MoveGroup.Result | None:
        """
        Plan a path from any valid starting pose to a named configuration.

        This "named configuration" can be defined in an SRDF or a remembered
        from a previous call to moveit python library's remember_joint_values()
        method, per docs here:
        https://docs.ros.org/en/jade/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#af9c9fc79be7fee5c366102db427fb28b

        Args:
        ----
        named_config (str): Named configuration.
        start_ee_pose (np.ndarray, optional): start pose;
            if None, use current robot pose.
        execute_immediately (bool): immediately execute the path.

        Return:
        ------
            MoveGroup.Result: Result of the move action.

        """
        goal_msg = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = 'fer_manipulator'
        request.num_planning_attempts = 5
        request.allowed_planning_time = 10.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1

        named_states = {
            'ready': np.array(
                [
                    0.0,
                    -0.7853981633974483,
                    0.0,
                    -2.356194490192345,
                    0.0,
                    1.5707963267948966,
                    0.7853981633974483,
                ]
            ),
            'extended': np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.1,
                    0.0,
                    1.5707963267948966,
                    0.7853981633974483,
                ]
            ),
        }
        if named_config not in named_states:
            raise ValueError('No such named configuration.')

        goal_joints = named_states[named_config]

        request.goal_constraints = self.joint_constraints(goal_joints)

        goal_msg.request = request
        planning_options = PlanningOptions()
        planning_options.plan_only = not execute_immediately
        goal_msg.planning_options = planning_options
        self._logger.info('Sending goal')
        response_goal = await self._c_move_group.send_goal_async(goal_msg)
        if response_goal is None:
            self._logger.error('response_goal=None')
            return
        self._logger.info(
            f'Received response goal handle: {response_goal.accepted}'
        )
        response = await response_goal.get_result_async()
        if response is None:
            self._logger.error('response=None')
            return None
        return response.result

    async def add_demo_board(self) -> None:
        """Spawn a board from board_detector at hard_coded location."""
        board = CollisionObject()
        board.header.frame_id = 'base'
        board.id = 'board'
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.8, 0.61, 0.02]
        board_pose = Pose()
        board_pose.position.x = 0.6
        board_pose.position.y = 0.0
        board_pose.position.z = 0.4
        board_pose.orientation.w = 1.0

        board.primitive_poses = []
        board.primitive_poses.append(board_pose)
        board.operation = CollisionObject.ADD

        # Publihs the addition
        scene_msg = PS()
        scene_msg.world.collision_objects = []
        scene_msg.world.collision_objects.append(board)
        scene_msg.is_diff = True
        color_msg = ObjectColor()
        color_msg.id = 'demo_board'
        color_msg.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
        scene_msg.object_colors = []
        scene_msg.object_colors.append(color_msg)
        await asyncio.sleep(1.0)
        self._scene_pub.publish(scene_msg)
        self._logger.info('Board in planning scene.')

    async def add_board(self) -> None:
        """Spawn a board from board_detector provided location."""
        board = CollisionObject()
        board.header.frame_id = 'base'
        board.id = 'board'
        box = SolidPrimitive()
        box.type = SolidPrimitive.Box
        box.dimensions = [0.8, 0.61, 0.02]

        board_pose = self._board_pose.pose
        board.primitive_poses.append(board_pose)
        board.operation = CollisionObject.ADD

        # Publihs the addition
        scene_msg = PS()
        scene_msg.world.collision_objects.append(board)
        scene_msg.is_diff = True
        color_msg = ObjectColor()
        color_msg.id = 'board'
        color_msg.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
        scene_msg.object_colors.append(color_msg)
        await asyncio.sleep(1.0)
        self._scene_pub.publihs(scene_msg)
        self._logger.info('Demo board in planning scene.')

    async def add_fixed_pen(self) -> None:
        """Spawn a collision object pen at a hardcoded location."""
        pen = CollisionObject()
        pen.header.frame_id = 'base'
        pen.id = 'pen'
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [0.10, 0.01]  # [height, radius in meters]
        # Hard coded pen location
        pen_pose = Pose()
        pen_pose.position.x = 0.5
        pen_pose.position.y = 0.3
        pen_pose.position.z = 0.191
        pen_pose.orientation.x = 0.0
        pen_pose.orientation.y = 0.7071068
        pen_pose.orientation.z = 0.0
        pen_pose.orientation.w = 0.7071068
        pen.primitives.append(cylinder)
        pen.primitive_poses.append(pen_pose)
        pen.operation = CollisionObject.ADD

        # Publish the addition
        scene_msg = PS()
        scene_msg.world.collision_objects.append(pen)
        scene_msg.is_diff = True
        color_msg = ObjectColor()
        color_msg.id = 'pen'
        color_msg.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        scene_msg.object_colors.append(color_msg)
        await asyncio.sleep(1.0)
        self._scene_pub.publish(scene_msg)
        self._logger.info('Pen in planning scene.')

    async def attach_pen(self) -> None:
        """Attach the pen to the robot hand."""
        attached_pen = AttachedCollisionObject()
        attached_pen.link_name = 'fer_hand_tcp'
        attached_pen.object.id = 'pen'
        attached_pen.touch_links = [
            'fer_hand',
            'fer_left_finger',
            'fer_right_finger',
            'fer_hand_tcp',
        ]
        attached_pen.object.operation = CollisionObject.ADD

        scene_msg = PS()
        scene_msg.robot_state.attached_collision_objects.append(attached_pen)
        scene_msg.is_diff = True
        self._scene_pub.publish(scene_msg)
        self._logger.info('Attached pen to gripper.')

    def start_state(self, joints):
        """Set start state."""
        robotstate = RobotState()
        robotstate.joint_state.name = [
            'fer_joint1',
            'fer_joint2',
            'fer_joint3',
            'fer_joint4',
            'fer_joint5',
            'fer_joint6',
            'fer_joint7',
        ]
        if joints is None:
            robotstate.joint_state.position = [
                0.0,
                -0.7853981633974483,
                0.0,
                -2.356194490192345,
                0.0,
                1.5707963267948966,
                0.7853981633974483,
            ]
        else:
            robotstate.joint_state.position = [float(j) for j in joints]
        return robotstate
