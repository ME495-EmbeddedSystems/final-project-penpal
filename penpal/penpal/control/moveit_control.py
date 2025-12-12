"""Controller implementation using MoveIt. Lacks force control."""

import asyncio
from typing import Any

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
from franka_msgs.srv import SetFullCollisionBehavior
from moveit_msgs.msg import PlanningScene as PS
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.srv import ApplyPlanningScene
from std_msgs.msg import ColorRGBA
from franka_msgs.action import Grasp, Move
import numpy as np

from penpal.control.pp_control import PPControlBase, PPControlError, Trajectory

from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from franka_msgs.srv import SetTCPFrame

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
        self._c_franka_move = ActionClient(
            node, Move, '/fer_gripper/move', callback_group=self._cbgroup
        )
        self._c_franka_grasp = ActionClient(
            node, Grasp, '/fer_gripper/grasp', callback_group=self._cbgroup
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
        self._ps_client = self._node.create_client(
            ApplyPlanningScene,
            '/apply_planning_scene',
            callback_group=self._cbgroup,
        )
        self._board_pose = None

        self._c_collision = node.create_client(
            SetFullCollisionBehavior,
            '/service_server/set_full_collision_behavior',
        )
        self._c_tcpframe = node.create_client(
            SetTCPFrame, '/service_server/set_tcp_frame'
        )

    async def configure(self) -> None:
        """One-time setup to use the controller."""
        await super().configure()
        if not self._c_collision.wait_for_service(timeout_sec=5.0):
            msg = 'Service SetFullCollisionBehavior not there.'
            self._logger.error(msg)
            raise PPControlError(msg)

        high_req = SetFullCollisionBehavior.Request()
        high_req.lower_torque_thresholds_nominal = [
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
        ]
        high_req.upper_torque_thresholds_nominal = [
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
        ]
        high_req.lower_force_thresholds_nominal = [
            8.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        high_req.upper_force_thresholds_nominal = [
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
        ]
        self._logger.info('Setting Orange Zone Thresholds Higher for Writing')
        await self.set_collision_thresholds(high_req)

        # wait for everything to boot up
        # await asyncio.sleep(3.0)

    async def send_goal_async(
        self,
        client: ActionClient,
        goal: Any,
        action_desc: str,
        raise_on_fail: bool = True,
    ) -> bool:
        """
        Send a goal to an action server. Handle errors loudly.

        Args:
            client (ActionClient): action client
            goal (Any): action goal request
            action_desc (str): string description of what this action is
            raise_on_fail (bool, optional): raise an exception on failure.

        Returns:
            bool: True if successful, false otherwise.

        """
        errmsg = f'Action {action_desc}: '
        handle = await client.send_goal_async(goal)
        if handle is None:
            errmsg += 'handle is None.'
            self._logger.error(errmsg)
            if raise_on_fail:
                raise PPControlError(errmsg)
            else:
                return False
        response = await handle.get_result_async()
        if response is None:
            errmsg += 'response is None.'
            self._logger.error(errmsg)
            if raise_on_fail:
                raise PPControlError(errmsg)
            else:
                return False
        result = response.result
        errcode = getattr(result, 'error_code', None)
        if errcode is not None:
            errcode = errcode.val
        success = getattr(result, 'success', None)
        # self._logger.info(f'RESULT: {result}')
        if (errcode is not None and errcode != MoveItErrorCodes.SUCCESS) or (
            success is not None and not success
        ):
            errmsg += f'Failed result (err={errcode} success={success})'
            self._logger.error(errmsg)
            if raise_on_fail:
                raise PPControlError(errmsg)
            else:
                return False
        else:
            errmsg += 'Success!'
            self._logger.info(errmsg)
            return True

    async def set_tcp_frame(self, T_en: np.ndarray) -> None:
        """
        Set the transformation from the EE to NE frame.

        Args:
            T_en (np.ndarray): _description_

        """
        self._logger.info('Calling SetTCPFrame service')
        if not self._c_tcpframe.wait_for_service(timeout_sec=5.0):
            self._logger.info('Service SetTCPFrame not there.')
            return

        req = SetTCPFrame.Request()
        req.transformation = T_en

        await self._c_tcpframe.call_async(req)

    async def set_collision_thresholds(
        self, req: SetFullCollisionBehavior.Request
    ) -> None:
        """Set the collision thresholds of the franka arm."""
        self._logger.info('Setting collision thresholds...')
        await asyncio.sleep(2.0)
        await self._c_collision.call_async(req)
        await asyncio.sleep(2.0)

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
        await self.plan_cartesian_path(pose_only, None, True, 0.1, 0.1)

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
        await self.send_goal_async(
            self._c_franka_grasp, goal_msg, f'Gripping to offset {offset_m}m'
        )

    async def reset_gripper(self) -> None:
        """Reset the gripper to fully open."""
        # remove the pen marker if it's present
        await self.remove_pen()
        await self.gripper_move(0.025)

    async def gripper_move(self, width: float, speed: float = 0.04) -> bool:
        """
        Move the gripper out to the desired offset.

        Args:
            width: Offset (meters) of each finger from the EE frame.
            speed: speed of gripper opening.

        """
        goal = Move.Goal()
        width = float(width)
        speed = float(speed)
        MostClosed = 0.0
        MostOpen = 0.1
        amount = min(MostOpen, max(width, MostClosed))
        goal.width = amount
        goal.speed = speed
        return await self.send_goal_async(
            self._c_franka_move, goal, f'Gripper move to {amount}m'
        )

    async def gripper_grasp(self, width: float, speed: float = 0.04) -> bool:
        """
        Move the gripper in to the desired offset.

        Args:
            width: Offset (meters) of each finger from the EE frame.
            speed: speed of gripper opening.

        """
        goal = Grasp.Goal()
        width = float(width)
        speed = float(speed)

        MostClosed = 0.0
        MostOpen = 0.1
        amount = min(MostOpen, max(width, MostClosed))
        goal.width = amount
        goal.speed = speed
        return await self.send_goal_async(
            self._c_franka_grasp, goal, f'Gripper grasp to {amount}m'
        )

    async def move_to_ee_pose(
        self,
        goal_ee_position: np.ndarray | None,
        goal_ee_orientation: np.ndarray | None,
        start_joints: np.ndarray | None = None,
        execute_immediately: bool = False,
    ) -> bool:
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
            orient_constraint.absolute_x_axis_tolerance = 0.05
            orient_constraint.absolute_y_axis_tolerance = 0.05
            orient_constraint.absolute_z_axis_tolerance = 0.05
            orient_constraint.weight = 1.0
            goal_constraint.orientation_constraints = []
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
        return await self.send_goal_async(
            self._c_move_group, goal_msg, 'move to EE pose'
        )

    async def plan_cartesian_path(
        self,
        waypoints: np.ndarray,
        start_ee_pose: np.ndarray | None = None,
        execute_immediately: bool = False,
        velocity_scale: float = 0.5,
        accel_scale: float = 0.5,
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
        request.max_velocity_scaling_factor = velocity_scale
        request.max_acceleration_scaling_factor = accel_scale

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

            await self.send_goal_async(
                self._c_execute_trajectory, request, 'Execute cartesian path'
            )
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
    ) -> bool:
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

        return await self.send_goal_async(
            self._c_move_group, goal_msg, f'Move to name {named_config}'
        )

    async def add_static_scene_collision_objects(self) -> None:
        """Add the necessary collision objects for safety (ie table)."""
        fudge = 2.0
        table_length_m = 3 * fudge
        table_width_m = 2 * fudge
        table_height_m = 4 * fudge  # approximate
        table = CollisionObject()
        table.header.frame_id = 'base'
        table.id = 'table'
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [
            table_length_m,
            table_width_m,
            table_height_m,
        ]  # [x, y, z]
        # Hard coded pen location
        table_pose = Pose()
        table_pose.position.x = 0.0
        table_pose.position.y = 0.0
        table_pose.position.z = -table_height_m / 2
        table_pose.orientation.x = 0.0
        table_pose.orientation.y = 0.0
        table_pose.orientation.z = 0.0
        table_pose.orientation.w = 1.0
        table.primitives = []
        table.primitives.append(box)
        table.primitive_poses = []
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        # Publish the addition
        scene_msg = PS()
        scene_msg.world.collision_objects = []
        scene_msg.world.collision_objects.append(table)
        scene_msg.is_diff = True
        color_msg = ObjectColor()
        color_msg.id = 'table'
        color_msg.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=1.0)
        scene_msg.object_colors = []
        scene_msg.object_colors.append(color_msg)
        self._scene_pub.publish(scene_msg)
        self._logger.info('Table in planning scene.')

    async def add_fixed_pen(self) -> None:
        """Spawn a collision object pen at a hardcoded location."""
        pen = CollisionObject()
        pen.header.frame_id = 'base'
        pen.id = 'pen'
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [0.10, 0.009]  # [height, radius in meters]
        # Hard coded pen location
        pen_pose = Pose()
        pen_pose.position.x = 0.45
        pen_pose.position.y = 0.2
        pen_pose.position.z = 0.02
        pen_pose.orientation.x = 0.0
        pen_pose.orientation.y = 0.7071068
        pen_pose.orientation.z = 0.0
        pen_pose.orientation.w = 0.7071068
        pen.primitives = []
        pen.primitives.append(cylinder)
        pen.primitive_poses = []
        pen.primitive_poses.append(pen_pose)
        pen.operation = CollisionObject.ADD

        # Publish the addition
        scene_msg = PS()
        scene_msg.world.collision_objects = []
        scene_msg.world.collision_objects.append(pen)
        scene_msg.is_diff = True
        color_msg = ObjectColor()
        color_msg.id = 'pen'
        color_msg.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        scene_msg.object_colors = []
        scene_msg.object_colors.append(color_msg)
        # await asyncio.sleep(1.0)
        self._scene_pub.publish(scene_msg)
        self._logger.info('Pen in planning scene.')

    async def remove_pen(self) -> None:
        """Remove the pen from the planning scene."""
        pen = CollisionObject()
        pen.header.frame_id = 'base'
        pen.id = 'pen'
        pen.operation = CollisionObject.REMOVE

        # Publish the removal
        scene_msg = PS()
        scene_msg.world.collision_objects = []
        scene_msg.world.collision_objects.append(pen)
        scene_msg.is_diff = True
        self._scene_pub.publish(scene_msg)
        self._logger.info('Removed pen from planning scene.')

    async def attach_pen(self) -> None:
        """Attach the pen to the robot hand."""
        attached_pen = AttachedCollisionObject()
        attached_pen.link_name = 'fer_hand_tcp'
        attached_pen.object.id = 'pen'
        attached_pen.touch_links = [
            'fer_hand',
            'fer_leftfinger',
            'fer_rightfinger',
            'fer_hand_tcp',
        ]
        attached_pen.object.operation = CollisionObject.ADD

        scene_msg = PS()
        scene_msg.robot_state.attached_collision_objects = []
        scene_msg.robot_state.attached_collision_objects.append(attached_pen)
        scene_msg.is_diff = True
        self._scene_pub.publish(scene_msg)
        self._logger.info('Attached pen to gripper.')

    def publish_destination_pose(self, pose: np.ndarray) -> None:
        """Publish a pose arrow."""

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
